from dotenv import load_dotenv
load_dotenv()

import click
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from repo2readme.config import reset_api_keys
import os
from collections import Counter

from repo2readme import __version__
from repo2readme.utils.logging_config import logging_options
from repo2readme.utils.tree import generate_tree_from_paths
from repo2readme.cache import (
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_MAX_ENTRIES,
    SummaryCache,
)
from repo2readme.loaders.repo_loader import RepoLoader
from repo2readme.summarize.summary import get_prompt_template_hash
from repo2readme.dependency_graph import build_dependency_graph
from repo2readme.providers import PROVIDERS, provider_choices_help

# Import new services
from repo2readme.services.cache_admin import (
    build_info_lines,
    cache_info,
    clear_cache,
    default_cache_dir,
    prune_cache,
)
from repo2readme.services.environment import setup_api_keys
from repo2readme.services.estimation import format_size, estimate_analysis_cost
from repo2readme.services.summarization import generate_all_summaries, generate_hierarchical_summaries
from repo2readme.services.orchestrator import run_pipeline
from repo2readme.services.reporting import partition_summaries, render_report


def cache_namespace(url: str | None, local: str | None) -> str:
    """Identify the repository a cache entry belongs to.

    A URL identifies itself. A local path is made absolute, so ``--local .``
    and ``--local /home/me/project`` from inside that project are one
    repository rather than two.
    """
    if url:
        return str(url).strip()
    return os.path.abspath(os.path.expanduser(str(local)))


@click.group()
@click.version_option(version=__version__, prog_name="repo2readme")
def main():
    """
    Use the `run` command to generate a README.
    Use the `reset` command to clear saved API keys.

    Note: First run will ask for your API keys.
    """


@main.command()
@logging_options
@click.option(
    "--url",
    "-u",
    help="Git repository URL (https, ssh, git:// or git@host:path).",
)
@click.option("--local", "-l", help="Local repo path")
@click.option("--output", "-o", default=None, type=click.Path(), help="Save README to file")
@click.option("--force", "-f", is_flag=True, help="Overwrite output file without confirmation")
@click.option(
    "--include",
    "include_patterns",
    multiple=True,
    help="Glob pattern for files to include even if ignored by default. Can be used multiple times.",
)
@click.option(
    "--exclude",
    "exclude_patterns",
    multiple=True,
    help="Glob pattern for files to exclude. Can be used multiple times.",
)
@click.option(
    "--max-file-size-kb",
    default=200,
    show_default=True,
    type=int,
    help="Maximum file size in KB to include during repository analysis.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview the analysis without making any API calls.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Exit with a non-zero status if any file fails to summarize.",
)
@click.option(
    "--respect-gitignore",
    is_flag=True,
    default=False,
    help="Respect .gitignore and .git/info/exclude patterns during repository traversal",
)
@click.option(
    "--max-workers",
    default=None,
    type=int,
    help="Number of parallel worker threads for file processing (default: 4, capped at file count)",
)
@click.option(
    "--cache-max-entries",
    default=DEFAULT_MAX_ENTRIES,
    show_default=True,
    type=int,
    help="Keep at most this many cached summaries, dropping the least recently used.",
)
@click.option(
    "--cache-max-age-days",
    default=DEFAULT_MAX_AGE_DAYS,
    show_default=True,
    type=float,
    help="Drop cached summaries older than this many days.",
)
@click.option(
    "--provider",
    default=None,
    help=f"LLM provider ({provider_choices_help()}). Run 'repo2readme providers' for details.",
)
@click.option(
    "--model",
    default=None,
    help="LLM model name",
)
@click.option(
    "--base-url",
    default=None,
    help="Base URL for OpenAI-compatible providers",
)
@click.option(
    "--branch",
    "-b",
    default="main",
    show_default=True,
    help="Branch to clone when using --url.",
)
def run(url, local, output, force, include_patterns, exclude_patterns, max_file_size_kb, dry_run, strict, respect_gitignore, max_workers, cache_max_entries, cache_max_age_days, provider, model, base_url, branch):
    """ Use --url for GitHub repo url and --local for local repo
    """
    if not url and not local:
        rprint("[red]Provide either --url or --local[/red]")
        return

    source = url if url else local

    # Initialize file summary cache
    cache_dir = default_cache_dir()
    summarization_config = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
    }
    # autosave=False batches the writes: the whole cache file has to be
    # rewritten for any change, so saving once per summarized file made a run
    # cost one full serialization per file. The run flushes once at the end,
    # in the finally block, so an interrupted run still keeps its work.
    #
    # The namespace is what this run is analysing. The cache directory is the
    # current working directory rather than the repository, so runs against
    # several repositories from one place share a file; without a namespace
    # each run saw the others' entries as deleted files and evicted them, and
    # two repositories analysed from the same directory never got a cache hit.
    summary_cache = SummaryCache(
        cache_dir=cache_dir,
        config=summarization_config,
        prompt_template_hash=get_prompt_template_hash(),
        autosave=False,
        namespace=cache_namespace(url, local),
        max_entries=cache_max_entries,
        max_age_days=cache_max_age_days,
    )

    with Progress() as progress:
        task = progress.add_task("[cyan]Loading repository...", total=1)
        try:
            loader = RepoLoader(
                source,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                max_file_size_kb=max_file_size_kb,
                respect_gitignore=respect_gitignore,
                max_workers=max_workers,
                branch=branch,
            )
            if dry_run:
                files, root_path, loader_obj, skipped = loader.load(return_skip_info=True)
            else:
                files, root_path, loader_obj = loader.load()
                skipped = []
        except Exception as e:
            rprint(f"[red]Failed to load repository: {e}[/red]")
            return
        progress.update(task, advance=1)

    documents = []
    for f in files:
        documents.append({
            "content": f.page_content,
            "metadata": f.metadata
        })
    
    # Build dependency graph for README enrichment
    dependency_graph = build_dependency_graph(documents)
    dependency_overview = dependency_graph.to_markdown_summary()

    # Build the tree from the documents that were actually loaded, so the
    # "Folder Structure" section of the README cannot advertise files that were
    # filtered out of the analysis.
    tree = generate_tree_from_paths(
        root_path,
        [doc["metadata"].get("relative_path", "") for doc in documents],
    )

    estimated_tokens, total_size_bytes, total_documents = estimate_analysis_cost(documents)

    if dry_run:
        rprint("\n[bold]Repository Tree[/bold]\n")
        rprint(tree)
        rprint("\n[bold]Files to be processed[/bold]\n")
        for doc in documents:
            rel_path = doc["metadata"].get("relative_path", "")
            rprint(f"✓ [green]{rel_path}[/green]")
        if skipped:
            skip_reasons = Counter()
            for _, reason in skipped:
                skip_reasons[reason] += 1
            rprint("\n[bold]Skipped Files Summary[/bold]\n")
            reason_order = ["excluded by pattern", "ignored by default rules", "exceeds maximum file size", "protected large file"]
            printed = set()
            for reason in reason_order:
                if reason in skip_reasons:
                    printed.add(reason)
                    rprint(f"{reason:30s}: {skip_reasons[reason]}")
            for reason in sorted(skip_reasons):
                if reason not in printed:
                    rprint(f"{reason:30s}: {skip_reasons[reason]}")
        rprint("\n[bold]Repository Analysis[/bold]\n")
        rprint(f"Files selected     : {total_documents}")
        rprint(f"Estimated tokens   : ~{estimated_tokens:,}")
        rprint(f"Request size       : ~{format_size(total_size_bytes)}")
        rprint("\n[green]Dry run complete.[/green]")
        rprint("[yellow]No API requests were made.[/yellow]")
        if hasattr(loader_obj, "cleanup"):
            loader_obj.cleanup()
        return

    # Normal execution: print estimation first
    rprint("\n[bold]Repository Analysis[/bold]\n")
    rprint(f"Files to summarize : {total_documents}")
    rprint(f"Estimated tokens   : ~{estimated_tokens:,}")
    rprint(f"Request size       : ~{format_size(total_size_bytes)}")

    try:
        if not force:
            proceed = click.confirm("\nProceed?", default=False)
            if not proceed:
                rprint("[yellow]Operation cancelled.[/yellow]")
                return

        try:
            setup_api_keys(provider)
        except Exception as e:
            rprint(f"[red]Failed to configure API keys: {e}[/red]")
            return

        with Progress() as progress:
            task = progress.add_task("[cyan]Generating summaries...[/cyan]", total=total_documents)
            summaries, errors = generate_all_summaries(
                documents=documents,
                summary_cache=summary_cache,
                provider=provider,
                model=model,
                base_url=base_url,
                max_workers=max_workers,
                progress=progress,
                task_id=task
            )

        # Failure placeholders must not reach the roll-up or the README prompt,
        # so split them out before anything else consumes them.
        successful_summaries, failures = partition_summaries(summaries)
        failures.extend(errors)
        render_report(total_documents, len(successful_summaries), failures, rprint)

        if total_documents and not successful_summaries:
            rprint(
                "\n[red]Every file failed to summarize, so there is nothing to "
                "generate a README from.[/red]"
            )
            raise SystemExit(1)

        with Progress() as progress:
            rollup_task = progress.add_task("[cyan]Generating directory summaries...[/cyan]", total=1)
            hierarchical_summaries = generate_hierarchical_summaries(
                file_summaries=successful_summaries,
                provider=provider,
                model=model,
                base_url=base_url,
                progress=progress,
                task_id=rollup_task
            )

        # Remove cache entries for files that no longer exist
        current_files = {doc["metadata"]["file_path"] for doc in documents}
        stale_entries = summary_cache.get_deleted_files(current_files)
        if stale_entries:
            stale_paths = [e["file_path"] for e in stale_entries]
            summary_cache.remove_entries(stale_paths)

        rprint("[cyan]Generating README...[/cyan]")
        
        readme = run_pipeline(
            summaries=hierarchical_summaries,
            tree=tree,
            dependency_overview=dependency_overview,
            provider=provider,
            model=model,
            base_url=base_url
        )

        if output is None:
            rprint("\n[green]Generated README:[/green]\n")
            rprint(readme)
        else:
            if os.path.exists(output) and not force:
                should_overwrite = click.confirm(
                    f"{output} already exists. Do you want to overwrite it?",
                    default=False,
                )

                if not should_overwrite:
                    rprint("[yellow]Output file was not overwritten.[/yellow]")
                    return

            with open(output, "w", encoding="utf-8") as f:
                f.write(readme)

            rprint(f"[green]Saved to {output}[/green]")

        if strict and failures:
            rprint(
                f"[red]--strict: {len(failures)} file(s) failed to summarize.[/red]"
            )
            raise SystemExit(1)

    finally:
        # Apply the bounds before the write, so the file that lands on disk is
        # already the pruned one rather than growing until someone notices.
        pruned = summary_cache.prune()
        if pruned.removed:
            rprint(
                f"[dim]Cache: removed {pruned.removed:,} old entries, "
                f"{pruned.entries_after:,} remain.[/dim]"
            )

        # One write for the whole run, including when the run was interrupted
        # part way through.
        summary_cache.flush()
        if hasattr(loader_obj, "cleanup"):
            loader_obj.cleanup()


@main.command()
def providers():
    """List the supported LLM providers and their defaults."""
    table = Table(title="Supported providers", header_style="bold cyan")
    table.add_column("Provider")
    table.add_column("Aliases")
    table.add_column("Default model")
    table.add_column("API key env var")

    for spec in PROVIDERS:
        table.add_row(
            spec.name,
            ", ".join(spec.aliases) or "-",
            spec.default_model,
            spec.env_var or "not required",
        )

    console = Console()
    console.print(table)
    rprint(
        "\nUse with: [cyan]repo2readme run --local . --provider <name> "
        "[--model <model>][/cyan]"
    )


@main.group()
def cache():
    """Inspect and manage the summary cache."""


@cache.command("info")
@click.option(
    "--cache-dir",
    default=None,
    type=click.Path(),
    help="Cache directory to inspect. Defaults to ./.repo2readme/cache.",
)
def cache_info_command(cache_dir):
    """Show what the summary cache currently holds."""
    for line in build_info_lines(cache_info(cache_dir or default_cache_dir())):
        rprint(line)


@cache.command("prune")
@click.option(
    "--cache-dir",
    default=None,
    type=click.Path(),
    help="Cache directory to prune. Defaults to ./.repo2readme/cache.",
)
@click.option(
    "--max-entries",
    default=DEFAULT_MAX_ENTRIES,
    show_default=True,
    type=int,
    help="Keep at most this many entries, dropping the least recently used.",
)
@click.option(
    "--max-age-days",
    default=DEFAULT_MAX_AGE_DAYS,
    show_default=True,
    type=float,
    help="Drop entries older than this many days.",
)
def cache_prune_command(cache_dir, max_entries, max_age_days):
    """Drop expired and surplus entries."""
    report = prune_cache(
        cache_dir or default_cache_dir(),
        max_entries=max_entries,
        max_age_days=max_age_days,
    )

    if not report.removed:
        rprint(f"[green]Nothing to prune ({report.entries_after:,} entries).[/green]")
        return

    rprint(
        f"[green]Removed {report.removed:,} entries "
        f"({report.expired:,} expired, {report.evicted:,} least recently used). "
        f"{report.entries_after:,} remain.[/green]"
    )


@cache.command("clear")
@click.option(
    "--cache-dir",
    default=None,
    type=click.Path(),
    help="Cache directory to clear. Defaults to ./.repo2readme/cache.",
)
@click.option(
    "--remove-directory",
    is_flag=True,
    default=False,
    help="Delete the cache directory itself, not just its entries.",
)
@click.option("--force", "-f", is_flag=True, help="Do not ask for confirmation.")
def cache_clear_command(cache_dir, remove_directory, force):
    """Delete every cached summary."""
    resolved = cache_dir or default_cache_dir()
    summary = cache_info(resolved)

    if not summary.exists:
        rprint(f"[yellow]No cache at {resolved}[/yellow]")
        return

    if not force and not click.confirm(
        f"Delete {summary.entries:,} cached summaries in {resolved}?",
        default=False,
    ):
        rprint("[yellow]Cache was left alone.[/yellow]")
        return

    removed = clear_cache(resolved, remove_directory=remove_directory)
    rprint(f"[green]Removed {removed:,} cached summaries.[/green]")


@main.command()
def reset():
    """Reset stored API keys"""

    if reset_api_keys():
        rprint("[green]API keys reset successfully![/green]")
        rprint("Run repo2readme again to reconfigure keys.")
    else:
        rprint("[yellow]No API key file found to reset.[/yellow]")


if __name__ == "__main__":
    main()