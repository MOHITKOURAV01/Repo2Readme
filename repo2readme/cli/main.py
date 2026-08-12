from dotenv import load_dotenv
load_dotenv()

import click
from rich import print as rprint
from rich.progress import Progress
from repo2readme.config import reset_api_keys
import os
from collections import Counter

from repo2readme.utils.tree import generate_tree
from repo2readme.cache import SummaryCache
from repo2readme.loaders.repo_loader import RepoLoader
from repo2readme.summarize.summary import get_prompt_template_hash
from repo2readme.dependency_graph import build_dependency_graph

# Import new services
from repo2readme.services.environment import setup_api_keys
from repo2readme.services.estimation import format_size, estimate_analysis_cost
from repo2readme.services.summarization import generate_all_summaries, generate_hierarchical_summaries
from repo2readme.services.orchestrator import run_pipeline


@click.group()
def main():
    """
    Use the `run` command to generate a README.
    Use the `reset` command to clear saved API keys.

    Note: First run will ask for your API keys.
    """


@main.command()
@click.option("--url", "-u", help="GitHub repo URL")
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
    "--provider",
    default=None,
    help="LLM provider (groq, google, openai, anthropic, openrouter, ollama, etc.)"
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
def run(url, local, output, force, include_patterns, exclude_patterns, max_file_size_kb, dry_run, respect_gitignore, max_workers, provider, model, base_url, branch):
    """ Use --url for GitHub repo url and --local for local repo
    """
    if not url and not local:
        rprint("[red]Provide either --url or --local[/red]")
        return

    source = url if url else local

    # Initialize file summary cache
    cache_dir = os.path.join(os.getcwd(), ".repo2readme", "cache")
    summarization_config = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
    }
    summary_cache = SummaryCache(
        cache_dir=cache_dir,
        config=summarization_config,
        prompt_template_hash=get_prompt_template_hash(),
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

    tree = generate_tree(root_path)

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
            
            rollup_task = progress.add_task("[cyan]Generating directory summaries...[/cyan]", total=1)
            hierarchical_summaries = generate_hierarchical_summaries(
                file_summaries=summaries,
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

    finally:
        if hasattr(loader_obj, "cleanup"):
            loader_obj.cleanup()


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