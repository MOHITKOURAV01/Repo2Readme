"""What a run is about to cost, stage by stage.

The confirmation prompt exists so the user can decide whether a run is worth
paying for. The number behind it used to be this::

    estimated_tokens = sum(max(1, len(doc["content"]) // 3) for doc in documents)

which is the size of the source code on disk, not the cost of the run. It was
wrong in both directions and silent about why:

* **Cache hits were charged at full price.** The cache was built twenty lines
  earlier in ``run()`` and never consulted, so a second run over an unchanged
  repository - which makes no summarization requests at all - printed the same
  figure as the first.
* **The directory roll-ups were not counted.** Above the roll-up threshold,
  one ``summarize_directory`` call happens per directory that has more than one
  entry, each carrying the JSON of everything below it.
* **The README loop was not counted.** ``run_pipeline`` runs up to three
  generate calls, each interleaved with a review call, and every generate
  prompt carries *all* the summaries, the tree and the dependency overview.
* **Prompt overhead and output tokens were not counted** at all.

Everything here is an estimate and is presented as one: the totals are rounded,
and the README loop is counted at its maximum, because it exits early on a good
score. Where a number can be derived from the code that will actually run - the
prompt templates, the roll-up tree walk, the iteration cap - it is derived
rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from repo2readme.services.summarization import (
    ROLLUP_THRESHOLD,
    count_directory_rollups,
)

# Bytes per token. Deliberately low: tokenizers differ by provider, and an
# estimate that is a little pessimistic costs a user nothing, while one that is
# optimistic costs them a surprise.
CHARS_PER_TOKEN = 3

# Room to leave for what comes back. A file summary is a small JSON object, a
# review is a score and a paragraph, a README is a document.
SUMMARY_OUTPUT_TOKENS = 400
ROLLUP_OUTPUT_TOKENS = 300
README_OUTPUT_TOKENS = 1_500
REVIEW_OUTPUT_TOKENS = 200

# Roll-up prompts and the README prompt embed summaries rather than source, so
# their input is proportional to how many summaries they carry.
TOKENS_PER_SUMMARY = SUMMARY_OUTPUT_TOKENS

STAGE_FILE_SUMMARIES = "File summaries"
STAGE_DIRECTORY_ROLLUPS = "Directory roll-ups"
STAGE_README = "README generation"
STAGE_REVIEW = "README review"


def format_size(size_bytes: int) -> str:
    """Formats a byte size into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def estimate_tokens(text: str | None) -> int:
    """Tokens in a piece of text, at :data:`CHARS_PER_TOKEN` bytes each."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def round_tokens(tokens: int) -> int:
    """Round to a precision the estimate can actually justify.

    ``~219,448`` has six significant figures and came from dividing a byte
    count by three. Two is as much as any of this supports.
    """
    if tokens < 1_000:
        return tokens
    step = 1_000 if tokens < 100_000 else 10_000
    return int(round(tokens / step) * step)


def _prompt_overhead(template: str, format_instructions: str = "") -> int:
    """Tokens a template contributes before any content is substituted in."""
    return estimate_tokens(template) + estimate_tokens(format_instructions)


def summary_prompt_overhead() -> int:
    """Measured from the summarizer's own template and parser."""
    from langchain_core.output_parsers import JsonOutputParser

    from repo2readme.summarize.summary import PROMPT_TEMPLATE

    return _prompt_overhead(
        PROMPT_TEMPLATE, JsonOutputParser().get_format_instructions()
    )


def rollup_prompt_overhead() -> int:
    """Measured from the directory roll-up's own template and parser."""
    from langchain_core.output_parsers import JsonOutputParser

    from repo2readme.summarize.directory_summary import DIR_PROMPT_TEMPLATE

    return _prompt_overhead(
        DIR_PROMPT_TEMPLATE, JsonOutputParser().get_format_instructions()
    )


def readme_prompt_overhead() -> int:
    """Measured from the README generator's own template."""
    from repo2readme.readme.readme_generator import README_PROMPT_TEMPLATE

    return _prompt_overhead(README_PROMPT_TEMPLATE)


def review_prompt_overhead() -> int:
    """Measured from the reviewer's own template and parser."""
    from repo2readme.readme.reviewer_agent import ReviewSchema

    try:
        from langchain_core.output_parsers import PydanticOutputParser

        instructions = PydanticOutputParser(
            pydantic_object=ReviewSchema
        ).get_format_instructions()
    except Exception:  # pragma: no cover - depends on the installed parser
        instructions = ""

    from repo2readme.readme.reviewer_agent import REVIEW_PROMPT_TEMPLATE

    return _prompt_overhead(REVIEW_PROMPT_TEMPLATE, instructions)


@dataclass(frozen=True)
class StageEstimate:
    """One stage of the run: how many requests, carrying roughly how much."""

    name: str
    requests: int
    tokens: int

    @property
    def is_bounded(self) -> bool:
        """Whether the request count is an upper bound rather than exact."""
        return self.name in (STAGE_README, STAGE_REVIEW)


@dataclass(frozen=True)
class RunEstimate:
    """What the run about to be confirmed will cost."""

    files_selected: int
    files_to_summarize: int
    files_cached: int
    total_bytes: int
    stages: tuple[StageEstimate, ...] = ()

    @property
    def total_requests(self) -> int:
        return sum(stage.requests for stage in self.stages)

    @property
    def total_tokens(self) -> int:
        return sum(stage.tokens for stage in self.stages)

    @property
    def billable_stages(self) -> tuple[StageEstimate, ...]:
        """Stages that will actually make a request."""
        return tuple(stage for stage in self.stages if stage.requests)


def estimate_analysis_cost(documents: list) -> tuple[int, int, int]:
    """
    Estimates the tokens, size in bytes, and number of documents.
    Returns: (estimated_tokens, total_size_bytes, total_documents)

    Kept for callers that only want the size of the input. It describes the
    source, not the run; :func:`estimate_run` describes the run.
    """
    estimated_tokens = sum(estimate_tokens(doc["content"]) for doc in documents)
    total_size_bytes = sum(len(doc["content"].encode("utf-8")) for doc in documents)
    return estimated_tokens, total_size_bytes, len(documents)


def _cached_paths(
    documents: Sequence[dict],
    summary_cache: Any,
    resolve_language,
) -> set[int]:
    """Indices of the documents whose summary is already cached.

    A cache that cannot answer - not supplied, or an older one without
    ``contains`` - simply reports nothing as cached, which makes the estimate
    an upper bound rather than an error.
    """
    if summary_cache is None or not hasattr(summary_cache, "contains"):
        return set()

    cached: set[int] = set()
    for index, doc in enumerate(documents):
        metadata = doc.get("metadata") or {}
        file_path = metadata.get("file_path")
        if not file_path:
            continue
        try:
            language = resolve_language(metadata, doc.get("content", ""))
            if summary_cache.contains(file_path, doc.get("content", ""), language):
                cached.add(index)
        except Exception:
            # An estimate must never be the thing that ends a run.
            continue
    return cached


def estimate_run(
    documents: Sequence[dict],
    summary_cache: Any = None,
    tree: str = "",
    dependency_overview: str = "",
    max_readme_iterations: int | None = None,
) -> RunEstimate:
    """Model the run: which stages will fire, how often, carrying how much.

    ``summary_cache`` is consulted read-only through
    :meth:`SummaryCache.contains`, so asking about a run does not disturb the
    counters or invalidate anything.
    """
    from repo2readme.services.summarization import resolve_language

    if max_readme_iterations is None:
        from repo2readme.services.orchestrator import MAX_README_ITERATIONS

        max_readme_iterations = MAX_README_ITERATIONS

    documents = list(documents)
    total_bytes = sum(
        len((doc.get("content") or "").encode("utf-8")) for doc in documents
    )

    cached = _cached_paths(documents, summary_cache, resolve_language)
    to_summarize = [
        doc for index, doc in enumerate(documents) if index not in cached
    ]

    stages: list[StageEstimate] = []

    # --- File summaries -------------------------------------------------
    overhead = summary_prompt_overhead()
    summary_tokens = sum(
        estimate_tokens(doc.get("content")) + overhead + SUMMARY_OUTPUT_TOKENS
        for doc in to_summarize
    )
    stages.append(
        StageEstimate(STAGE_FILE_SUMMARIES, len(to_summarize), summary_tokens)
    )

    # --- Directory roll-ups ---------------------------------------------
    # Every selected file has a summary by this point, cached or not, so the
    # roll-up is counted over all of them rather than over the ones being
    # summarized now.
    file_paths = [
        (doc.get("metadata") or {}).get("file_path", "") for doc in documents
    ]
    file_paths = [path for path in file_paths if path]
    rollup_calls, rollup_items = count_directory_rollups(file_paths)
    rollup_overhead = rollup_prompt_overhead()
    rollup_tokens = (
        rollup_items * TOKENS_PER_SUMMARY
        + rollup_calls * (rollup_overhead + ROLLUP_OUTPUT_TOKENS)
    )
    stages.append(
        StageEstimate(STAGE_DIRECTORY_ROLLUPS, rollup_calls, rollup_tokens)
    )

    # --- README generation ----------------------------------------------
    # The prompt carries whatever the roll-up handed on: the file summaries
    # themselves below the threshold, the top-level roll-ups above it.
    summaries_reaching_readme = (
        len(documents) if len(documents) <= ROLLUP_THRESHOLD else max(1, rollup_calls)
    )
    readme_input = (
        summaries_reaching_readme * TOKENS_PER_SUMMARY
        + estimate_tokens(tree)
        + estimate_tokens(dependency_overview)
        + readme_prompt_overhead()
    )
    readme_rounds = max(0, max_readme_iterations) if documents else 0
    # From the second round on, the previous draft and the reviewer's feedback
    # are fed back in.
    readme_tokens = readme_rounds * (readme_input + README_OUTPUT_TOKENS)
    if readme_rounds > 1:
        readme_tokens += (readme_rounds - 1) * (
            README_OUTPUT_TOKENS + REVIEW_OUTPUT_TOKENS
        )
    stages.append(StageEstimate(STAGE_README, readme_rounds, readme_tokens))

    # --- README review ---------------------------------------------------
    review_tokens = readme_rounds * (
        README_OUTPUT_TOKENS + review_prompt_overhead() + REVIEW_OUTPUT_TOKENS
    )
    stages.append(StageEstimate(STAGE_REVIEW, readme_rounds, review_tokens))

    return RunEstimate(
        files_selected=len(documents),
        files_to_summarize=len(to_summarize),
        files_cached=len(cached),
        total_bytes=total_bytes,
        stages=tuple(stages),
    )


# Wide enough that a monorepo's totals do not push the columns out of line.
_REQUESTS_WIDTH = 9
_TOKENS_WIDTH = 11


def _row(label: str, name_width: int, requests: int, tokens: int) -> str:
    return (
        f"{label:<{name_width}}   {requests:>{_REQUESTS_WIDTH}}   "
        f"{round_tokens(tokens):>{_TOKENS_WIDTH},}"
    )


def build_estimate_lines(estimate: RunEstimate) -> list[str]:
    """Render a :class:`RunEstimate` as Rich-markup lines."""
    selected = f"Files selected     : {estimate.files_selected}"
    if estimate.files_cached:
        selected += (
            f"  ({estimate.files_to_summarize} to summarize, "
            f"{estimate.files_cached} cached)"
        )
    lines = [
        "",
        "[bold]Repository Analysis[/bold]",
        "",
        selected,
        f"Source size        : ~{format_size(estimate.total_bytes)}",
        "",
    ]

    stages = estimate.billable_stages
    if not stages:
        lines.append("Nothing to send: every selected file is already cached.")
        return lines

    rows = [
        (f"{stage.name} (max)" if stage.is_bounded else stage.name, stage)
        for stage in stages
    ]
    total_label = "Total (upper bound)"
    name_width = max(len(label) for label, _ in rows + [(total_label, None)])
    lines.append(
        f"{'Stage':<{name_width}}   {'Requests':>{_REQUESTS_WIDTH}}   "
        f"{'~Tokens':>{_TOKENS_WIDTH}}"
    )
    for label, stage in rows:
        lines.append(_row(label, name_width, stage.requests, stage.tokens))
    lines.append(
        f"{'-' * name_width}   {'-' * _REQUESTS_WIDTH}   {'-' * _TOKENS_WIDTH}"
    )
    lines.append(
        _row(
            total_label, name_width, estimate.total_requests, estimate.total_tokens
        )
    )
    return lines


def render_estimate(estimate: RunEstimate, printer) -> None:
    """Print the breakdown using ``printer`` (normally ``rich.print``)."""
    for line in build_estimate_lines(estimate):
        printer(line)
