"""Decide whether a run has anything to analyze, before it costs anything.

``run()`` guarded against every file *failing* to summarize but not against
there being no files at all::

    if total_documents and not successful_summaries:
        raise SystemExit(1)

``total_documents`` is ``0`` for an empty repository, so ``0 and ...`` is falsy
and the one case the guard exists for walked straight past it. Nothing
downstream stopped either: ``generate_hierarchical_summaries([])`` returns
``[]``, and the README prompt with an empty ``{summaries}`` slot is a set of
instructions with no subject, which the model answers by inventing a project.
The result was written over the user's own README, with exit status 0.

This module makes that state a first-class outcome. It also groups the skip
reasons the traversal collected, so "81 files found, all of them skipped" can
name the rule responsible instead of only stating the count.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# Reasons arrive from three layers - the filter, the discovery walk and the
# per-file loader - and some of them embed measurements, so they cannot be
# counted as they are:
#
#     "exceeds maximum file size (317440 B > 204800 B limit)"
#
# Every distinct size produced its own row, which made the breakdown longer the
# worse the problem was. Each raw reason is mapped to a stable category first;
# the measurements stay in ``ctx.skipped`` for anyone who wants them.
_REASON_PREFIXES: tuple[tuple[str, str], ...] = (
    ("excluded by pattern", "excluded by --exclude"),
    ("ignored by default rules", "ignored by default rules"),
    ("protected large file", "protected large file"),
    ("exceeds maximum file size", "over --max-file-size-kb"),
    ("cannot determine file size", "unreadable"),
    ("ignored by gitignore", "ignored by .gitignore"),
    ("binary_file", "binary content"),
    ("broken symbolic link", "broken symbolic link"),
    ("symbolic link outside repository", "symlink outside the repository"),
    ("circular or duplicate symbolic link", "circular symbolic link"),
    ("encoding_error", "could not be decoded"),
    ("unexpected_error", "unexpected error"),
    ("filtered", "ignored by default rules"),
)

# Order the categories are reported in: the ones a user can act on first.
CATEGORY_ORDER: tuple[str, ...] = tuple(label for _, label in _REASON_PREFIXES)

OTHER_CATEGORY = "other"

# The flags that most often explain an empty analysis, in the order they are
# worth checking.
_ACTIONABLE_HINTS: dict[str, str] = {
    "excluded by --exclude": "Loosen or drop the --exclude pattern.",
    "ignored by default rules": (
        "Use --include to override the built-in ignore rules for the paths "
        "you need."
    ),
    "over --max-file-size-kb": "Raise --max-file-size-kb.",
    "ignored by .gitignore": "Drop --respect-gitignore.",
    "protected large file": "Name the file exactly in --include to read it.",
}


def categorize_skip_reason(reason: str | None) -> str:
    """Map a raw skip reason onto one of :data:`CATEGORY_ORDER`.

    Unrecognised reasons keep their own text rather than being folded into
    ``other``, so a new reason added elsewhere in the pipeline still shows up
    with something readable instead of disappearing into a bucket.
    """
    text = (reason or "").strip()
    if not text:
        return OTHER_CATEGORY

    lowered = text.lower()
    for prefix, label in _REASON_PREFIXES:
        if lowered.startswith(prefix):
            return label

    # An unknown reason that carries a measurement or a nested exception is cut
    # at the first delimiter so it can still be counted.
    for delimiter in (":", "("):
        head, found, _ = text.partition(delimiter)
        if found:
            return head.strip() or OTHER_CATEGORY

    return text


def group_skip_reasons(
    skipped: Iterable[tuple[str, str]] | None,
) -> list[tuple[str, int]]:
    """Count skipped entries by category, most frequent first.

    Ties are broken by :data:`CATEGORY_ORDER` so the output is deterministic
    and the actionable categories lead.
    """
    counts: Counter[str] = Counter()
    for entry in skipped or ():
        # Entries are ``(path, reason)``; tolerate a bare path from a caller
        # that only recorded what it dropped.
        reason = entry[1] if isinstance(entry, (tuple, list)) and len(entry) > 1 else ""
        counts[categorize_skip_reason(reason)] += 1

    def rank(category: str) -> int:
        try:
            return CATEGORY_ORDER.index(category)
        except ValueError:
            return len(CATEGORY_ORDER)

    return sorted(counts.items(), key=lambda item: (-item[1], rank(item[0]), item[0]))


def build_skip_summary_lines(
    skipped: Iterable[tuple[str, str]] | None,
    heading: str = "Skipped Files Summary",
) -> list[str]:
    """The skip breakdown as Rich-markup lines, empty when nothing was skipped."""
    grouped = group_skip_reasons(skipped)
    if not grouped:
        return []

    width = max(len(category) for category, _ in grouped)
    lines = ["", f"[bold]{heading}[/bold]", ""]
    lines.extend(f"{category:<{width}} : {count}" for category, count in grouped)
    return lines


@dataclass(frozen=True)
class EmptyAnalysis:
    """A run that has nothing to summarize, and why."""

    root_path: str
    skipped_count: int
    reasons: tuple[tuple[str, int], ...] = ()

    @property
    def everything_filtered(self) -> bool:
        """Whether files were found and then all removed by the filters."""
        return self.skipped_count > 0

    @property
    def headline(self) -> str:
        if self.everything_filtered:
            return (
                f"All {self.skipped_count} file(s) found under {self.root_path} "
                "were skipped, so there is nothing to summarize."
            )
        return f"No files were found under {self.root_path}."

    def hints(self) -> list[str]:
        """Suggestions drawn from the categories that actually occurred."""
        return [
            _ACTIONABLE_HINTS[category]
            for category, _ in self.reasons
            if category in _ACTIONABLE_HINTS
        ]


def check_analysis_not_empty(
    root_path: str,
    document_count: int,
    skipped: Sequence[tuple[str, str]] | None = None,
) -> EmptyAnalysis | None:
    """Return an :class:`EmptyAnalysis` when the run has no work, else ``None``."""
    if document_count > 0:
        return None

    skipped = list(skipped or ())
    return EmptyAnalysis(
        root_path=str(root_path),
        skipped_count=len(skipped),
        reasons=tuple(group_skip_reasons(skipped)),
    )


def build_empty_analysis_lines(
    result: EmptyAnalysis,
    suggest_dry_run: bool = True,
) -> list[str]:
    """Render an :class:`EmptyAnalysis` as Rich-markup lines.

    ``suggest_dry_run`` is turned off by the dry run itself, which has already
    listed everything it skipped.
    """
    lines = ["", f"[red]{result.headline}[/red]"]

    if result.reasons:
        lines.append("")
        width = max(len(category) for category, _ in result.reasons)
        lines.extend(
            f"{category:<{width}} : {count}" for category, count in result.reasons
        )

    hints = result.hints()
    if hints:
        lines.append("")
        lines.extend(f"[yellow]{hint}[/yellow]" for hint in hints)

    lines.append("")
    tail = "[yellow]Nothing was written and no API requests were made."
    if suggest_dry_run:
        tail += " Re-run with --dry-run to see every skipped path."
    lines.append(tail + "[/yellow]")
    return lines


def render_empty_analysis(
    result: EmptyAnalysis,
    printer,
    suggest_dry_run: bool = True,
) -> None:
    """Print the explanation using ``printer`` (normally ``rich.print``)."""
    for line in build_empty_analysis_lines(result, suggest_dry_run=suggest_dry_run):
        printer(line)
