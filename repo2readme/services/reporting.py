"""Turn summarization failures into something the user actually sees.

``summarize_file()`` never raises: it logs and returns ``{"file_path": ...,
"error": ...}``. Those placeholders used to be indistinguishable from real
summaries, so they flowed into the directory roll-up and the README prompt
while the run still reported success. This module separates them out and
renders a short report.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from typing import Any

from repo2readme.utils.console import safe

# Provider errors can be several hundred characters of JSON. Keep the console
# readable; the full text is still available in the returned failure objects.
MAX_REASON_LENGTH = 140

# Beyond this, individual paths are collapsed into a "... and N more" line.
MAX_PATHS_PER_GROUP = 5


@dataclass(frozen=True)
class SummaryFailure:
    """A single file that could not be summarized."""

    file_path: str
    reason: str

    def short_reason(self, max_length: int = MAX_REASON_LENGTH) -> str:
        return truncate_reason(self.reason, max_length)


@dataclass(frozen=True)
class FailureGroup:
    """Failures that share the same (truncated) reason."""

    reason: str
    file_paths: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.file_paths)


def truncate_reason(reason: str, max_length: int = MAX_REASON_LENGTH) -> str:
    """Collapse whitespace in a provider error and cap its length."""
    collapsed = " ".join(str(reason).split())
    if not collapsed:
        return "unknown error"
    if len(collapsed) <= max_length:
        return collapsed
    return collapsed[: max_length - 3].rstrip() + "..."


def is_failed_summary(entry: Any) -> bool:
    """True when an entry is a failure placeholder rather than a summary.

    Only a dict carrying a non-empty ``error`` value counts. A summary that
    happens to *describe* error handling is a normal dict without that key, and
    a legitimate summary containing ``"error": ""`` is not treated as a failure.
    """
    if not isinstance(entry, dict):
        return False
    if "error" not in entry:
        return False
    error = entry["error"]
    if error is None:
        return False
    if isinstance(error, str):
        return bool(error.strip())
    return True


def as_failure(entry: dict) -> SummaryFailure:
    """Convert a failure placeholder into a :class:`SummaryFailure`."""
    return SummaryFailure(
        file_path=str(entry.get("file_path") or "<unknown file>"),
        reason=str(entry.get("error")),
    )


def partition_summaries(
    summaries: Iterable[Any],
) -> tuple[list[Any], list[SummaryFailure]]:
    """Split summaries into usable ones and failures.

    Entries that are not dicts (some providers return plain strings) are kept
    as successes, matching how the downstream roll-up already treats them.
    """
    successful: list[Any] = []
    failures: list[SummaryFailure] = []

    for entry in summaries:
        if is_failed_summary(entry):
            failures.append(as_failure(entry))
        else:
            successful.append(entry)

    return successful, failures


def group_failures(failures: Sequence[SummaryFailure]) -> list[FailureGroup]:
    """Group failures by reason, most frequent first.

    Thirty rate-limited files share one reason and should render as one line,
    not thirty.
    """
    counts = Counter(failure.short_reason() for failure in failures)
    paths: dict[str, list[str]] = {}
    for failure in failures:
        paths.setdefault(failure.short_reason(), []).append(failure.file_path)

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [FailureGroup(reason=reason, file_paths=tuple(paths[reason])) for reason, _ in ordered]


def build_report_lines(
    total: int,
    succeeded: int,
    failures: Sequence[SummaryFailure],
    max_paths_per_group: int = MAX_PATHS_PER_GROUP,
) -> list[str]:
    """Build the console report as Rich-markup lines.

    Returns an empty list when nothing failed, so the happy path stays quiet.
    """
    if not failures:
        return []

    lines = [
        "",
        "[bold yellow]Summarization report[/bold yellow]",
        "",
        f"Succeeded          : {succeeded}/{total}",
        f"Failed             : {len(failures)}/{total}",
        "",
    ]

    # The reason is the provider's own error text and the paths are file names;
    # both routinely contain square brackets, which the console would otherwise
    # read as style tags and delete.
    for group in group_failures(failures):
        lines.append(
            f"[yellow]{group.count} file(s):[/yellow] {safe(group.reason)}"
        )
        for path in group.file_paths[:max_paths_per_group]:
            lines.append(f"    - {safe(path)}")
        remaining = group.count - max_paths_per_group
        if remaining > 0:
            lines.append(f"    ... and {remaining} more")
        lines.append("")

    lines.append(
        "[yellow]The README was generated from the files that succeeded.[/yellow]"
    )
    return lines


def render_report(
    total: int,
    succeeded: int,
    failures: Sequence[SummaryFailure],
    printer,
) -> None:
    """Print the report using ``printer`` (normally ``rich.print``)."""
    for line in build_report_lines(total, succeeded, failures):
        printer(line)
