"""Inspecting and managing the summary cache from the command line.

There was no way to look at the cache or clear it. ``repo2readme reset`` clears
API keys and nothing else, so the only way to drop the summary cache was to
know that it lives in ``.repo2readme/cache/summaries.json`` under the current
working directory and remove it by hand.

This module backs ``repo2readme cache info``, ``cache prune`` and
``cache clear``. It deliberately keeps no knowledge of the cache's internals
beyond what :class:`~repo2readme.cache.SummaryCache` exposes.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone

from repo2readme.cache import (
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_MAX_ENTRIES,
    CacheSummary,
    PruneReport,
    SummaryCache,
)

CACHE_DIR_NAME = os.path.join(".repo2readme", "cache")


def default_cache_dir(base: str | None = None) -> str:
    """Where the CLI keeps the summary cache."""
    return os.path.join(base or os.getcwd(), CACHE_DIR_NAME)


def _open_cache(cache_dir: str, **kwargs) -> SummaryCache:
    """A cache instance for administration.

    The config hash is irrelevant here: ``describe``, ``prune`` and ``clear``
    do not go through the lookup path that compares it, so nothing is
    invalidated just by looking.
    """
    return SummaryCache(
        cache_dir=cache_dir,
        config={},
        prompt_template_hash="",
        autosave=False,
        **kwargs,
    )


def cache_info(cache_dir: str) -> CacheSummary:
    """Describe the cache without modifying it."""
    return _open_cache(cache_dir).describe()


def prune_cache(
    cache_dir: str,
    max_entries: int | None = DEFAULT_MAX_ENTRIES,
    max_age_days: float | None = DEFAULT_MAX_AGE_DAYS,
) -> PruneReport:
    """Drop expired and surplus entries, and write the result."""
    cache = _open_cache(cache_dir)
    report = cache.prune(max_entries=max_entries, max_age_days=max_age_days)
    cache.flush()
    return report


def clear_cache(cache_dir: str, remove_directory: bool = False) -> int:
    """
    Drop every entry. Returns how many were removed.

    With ``remove_directory`` the cache directory itself is deleted, which is
    what someone reaching for "clear" after the tool has been used in many
    directories usually wants.
    """
    if not os.path.exists(cache_dir):
        return 0

    if remove_directory:
        removed = cache_info(cache_dir).entries
        shutil.rmtree(cache_dir, ignore_errors=True)
        return removed

    cache = _open_cache(cache_dir)
    removed = cache.clear()
    cache.flush()
    return removed


def format_size(size_bytes: int) -> str:
    """Human readable byte count."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_timestamp(value: float | None) -> str:
    """A readable date for a cache timestamp, or a dash when there is none."""
    if not value:
        return "-"
    return (
        datetime.fromtimestamp(value, tz=timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%d %H:%M")
    )


def build_info_lines(summary: CacheSummary) -> list[str]:
    """The body of ``repo2readme cache info``, as Rich markup lines."""
    if not summary.exists:
        return [
            f"[yellow]No cache at {summary.cache_file}[/yellow]",
            "It is created on the first run that summarizes a file.",
        ]

    lines = [
        "",
        "[bold]Summary cache[/bold]",
        "",
        f"Location           : {summary.cache_file}",
        f"Schema version     : {summary.schema_version or 'unknown'}",
        f"Entries            : {summary.entries:,}",
        f"Size on disk       : {format_size(summary.size_bytes)}",
        f"Repositories       : {summary.repositories}",
        f"Oldest entry       : {format_timestamp(summary.oldest_created_at)}",
        f"Newest entry       : {format_timestamp(summary.newest_created_at)}",
    ]

    if summary.namespaces:
        lines.append("")
        lines.append("[bold]Entries per repository[/bold]")
        lines.append("")
        for namespace, count in sorted(
            summary.namespaces.items(), key=lambda item: (-item[1], item[0])
        ):
            lines.append(f"{count:>8,}  {namespace}")

    return lines
