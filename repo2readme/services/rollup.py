"""Rolling file summaries up into directory summaries.

The roll-up used to be a plain recursive walk::

    def process_dir(node):
        for child_name, child_node in node["children"].items():
            child_summary = process_dir(child_node)   # blocking
            ...
        dir_summary = summarize_directory(...)        # blocking LLM call

Every ``summarize_directory()`` call is a network round trip and they ran
strictly one after another, even though sibling directories do not depend on
each other at all. The file step directly above it in
``services/summarization.py`` has used a thread pool since it was written; this
step never did, and ``--max-workers`` was not even passed to it.

Three other things came out of the same function:

* Nothing was cached. A run where every file summary was a cache hit still paid
  for the entire roll-up.
* ``summarize_directory()`` returns an ``{"error": ...}`` placeholder on
  failure, exactly like ``summarize_file()``. File-level placeholders are
  filtered out by ``partition_summaries()`` before they can reach the README
  prompt; directory-level ones were not filtered by anything and went straight
  into it.
* The 15-file threshold was a literal in the middle of the function.

This module owns all four concerns. The tree is processed one depth level at a
time, deepest first: a directory only depends on its own children, so
everything at the same depth can run concurrently and every child is finished
before its parent starts.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from repo2readme.services.reporting import SummaryFailure, is_failed_summary
from repo2readme.summarize.directory_summary import summarize_directory

logger = logging.getLogger(__name__)

# Repositories smaller than this are handed to the README prompt as they are:
# the roll-up exists to keep the prompt within a sensible size, and below this
# it only costs calls and loses detail.
DEFAULT_ROLLUP_THRESHOLD = 15

# Same default as the file summarization step.
DEFAULT_MAX_WORKERS = 4

# Directory entries share the summary cache with file entries. The prefix keeps
# the two apart: a directory is keyed on the summaries of its contents, not on
# any file's content, and it must not be mistaken for a path on disk.
DIRECTORY_KEY_PREFIX = "<dir>:"

# The "language" recorded for a directory entry, so a directory can never
# collide with a file summary that happens to share its path.
DIRECTORY_LANGUAGE = "directory"


def directory_cache_key(dir_path: str) -> str:
    """The summary cache key for a directory."""
    return f"{DIRECTORY_KEY_PREFIX}{dir_path}"


def is_directory_key(cache_key: str) -> bool:
    """Whether a cache key belongs to a directory rather than a file."""
    return str(cache_key).startswith(DIRECTORY_KEY_PREFIX)


def contents_fingerprint(contents: list) -> str:
    """A stable representation of what a directory summary was built from.

    Used as the cached "content" for a directory, so the entry is invalidated
    exactly when one of the summaries underneath it changes, and not otherwise.
    """
    return json.dumps(contents, sort_keys=True, default=str)


@dataclass
class RollupResult:
    """What the roll-up produced.

    Attributes
    ----------
    summaries:
        The top level summaries to hand to the README prompt.
    failures:
        Directories whose summary could not be generated. These are reported
        like file failures instead of being fed to the model.
    cache_keys:
        Every directory cache key this run is responsible for. The CLI adds
        them to the set of live keys so the stale-entry sweep does not delete
        the directory summaries it just wrote.
    """

    summaries: list = field(default_factory=list)
    failures: list[SummaryFailure] = field(default_factory=list)
    cache_keys: set[str] = field(default_factory=set)
    # Directories at the same depth are summarized on different threads, and
    # they all record into the same result.
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def record_key(self, cache_key: str) -> None:
        with self._lock:
            self.cache_keys.add(cache_key)

    def record_failure(self, failure: SummaryFailure) -> None:
        with self._lock:
            self.failures.append(failure)


def build_directory_tree(file_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Builds a tree structure of the repository based on file paths.
    """
    tree = {"type": "dir", "path": ".", "files": [], "children": {}}
    for summary in file_summaries:
        if isinstance(summary, str):
            continue
        path = summary.get("file_path", "")
        if not path:
            continue
        parts = path.split("/")
        current = tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                current["files"].append(summary)
            else:
                if part not in current["children"]:
                    current["children"][part] = {
                        "type": "dir",
                        "path": "/".join(parts[:i+1]),
                        "files": [],
                        "children": {}
                    }
                current = current["children"][part]
    return tree


def nodes_by_depth(tree: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Group the tree's directories by depth, root first.

    A directory depends only on its own children, so everything in one group
    can be summarized concurrently once the group below it is done.
    """
    levels: list[list[dict[str, Any]]] = []
    current = [tree]

    while current:
        levels.append(current)
        current = [child for node in current for child in node["children"].values()]

    return levels


def count_directories(tree: dict[str, Any]) -> int:
    """How many directories will be summarized, excluding the root."""
    return sum(len(level) for level in nodes_by_depth(tree)) - 1


def generate_hierarchical_summaries(
    file_summaries: list[dict[str, Any]],
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_workers: int | None = None,
    summary_cache=None,
    threshold: int = DEFAULT_ROLLUP_THRESHOLD,
    progress=None,
    task_id=None,
) -> RollupResult:
    """
    Roll file summaries up into directory summaries.

    Sibling directories are summarized concurrently, bounded by ``max_workers``
    (the same option the file step uses). When ``summary_cache`` is given, a
    directory summary is reused as long as none of the summaries underneath it
    changed, so an incremental run stops paying for the whole roll-up.

    Repositories with ``threshold`` files or fewer skip the roll-up entirely
    and are returned unchanged.
    """
    result = RollupResult()

    if len(file_summaries) <= threshold:
        result.summaries = list(file_summaries)
        if progress and task_id is not None:
            progress.update(task_id, advance=1)
        return result

    tree = build_directory_tree(file_summaries)
    levels = nodes_by_depth(tree)

    if progress and task_id is not None:
        progress.update(task_id, total=max(count_directories(tree), 1), completed=0)

    workers = max(1, min(max_workers or DEFAULT_MAX_WORKERS, _widest_level(levels)))

    # results are keyed by the identity of the node, since two directories can
    # share a name at different points in the tree.
    rolled: dict[int, Any] = {}

    def summarize(node: dict[str, Any]) -> Any:
        contents = _contents_of(node, rolled)
        is_root = node["path"] == "."

        try:
            if is_root or not contents:
                return contents if is_root else None

            if len(contents) == 1:
                # Nothing to synthesize: the single child already is this
                # directory's summary, and asking the model to restate it
                # costs a call and loses detail.
                return contents[0]

            return _summarize_directory_cached(
                node["path"],
                contents,
                provider=provider,
                model=model,
                base_url=base_url,
                summary_cache=summary_cache,
                result=result,
            )
        finally:
            if not is_root:
                _advance(progress, task_id)

    # Deepest level first; the root (level 0) is handled after the loop.
    for level in reversed(levels[1:]):
        if len(level) == 1 or workers == 1:
            for node in level:
                rolled[id(node)] = summarize(node)
            continue

        with ThreadPoolExecutor(max_workers=min(workers, len(level))) as executor:
            for node, summary in zip(level, executor.map(summarize, level)):
                rolled[id(node)] = summary

    top_level = summarize(tree)

    if top_level is None:
        result.summaries = []
    elif isinstance(top_level, list):
        result.summaries = top_level
    else:
        result.summaries = [top_level]

    return result


def _widest_level(levels: list[list[dict[str, Any]]]) -> int:
    """The most directories that can ever run at once."""
    return max((len(level) for level in levels), default=1)


def _advance(progress, task_id) -> None:
    if progress and task_id is not None:
        progress.update(task_id, advance=1)


def _contents_of(node: dict[str, Any], rolled: dict[int, Any]) -> list:
    """This directory's own files plus whatever its children rolled up to."""
    contents: list = []

    for child in node["children"].values():
        child_summary = rolled.get(id(child))
        if not child_summary:
            continue
        if isinstance(child_summary, list):
            contents.extend(child_summary)
        else:
            contents.append(child_summary)

    contents.extend(node["files"])
    return contents


def _summarize_directory_cached(
    dir_path: str,
    contents: list,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    summary_cache,
    result: RollupResult,
) -> Any:
    """One directory summary, cached, with failures kept out of the prompt."""
    cache_key = directory_cache_key(dir_path)
    fingerprint = contents_fingerprint(contents)
    result.record_key(cache_key)

    if summary_cache is not None:
        cached = summary_cache.get(cache_key, fingerprint, DIRECTORY_LANGUAGE)
        if cached is not None:
            return cached

    try:
        summary = summarize_directory(
            dir_path=dir_path,
            contents_summaries=contents,
            provider=provider,
            model_name=model,
            base_url=base_url,
        )
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return _failed(dir_path, str(exc), contents, result)

    if is_failed_summary(summary):
        return _failed(dir_path, str(summary.get("error")), contents, result)

    if summary_cache is not None:
        summary_cache.put(cache_key, fingerprint, DIRECTORY_LANGUAGE, summary, 0)

    return summary


def _failed(
    dir_path: str, reason: str, contents: list, result: RollupResult
) -> list:
    """Record a failed roll-up and fall back to the contents it was built from.

    Passing the children up unchanged keeps their detail in the README prompt.
    The alternative - handing the model the ``{"error": ...}`` placeholder,
    which is what used to happen - tells it nothing and invites it to invent
    something.
    """
    logger.warning("Directory summary failed for %s: %s", dir_path, reason)
    result.record_failure(SummaryFailure(file_path=dir_path, reason=reason))
    return contents
