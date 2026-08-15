"""Rendering of the repository tree shown in ``--dry-run`` and embedded in the
generated README.

The tree is not decoration: the README prompt instructs the model to reproduce
it verbatim as the "Folder Structure" section, so it has to describe exactly the
set of files that were analyzed. The preferred entry point is therefore
:func:`generate_tree_from_paths`, which builds the tree from the documents the
loader actually produced. :func:`generate_tree` walks the filesystem instead and
takes the same filtering options as the loader, for callers that do not have a
document list to hand.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from repo2readme.utils.filter import github_file_filter
from repo2readme.utils.gitignore import is_gitignored

# A tree is pasted into the generator prompt on every iteration of the review
# loop, so an unbounded one from a large monorepo is an expensive way to spend
# the context window. These caps are generous enough that ordinary projects are
# never truncated, and truncation is always shown rather than silent.
DEFAULT_MAX_DEPTH = 8
DEFAULT_MAX_ENTRIES_PER_DIR = 50


def _new_node() -> dict:
    return {"dirs": {}, "files": []}


def _insert(root_node: dict, relative_path: str) -> None:
    """Insert a repository-relative file path into the tree."""
    parts = [
        part
        for part in relative_path.replace("\\", "/").split("/")
        if part and part != "."
    ]
    if not parts:
        return

    node = root_node
    for part in parts[:-1]:
        node = node["dirs"].setdefault(part, _new_node())
    node["files"].append(parts[-1])


def _count_entries(node: dict) -> int:
    """Total number of files and directories at or below ``node``."""
    total = len(node["files"]) + len(node["dirs"])
    for child in node["dirs"].values():
        total += _count_entries(child)
    return total


def _render(
    node: dict,
    prefix: str,
    depth: int,
    max_depth: int | None,
    max_entries_per_dir: int | None,
    lines: list[str],
) -> None:
    """Append the rendered children of ``node`` to ``lines``.

    Connectors are chosen from the entries that are actually rendered, so
    ``└──`` always marks the last visible child - including when the last file
    in a directory was filtered out, or when a subdirectory follows the files.
    """
    entries: list[tuple[str, bool]] = [(name, True) for name in sorted(node["dirs"])]
    entries += [(name, False) for name in sorted(set(node["files"]))]

    truncated = 0
    if max_entries_per_dir is not None and len(entries) > max_entries_per_dir:
        truncated = len(entries) - max_entries_per_dir
        entries = entries[:max_entries_per_dir]

    total_rendered = len(entries) + (1 if truncated else 0)

    for index, (name, is_dir) in enumerate(entries):
        is_last = index == total_rendered - 1
        connector = "└── " if is_last else "├── "

        if not is_dir:
            lines.append(f"{prefix}{connector}{name}")
            continue

        lines.append(f"{prefix}{connector}{name}/")
        child_prefix = prefix + ("    " if is_last else "│   ")
        child = node["dirs"][name]

        if max_depth is not None and depth + 1 >= max_depth:
            hidden = _count_entries(child)
            if hidden:
                lines.append(
                    f"{child_prefix}└── ... ({hidden} more, depth limit reached)"
                )
            continue

        _render(child, child_prefix, depth + 1, max_depth, max_entries_per_dir, lines)

    if truncated:
        lines.append(f"{prefix}└── ... ({truncated} more)")


def _root_name(root: str) -> str:
    """
    Directory name of ``root``, tolerating a trailing separator.

    ``.`` and ``..`` are resolved first, so ``--local .`` names the project
    rather than rendering the tree under a root called ``.``.
    """
    normalized = root.replace("\\", "/").rstrip("/")

    if not normalized or normalized in (".", "..") or normalized.endswith(("/.", "/..")):
        normalized = os.path.abspath(root).replace("\\", "/").rstrip("/")

    name = os.path.basename(normalized)
    # A root of "/" (or a bare drive) has no basename; fall back to the path.
    return name or normalized or root


def generate_tree_from_paths(
    root: str,
    relative_paths: Iterable[str],
    max_depth: int | None = DEFAULT_MAX_DEPTH,
    max_entries_per_dir: int | None = DEFAULT_MAX_ENTRIES_PER_DIR,
) -> str:
    """
    Render a tree from an explicit list of repository-relative file paths.

    This is the form the CLI uses: passing the paths of the documents that were
    loaded makes it impossible for the tree and the analyzed file set to
    disagree.
    """
    root_node = _new_node()
    for relative_path in relative_paths:
        if relative_path:
            _insert(root_node, relative_path)

    lines = [f"{_root_name(root)}/"]
    _render(root_node, "", 0, max_depth, max_entries_per_dir, lines)
    return "\n".join(lines)


def _collect_relative_paths(
    root: str,
    include_patterns: Iterable[str] | None,
    exclude_patterns: Iterable[str] | None,
    max_file_size_kb: int | None,
    respect_gitignore: bool,
) -> list[str]:
    """Walk ``root`` and return the relative paths that survive filtering."""
    relative_paths: list[str] = []

    for current_dir, dirs, files in os.walk(root):
        kept_dirs = []
        for directory in dirs:
            full_path = os.path.join(current_dir, directory)
            rel_path = os.path.relpath(full_path, root).replace("\\", "/")

            # github_file_filter relativizes against root_path itself, so it
            # must be handed the absolute path - giving it an already-relative
            # one makes os.path.relpath resolve against the CWD and the
            # resulting match path is meaningless.
            allowed, _ = github_file_filter(
                full_path,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                root_path=root,
                max_file_size_kb=None,
            )
            if not allowed:
                continue
            if respect_gitignore and is_gitignored(full_path, root):
                continue
            kept_dirs.append(directory)

        dirs[:] = kept_dirs

        for file_name in files:
            full_path = os.path.join(current_dir, file_name)
            rel_path = os.path.relpath(full_path, root).replace("\\", "/")

            allowed, _ = github_file_filter(
                full_path,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                root_path=root,
                max_file_size_kb=max_file_size_kb,
            )
            if not allowed:
                continue
            if respect_gitignore and is_gitignored(full_path, root):
                continue
            relative_paths.append(rel_path)

    return relative_paths


def generate_tree(
    root: str,
    include_patterns: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
    max_file_size_kb: int | None = 200,
    respect_gitignore: bool = False,
    max_depth: int | None = DEFAULT_MAX_DEPTH,
    max_entries_per_dir: int | None = DEFAULT_MAX_ENTRIES_PER_DIR,
) -> str:
    """
    Walk ``root`` and render its tree.

    The filtering options mirror the loader's, so a tree generated with the same
    arguments the loader was given describes the same set of files.
    """
    relative_paths = _collect_relative_paths(
        root,
        include_patterns,
        exclude_patterns,
        max_file_size_kb,
        respect_gitignore,
    )
    return generate_tree_from_paths(
        root,
        relative_paths,
        max_depth=max_depth,
        max_entries_per_dir=max_entries_per_dir,
    )


def extract_tree(
    root: str,
    include_patterns: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
    max_file_size_kb: int | None = 200,
    respect_gitignore: bool = False,
    max_depth: int | None = DEFAULT_MAX_DEPTH,
    max_entries_per_dir: int | None = DEFAULT_MAX_ENTRIES_PER_DIR,
) -> tuple[str, list[str]]:
    """
    Return the rendered tree together with the absolute paths it was built from.

    Both come from a single walk, so the listing can no longer drift from the
    rendering.
    """
    relative_paths = _collect_relative_paths(
        root,
        include_patterns,
        exclude_patterns,
        max_file_size_kb,
        respect_gitignore,
    )
    tree_structure = generate_tree_from_paths(
        root,
        relative_paths,
        max_depth=max_depth,
        max_entries_per_dir=max_entries_per_dir,
    )
    file_paths = [
        os.path.join(root, rel.replace("/", os.sep)) for rel in relative_paths
    ]
    return tree_structure, file_paths
