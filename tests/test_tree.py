"""Tests for repository tree rendering.

The tree ends up in the README as the "Folder Structure" section, so these
cover both the rendering itself (connectors, nesting, truncation) and the
requirement that it describe exactly the files that were analyzed.
"""

import os

import pytest

from repo2readme.utils.tree import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_ENTRIES_PER_DIR,
    extract_tree,
    generate_tree,
    generate_tree_from_paths,
)


@pytest.fixture
def repo(tmp_path):
    """A small repository with a nested package, a test dir and an ignored file."""
    project = tmp_path / "project"
    (project / "src" / "api").mkdir(parents=True)
    (project / "tests").mkdir()

    (project / "README.md").write_text("# hi", encoding="utf-8")
    (project / "src" / "main.py").write_text("x", encoding="utf-8")
    (project / "src" / "util.py").write_text("x", encoding="utf-8")
    (project / "src" / "logo.png").write_bytes(b"\x89PNG")
    (project / "src" / "api" / "routes.py").write_text("x", encoding="utf-8")
    (project / "tests" / "test_main.py").write_text("x", encoding="utf-8")
    return project


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_rendering_is_nested_and_sorted(repo):
    assert generate_tree(str(repo)) == "\n".join(
        [
            "project/",
            "├── src/",
            "│   ├── api/",
            "│   │   └── routes.py",
            "│   ├── main.py",
            "│   └── util.py",
            "├── tests/",
            "│   └── test_main.py",
            "└── README.md",
        ]
    )


def test_last_visible_entry_gets_the_closing_connector(repo):
    """The old renderer picked connectors from the unfiltered index, so a
    directory whose last file was filtered out never closed."""
    lines = generate_tree(str(repo)).splitlines()

    # logo.png is filtered out; util.py is the last visible child of src/.
    assert "│   └── util.py" in lines
    assert "│   ├── util.py" not in lines

    # README.md is the last entry of the root, after the directories.
    assert lines[-1] == "└── README.md"

    # Exactly one closing connector per directory level in this fixture.
    assert sum(1 for line in lines if line.startswith("└── ")) == 1


def test_directories_are_listed_before_files(repo):
    lines = generate_tree(str(repo)).splitlines()
    first_file = next(i for i, line in enumerate(lines) if "README.md" in line)
    last_dir = max(i for i, line in enumerate(lines) if line.rstrip().endswith("/"))
    assert last_dir < first_file


def test_trailing_separator_on_root_is_tolerated(repo):
    """`--local /path/to/repo/` used to lose the root name and all nesting."""
    assert generate_tree(str(repo)) == generate_tree(str(repo) + os.sep)
    assert generate_tree(str(repo) + os.sep).splitlines()[0] == "project/"


def test_root_name_is_not_stripped_from_nested_paths(tmp_path):
    """`str.replace` replaced every occurrence of the root, not just the prefix."""
    root = tmp_path / "app"
    nested = root / "app" / "app"
    nested.mkdir(parents=True)
    (nested / "main.py").write_text("x", encoding="utf-8")

    assert generate_tree(str(root)) == "\n".join(
        [
            "app/",
            "└── app/",
            "    └── app/",
            "        └── main.py",
        ]
    )


def test_empty_repository_renders_just_the_root(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert generate_tree(str(empty)) == "empty/"


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_exclude_patterns_are_honoured(repo):
    tree = generate_tree(str(repo), exclude_patterns=["tests/*"])
    assert "test_main.py" not in tree
    assert "main.py" in tree


def test_include_patterns_are_honoured(repo):
    # A plain data file, not a manifest, so it stays ignored by default.
    (repo / "data.json").write_text("{}", encoding="utf-8")

    assert "data.json" not in generate_tree(str(repo))
    assert "data.json" in generate_tree(str(repo), include_patterns=["data.json"])


def test_size_limit_is_honoured(repo):
    (repo / "src" / "huge.py").write_text("x" * 300 * 1024, encoding="utf-8")

    assert "huge.py" in generate_tree(str(repo), max_file_size_kb=None)
    assert "huge.py" not in generate_tree(str(repo), max_file_size_kb=200)


def test_gitignore_is_honoured(repo):
    (repo / ".gitignore").write_text("tests/\n", encoding="utf-8")

    assert "test_main.py" in generate_tree(str(repo))
    assert "test_main.py" not in generate_tree(str(repo), respect_gitignore=True)


def test_default_ignores_still_apply(repo):
    (repo / "node_modules" / "left-pad").mkdir(parents=True)
    (repo / "node_modules" / "left-pad" / "index.js").write_text(
        "x", encoding="utf-8"
    )

    tree = generate_tree(str(repo))
    assert "node_modules" not in tree
    assert "index.js" not in tree
    assert "logo.png" not in tree


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_entries_per_directory_are_capped_and_the_cut_is_shown(tmp_path):
    wide = tmp_path / "wide"
    wide.mkdir()
    for i in range(30):
        (wide / f"mod_{i:02d}.py").write_text("x", encoding="utf-8")

    tree = generate_tree(str(wide), max_entries_per_dir=10)
    lines = tree.splitlines()

    assert len(lines) == 1 + 10 + 1
    assert lines[-1] == "└── ... (20 more)"
    assert lines[-2].startswith("├── ")


def test_depth_is_capped_and_the_cut_is_shown(tmp_path):
    deep = tmp_path / "deep"
    path = deep
    for level in range(6):
        path = path / f"level{level}"
    path.mkdir(parents=True)
    (path / "leaf.py").write_text("x", encoding="utf-8")

    tree = generate_tree(str(deep), max_depth=3)
    assert "leaf.py" not in tree
    assert "depth limit reached" in tree
    assert tree.count("level") == 3


def test_truncation_is_off_when_limits_are_none(tmp_path):
    wide = tmp_path / "wide"
    wide.mkdir()
    for i in range(80):
        (wide / f"mod_{i:02d}.py").write_text("x", encoding="utf-8")

    tree = generate_tree(str(wide), max_depth=None, max_entries_per_dir=None)
    assert "more" not in tree
    assert len(tree.splitlines()) == 81


def test_defaults_do_not_truncate_an_ordinary_repository(repo):
    assert DEFAULT_MAX_DEPTH >= 8
    assert DEFAULT_MAX_ENTRIES_PER_DIR >= 50
    assert "..." not in generate_tree(str(repo))


# ---------------------------------------------------------------------------
# generate_tree_from_paths
# ---------------------------------------------------------------------------


def test_from_paths_renders_the_given_files_only():
    tree = generate_tree_from_paths(
        "/tmp/project",
        ["src/main.py", "src/api/routes.py", "README.md"],
    )
    assert tree == "\n".join(
        [
            "project/",
            "├── src/",
            "│   ├── api/",
            "│   │   └── routes.py",
            "│   └── main.py",
            "└── README.md",
        ]
    )


def test_from_paths_matches_the_walk_for_the_same_file_set(repo):
    _, absolute_paths = extract_tree(str(repo))
    relative_paths = [os.path.relpath(p, str(repo)) for p in absolute_paths]

    assert generate_tree_from_paths(str(repo), relative_paths) == generate_tree(
        str(repo)
    )


def test_from_paths_accepts_windows_separators():
    tree = generate_tree_from_paths("/tmp/project", ["src\\api\\routes.py"])
    assert tree == "project/\n└── src/\n    └── api/\n        └── routes.py"


def test_from_paths_ignores_empty_entries():
    assert generate_tree_from_paths("/tmp/project", ["", "", "."]) == "project/"


def test_from_paths_deduplicates_repeated_files():
    tree = generate_tree_from_paths("/tmp/p", ["a.py", "a.py", "b.py"])
    assert tree == "p/\n├── a.py\n└── b.py"


# ---------------------------------------------------------------------------
# extract_tree
# ---------------------------------------------------------------------------


def test_extract_tree_returns_paths_consistent_with_the_tree(repo):
    tree, files = extract_tree(str(repo))
    basenames = {os.path.basename(p) for p in files}

    assert basenames == {"README.md", "main.py", "util.py", "routes.py", "test_main.py"}
    for name in basenames:
        assert name in tree
    assert all(os.path.isabs(p) for p in files)


def test_extract_tree_applies_the_same_filters(repo):
    tree, files = extract_tree(str(repo), exclude_patterns=["tests/*"])
    assert "test_main.py" not in tree
    assert not any("test_main.py" in p for p in files)


# ---------------------------------------------------------------------------
# End to end: the tree the CLI prints describes the files it analyzed
# ---------------------------------------------------------------------------


def _dry_run(repo, *extra_args):
    import importlib

    from click.testing import CliRunner

    cli_main = importlib.import_module("repo2readme.cli.main")

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(repo), "--dry-run", *extra_args]
    )
    assert result.exit_code == 0, result.output
    return result.output


def _tree_and_processed(output):
    """Split a --dry-run transcript into the tree block and the file list."""
    tree_block, rest = output.split("Files to be processed", 1)
    processed = {
        line.strip().lstrip("✓ ").strip()
        for line in rest.splitlines()
        if line.strip().startswith("✓")
    }
    return tree_block, processed


def test_dry_run_tree_lists_exactly_the_files_that_will_be_processed(repo):
    tree, processed = _tree_and_processed(_dry_run(repo))

    assert processed
    for relative_path in processed:
        assert os.path.basename(relative_path) in tree

    tree_files = {
        line.rsplit(" ", 1)[-1]
        for line in tree.splitlines()
        if ("├── " in line or "└── " in line) and not line.rstrip().endswith("/")
    }
    assert tree_files == {os.path.basename(p) for p in processed}


def test_dry_run_tree_omits_excluded_files(repo):
    tree, processed = _tree_and_processed(_dry_run(repo, "--exclude", "*.md"))

    assert "README.md" not in tree
    assert not any(p.endswith(".md") for p in processed)
    assert "main.py" in tree


def test_dry_run_tree_omits_oversized_files(repo):
    (repo / "src" / "huge.py").write_text("x" * 300 * 1024, encoding="utf-8")

    tree, processed = _tree_and_processed(_dry_run(repo, "--max-file-size-kb", "1"))
    assert "huge.py" not in tree
    assert not any("huge.py" in p for p in processed)


def test_dry_run_tree_includes_files_pulled_in_by_include(repo):
    (repo / "data.json").write_text("{}", encoding="utf-8")

    plain, _ = _tree_and_processed(_dry_run(repo))
    forced, forced_processed = _tree_and_processed(
        _dry_run(repo, "--include", "data.json")
    )

    assert "data.json" not in plain
    assert "data.json" in forced
    assert any("data.json" in p for p in forced_processed)
