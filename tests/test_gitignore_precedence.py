"""
Tests for which ``.gitignore`` wins when several of them apply (issue #173).

The matcher already consulted every level between the root and the path. It
consulted them root-first and stopped at the first match, so the broadest rule
decided and a nested ``!pattern`` never got to speak. Git's rule is that the
nearest file wins.

Covers:
- a nested negation re-includes a file the root ignored
- a nested rule ignores a file the root said nothing about
- a nested negation does not leak sideways to a sibling directory
- depth, not order in the file, is what decides between two opinions
- a directory excluded higher up cannot be re-opened from inside
- directory-only (``build/``) and root-anchored (``/dist``) rules still work
- the answers agree with the git binary, where it is installed
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from repo2readme.utils import gitignore as gitignore_module
from repo2readme.utils.gitignore import clear_matcher_cache, is_gitignored

pytestmark = pytest.mark.skipif(
    gitignore_module.pathspec is None, reason="pathspec is not installed"
)


@pytest.fixture(autouse=True)
def _isolated_matcher_cache():
    clear_matcher_cache()
    yield
    clear_matcher_cache()


def _repo(tmp_path, files: dict[str, str]):
    """Materialise ``{relative path: contents}`` under ``tmp_path``."""
    for relative, body in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return tmp_path


def _ignored(root, relative: str) -> bool:
    return is_gitignored(str(root / relative), str(root))


# ---------------------------------------------------------------------------
# The nearest .gitignore decides
# ---------------------------------------------------------------------------


def test_nested_negation_re_includes_a_file_the_root_ignored(tmp_path):
    root = _repo(
        tmp_path,
        {
            ".gitignore": "*.log\n",
            "frontend/.gitignore": "!important.log\n",
            "frontend/important.log": "keep me\n",
            "frontend/other.log": "noise\n",
            "backend/other.log": "noise\n",
        },
    )

    assert _ignored(root, "frontend/important.log") is False
    # Everything the negation does not name is still ignored by the root rule.
    assert _ignored(root, "frontend/other.log") is True
    assert _ignored(root, "backend/other.log") is True


def test_a_negation_does_not_leak_into_a_sibling_directory(tmp_path):
    root = _repo(
        tmp_path,
        {
            ".gitignore": "*.log\n",
            "frontend/.gitignore": "!debug.log\n",
            "frontend/debug.log": "keep\n",
            "backend/debug.log": "drop\n",
        },
    )

    assert _ignored(root, "frontend/debug.log") is False
    assert _ignored(root, "backend/debug.log") is True


def test_a_nested_rule_ignores_what_the_root_never_mentioned(tmp_path):
    root = _repo(
        tmp_path,
        {
            ".gitignore": "*.log\n",
            "frontend/.gitignore": "bundle.js\n",
            "frontend/bundle.js": "generated\n",
            "backend/bundle.js": "hand written\n",
        },
    )

    assert _ignored(root, "frontend/bundle.js") is True
    assert _ignored(root, "backend/bundle.js") is False


def test_the_deepest_opinion_wins_over_an_intermediate_one(tmp_path):
    root = _repo(
        tmp_path,
        {
            ".gitignore": "*.txt\n",
            "a/.gitignore": "!notes.txt\n",
            "a/b/.gitignore": "notes.txt\n",
            "a/notes.txt": "kept by a/.gitignore\n",
            "a/b/notes.txt": "dropped again by a/b/.gitignore\n",
        },
    )

    assert _ignored(root, "a/notes.txt") is False
    assert _ignored(root, "a/b/notes.txt") is True


def test_a_directory_with_no_rules_defers_to_the_level_above(tmp_path):
    root = _repo(
        tmp_path,
        {
            ".gitignore": "*.log\n",
            "a/b/c/app.log": "noise\n",
        },
    )

    assert _ignored(root, "a/b/c/app.log") is True


def test_an_unmatched_path_is_not_ignored(tmp_path):
    root = _repo(
        tmp_path,
        {
            ".gitignore": "*.log\n",
            "src/main.py": "print(1)\n",
        },
    )

    assert _ignored(root, "src/main.py") is False


# ---------------------------------------------------------------------------
# An excluded directory cannot be re-opened from inside
# ---------------------------------------------------------------------------


def test_a_file_under_an_ignored_directory_stays_ignored(tmp_path):
    root = _repo(
        tmp_path,
        {
            ".gitignore": "build/\n",
            "build/.gitignore": "!keep.txt\n",
            "build/keep.txt": "git will not re-include this\n",
        },
    )

    assert _ignored(root, "build") is True
    assert _ignored(root, "build/keep.txt") is True


def test_a_re_included_directory_lets_its_contents_through(tmp_path):
    root = _repo(
        tmp_path,
        {
            ".gitignore": "vendor/\n",
            "packages/.gitignore": "!vendor/\n",
            "packages/vendor/lib.js": "checked in on purpose\n",
        },
    )

    assert _ignored(root, "packages/vendor") is False
    assert _ignored(root, "packages/vendor/lib.js") is False


# ---------------------------------------------------------------------------
# Rule forms that have to keep working
# ---------------------------------------------------------------------------


def test_directory_only_rules_do_not_match_a_file_of_the_same_name(tmp_path):
    root = _repo(
        tmp_path,
        {
            ".gitignore": "build/\n",
            "build/out.js": "generated\n",
            "build.py": "a file, not the directory\n",
        },
    )

    assert _ignored(root, "build") is True
    assert _ignored(root, "build.py") is False


def test_root_anchored_rules_stay_anchored_to_their_own_directory(tmp_path):
    root = _repo(
        tmp_path,
        {
            ".gitignore": "/dist\n",
            "dist/app.js": "generated\n",
            "packages/dist/app.js": "not the root dist\n",
        },
    )

    assert _ignored(root, "dist") is True
    assert _ignored(root, "packages/dist") is False


def test_git_info_exclude_is_still_read_at_the_root(tmp_path):
    root = _repo(
        tmp_path,
        {
            ".git/info/exclude": "secret.txt\n",
            "secret.txt": "local only\n",
            "src/secret.txt": "also matched, the rule is not anchored\n",
        },
    )

    assert _ignored(root, "secret.txt") is True
    assert _ignored(root, "src/secret.txt") is True


def test_a_nested_negation_can_override_git_info_exclude(tmp_path):
    root = _repo(
        tmp_path,
        {
            ".git/info/exclude": "*.tmp\n",
            "scratch/.gitignore": "!keep.tmp\n",
            "scratch/keep.tmp": "kept\n",
            "scratch/drop.tmp": "dropped\n",
        },
    )

    assert _ignored(root, "scratch/keep.tmp") is False
    assert _ignored(root, "scratch/drop.tmp") is True


def test_a_repository_with_no_rules_ignores_nothing(tmp_path):
    root = _repo(tmp_path, {"src/main.py": "print(1)\n"})

    assert _ignored(root, "src/main.py") is False


def test_a_path_outside_the_root_is_not_ignored(tmp_path):
    root = _repo(tmp_path, {".gitignore": "*.log\n"})
    outside = tmp_path.parent / "elsewhere.log"
    outside.write_text("noise\n", encoding="utf-8")

    assert is_gitignored(str(outside), str(root)) is False


# ---------------------------------------------------------------------------
# Cross-check against git itself
# ---------------------------------------------------------------------------


def _git_ignores(root, relative: str) -> bool:
    """What ``git`` says, read from the untracked listing.

    ``git check-ignore`` exits zero whenever *some* pattern matched, including a
    negation, so it cannot answer this on its own. ``git status`` lists exactly
    the paths git would offer to add, which is the question being asked.
    """
    subprocess.run(
        ["git", "init", "-q", str(root)], check=True, capture_output=True
    )
    listing = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "-uall"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    untracked = {line[3:] for line in listing.splitlines() if line.startswith("?? ")}
    return relative not in untracked


CROSS_CHECK_CASES = [
    pytest.param(
        {
            ".gitignore": "*.log\n",
            "frontend/.gitignore": "!important.log\n",
            "frontend/important.log": "keep\n",
        },
        "frontend/important.log",
        id="nested-negation",
    ),
    pytest.param(
        {
            ".gitignore": "*.log\n",
            "frontend/.gitignore": "!important.log\n",
            "frontend/other.log": "drop\n",
        },
        "frontend/other.log",
        id="nested-negation-does-not-widen",
    ),
    pytest.param(
        {
            ".gitignore": "*.txt\n",
            "a/.gitignore": "!notes.txt\n",
            "a/b/.gitignore": "notes.txt\n",
            "a/b/notes.txt": "drop\n",
        },
        "a/b/notes.txt",
        id="deepest-wins",
    ),
    pytest.param(
        {
            ".gitignore": "build/\n",
            "build/.gitignore": "!keep.txt\n",
            "build/keep.txt": "drop\n",
        },
        "build/keep.txt",
        id="excluded-directory-is-final",
    ),
    pytest.param(
        {
            ".gitignore": "/dist\n",
            "packages/dist/app.js": "keep\n",
        },
        "packages/dist/app.js",
        id="root-anchored",
    ),
    pytest.param(
        {
            ".gitignore": "*.log\n",
            "src/main.py": "print(1)\n",
        },
        "src/main.py",
        id="unmatched",
    ),
]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
@pytest.mark.parametrize("files,relative", CROSS_CHECK_CASES)
def test_matches_the_git_binary(tmp_path, files, relative):
    root = _repo(tmp_path, files)

    assert _ignored(root, relative) == _git_ignores(root, relative)
