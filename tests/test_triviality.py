"""Tests for the content-emptiness rule that replaced the __init__.py ignore."""

import pytest

from repo2readme.loaders.traversal.pipeline import TraversalPipeline
from repo2readme.utils.filter import (
    IGNORE_FILES,
    classify_default_ignore,
    github_file_filter,
)
from repo2readme.utils.triviality import (
    LINE_COMMENT_MARKERS,
    SKIP_REASON,
    is_effectively_empty,
    strip_comments,
)


# ---------------------------------------------------------------------------
# The filter no longer knows the name
# ---------------------------------------------------------------------------


def test_init_py_is_not_in_the_name_based_ignore_list():
    assert "__init__.py" not in IGNORE_FILES


@pytest.mark.parametrize(
    "path",
    [
        "__init__.py",
        "pkg/__init__.py",
        "src/deeply/nested/pkg/__init__.py",
    ],
)
def test_init_py_is_no_longer_ignored_by_default(path):
    assert classify_default_ignore(path) is None
    allowed, _ = github_file_filter(path, max_file_size_kb=None)
    assert allowed is True


def test_directory_rules_still_win_over_init_py():
    """An __init__.py inside an ignored directory stays ignored."""
    assert classify_default_ignore("node_modules/pkg/__init__.py") == "build_directory"
    assert classify_default_ignore(".venv/lib/pkg/__init__.py") == "build_directory"


def test_the_other_ignore_names_are_untouched():
    for name in ("package-lock.json", "yarn.lock", ".gitignore", ".env"):
        assert classify_default_ignore(name) is not None


# ---------------------------------------------------------------------------
# is_effectively_empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("content", ["", "\n", "   ", "\n\n\t\n", "\r\n\r\n"])
def test_blank_content_is_empty(content):
    assert is_effectively_empty(content) is True


def test_none_is_empty():
    assert is_effectively_empty(None) is True


@pytest.mark.parametrize(
    "content, language",
    [
        ("# intentionally empty\n", "python"),
        ("#\n#\n#\n", "python"),
        ("   # indented comment\n", "python"),
        ("// nothing here\n", "javascript"),
        ("/* license header */\n", "javascript"),
        ("/*\n * Copyright 2020\n */\n", "typescript"),
        ("<!-- generated -->\n", "html"),
        ("-- no tables yet\n", "sql"),
        ("REM placeholder\n", "batch"),
        ("; section marker\n", "ini"),
    ],
)
def test_comment_only_files_are_empty(content, language):
    assert is_effectively_empty(content, language) is True


@pytest.mark.parametrize(
    "content, language",
    [
        ("x = 1\n", "python"),
        ("# a comment\nx = 1\n", "python"),
        ("from .client import Client\n", "python"),
        ('__all__ = ["Client"]\n', "python"),
        ("export const x = 1;\n", "javascript"),
        ("/* header */\nconst x = 1;\n", "javascript"),
        ("SELECT 1;\n", "sql"),
    ],
)
def test_files_with_real_content_are_not_empty(content, language):
    assert is_effectively_empty(content, language) is False


def test_a_docstring_is_content_not_a_comment():
    """The package docstring is the most useful line in an __init__.py."""
    assert is_effectively_empty('"""Order service."""\n', "python") is False
    assert is_effectively_empty("'''Order service.'''\n", "python") is False


def test_a_markdown_heading_is_not_a_comment():
    """`#` opens a heading in Markdown, not a comment."""
    assert is_effectively_empty("# Sample\n", "markdown") is False


def test_an_unknown_language_keeps_its_comments():
    """Guessing wrong should cost a request, never a file."""
    assert is_effectively_empty("# a note\n", "some-new-language") is False
    assert is_effectively_empty("# a note\n", None) is False


def test_a_trailing_comment_does_not_remove_the_code():
    assert is_effectively_empty("x = 1  # set up\n", "python") is False


def test_language_matching_is_case_insensitive():
    assert is_effectively_empty("# c\n", "Python") is True


# ---------------------------------------------------------------------------
# strip_comments
# ---------------------------------------------------------------------------


def test_strip_comments_keeps_code_lines():
    source = "# header\nimport os\n# trailer\nprint(os)\n"
    assert strip_comments(source, "python") == "import os\nprint(os)"


def test_strip_comments_is_a_noop_for_an_unmapped_language():
    source = "# header\ncode\n"
    assert strip_comments(source, "brainfuck") == source


def test_strip_comments_handles_empty_input():
    assert strip_comments("", "python") == ""


def test_every_marker_table_entry_is_a_tuple():
    for language, markers in LINE_COMMENT_MARKERS.items():
        assert isinstance(markers, tuple), language


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_an_init_py_with_an_api_is_analyzed(tmp_path):
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        '"""Order service."""\nfrom app.client import Client\n__all__ = ["Client"]\n',
        encoding="utf-8",
    )
    (pkg / "client.py").write_text("class Client:\n    pass\n", encoding="utf-8")

    documents, ctx = TraversalPipeline(str(tmp_path)).run()

    paths = sorted(doc.metadata["relative_path"] for doc in documents)
    assert paths == ["app/__init__.py", "app/client.py"]


def test_an_empty_init_py_is_skipped_with_a_content_reason(tmp_path):
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "client.py").write_text("class Client:\n    pass\n", encoding="utf-8")

    documents, ctx = TraversalPipeline(str(tmp_path)).run()

    assert [doc.metadata["relative_path"] for doc in documents] == ["app/client.py"]
    assert ("app/__init__.py", SKIP_REASON) in ctx.skipped


def test_a_marker_init_py_holding_only_a_comment_is_skipped(tmp_path):
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("# intentionally empty\n", encoding="utf-8")
    (pkg / "client.py").write_text("class Client:\n    pass\n", encoding="utf-8")

    documents, ctx = TraversalPipeline(str(tmp_path)).run()

    assert [doc.metadata["relative_path"] for doc in documents] == ["app/client.py"]
    assert ("app/__init__.py", SKIP_REASON) in ctx.skipped


def test_the_rule_is_not_about_the_name(tmp_path):
    """A zero-byte file of any name costs nothing either."""
    (tmp_path / "conftest.py").write_text("", encoding="utf-8")
    (tmp_path / "helpers.js").write_text("// TODO\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("print(1)\n", encoding="utf-8")

    documents, ctx = TraversalPipeline(str(tmp_path)).run()

    assert [doc.metadata["relative_path"] for doc in documents] == ["main.py"]
    skipped = dict(ctx.skipped)
    assert skipped["conftest.py"] == SKIP_REASON
    assert skipped["helpers.js"] == SKIP_REASON


def test_a_readme_of_one_heading_is_kept(tmp_path):
    (tmp_path / "README.md").write_text("# Sample\n", encoding="utf-8")
    documents, _ = TraversalPipeline(str(tmp_path)).run()
    assert [doc.metadata["relative_path"] for doc in documents] == ["README.md"]
