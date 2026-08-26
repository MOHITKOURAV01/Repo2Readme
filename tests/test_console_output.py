"""The README the CLI prints has to be the README the model produced.

From the issue: without ``--output`` the finished README is printed so it can be
redirected, and it went through ``rich.print``. Rich parses ``[...]`` as a style
tag, so ``cfg[key]`` came out as ``cfg``, ``items[i]`` came out as ``items``
followed by italics, and every long line was hard-wrapped at 80 columns. A
table of contents, a ``[!NOTE]`` callout and a Next.js ``src/[id]/page.tsx``
path are all made of exactly those characters.
"""

import importlib
import io

import pytest
from click.testing import CliRunner

from repo2readme.utils.console import (
    echo_document,
    echo_documents,
    notify,
    safe,
    status_console,
)

cli_main = importlib.import_module("repo2readme.cli.main")


README = """# Sample Project

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)

> [!NOTE]
> Values are read from cfg[key] at startup.

## Installation

```python
items[i] = load(paths[0])
```

## Usage

A description line that is long enough that a console eighty columns wide would
break it in half, which is precisely what must not happen to a redirected file.
"""


# ---------------------------------------------------------------------------
# safe()
# ---------------------------------------------------------------------------


def test_safe_escapes_square_brackets():
    assert safe("cfg[key]") != "cfg[key]"
    assert "cfg" in safe("cfg[key]")


def test_safe_accepts_non_strings():
    assert safe(7) == "7"
    assert safe(None) == ""
    assert safe(ValueError("boom [i]")) .startswith("boom ")


def test_safe_output_survives_a_rich_render():
    from rich.console import Console

    buffer = io.StringIO()
    Console(file=buffer, width=200).print(f"[green]{safe('src/[id]/page.tsx')}[/green]")
    assert "src/[id]/page.tsx" in buffer.getvalue()


def test_safe_neutralises_an_unclosed_tag():
    from rich.console import Console

    buffer = io.StringIO()
    # Without escaping this raises rich.errors.MarkupError.
    Console(file=buffer, width=200).print(f"[red]{safe('unterminated [bold')}[/red]")
    assert "unterminated [bold" in buffer.getvalue()


# ---------------------------------------------------------------------------
# echo_document()
# ---------------------------------------------------------------------------


def test_echo_document_writes_the_text_unchanged():
    buffer = io.StringIO()
    echo_document(README, file=buffer)
    assert buffer.getvalue() == README


def test_echo_document_does_not_wrap_long_lines():
    line = "x" * 400
    buffer = io.StringIO()
    echo_document(line, file=buffer)
    assert buffer.getvalue() == line + "\n"


def test_echo_document_adds_one_trailing_newline_at_most():
    buffer = io.StringIO()
    echo_document("no newline", file=buffer)
    assert buffer.getvalue() == "no newline\n"

    buffer = io.StringIO()
    echo_document("has one\n", file=buffer)
    assert buffer.getvalue() == "has one\n"


def test_echo_document_leaves_an_empty_document_alone():
    buffer = io.StringIO()
    echo_document("", file=buffer)
    assert buffer.getvalue() == ""


def test_echo_documents_writes_in_order():
    buffer = io.StringIO()
    echo_documents("first", "second", file=buffer)
    assert buffer.getvalue() == "first\nsecond\n"


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def _stub_pipeline(monkeypatch, readme=README):
    """Run the CLI as far as printing, without touching a provider."""
    monkeypatch.setattr(cli_main, "setup_api_keys", lambda provider: None)
    monkeypatch.setattr(
        cli_main,
        "generate_all_summaries",
        lambda **kwargs: ([{"file_path": "main.py", "description": "d"}], []),
    )
    monkeypatch.setattr(
        cli_main,
        "generate_hierarchical_summaries",
        lambda **kwargs: [{"file_path": "main.py", "description": "d"}],
    )
    monkeypatch.setattr(cli_main, "run_pipeline", lambda **kwargs: readme)


def test_status_output_goes_to_stderr(capsys):
    notify("[green]Saved[/green]")
    captured = capsys.readouterr()
    assert "Saved" in captured.err
    assert captured.out == ""


def test_the_status_console_is_a_stderr_console():
    assert status_console().stderr is True


def test_redirected_stdout_is_exactly_the_readme(monkeypatch, tmp_path):
    """What `repo2readme run --local . > README.md` actually writes.

    The commentary - the token estimate, the progress bar, "Generated README:" -
    was on stdout too, so it landed in the file alongside the README.
    """
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
    _stub_pipeline(monkeypatch)

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--force"]
    )

    assert result.exit_code == 0
    assert result.stdout == README


def test_the_commentary_is_still_shown(monkeypatch, tmp_path):
    """Moved, not removed: the user still sees what the run is doing."""
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
    _stub_pipeline(monkeypatch)

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--force"]
    )

    assert "Repository Analysis" in result.stderr
    assert "Generated README" in result.stderr


def test_redirected_stdout_after_a_failed_write_is_exactly_the_readme(
    monkeypatch, tmp_path
):
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
    _stub_pipeline(monkeypatch)

    def refuse(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(cli_main, "write_readme", refuse)

    result = CliRunner().invoke(
        cli_main.main,
        ["run", "--local", str(tmp_path), "--force", "--output", str(tmp_path / "OUT.md")],
    )

    assert result.exit_code == 1
    assert result.stdout == README
    assert "Printing the README instead." in result.stderr


def test_writing_to_a_file_leaves_stdout_empty(monkeypatch, tmp_path):
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
    _stub_pipeline(monkeypatch)
    destination = tmp_path / "OUT.md"

    result = CliRunner().invoke(
        cli_main.main,
        ["run", "--local", str(tmp_path), "--force", "--output", str(destination)],
    )

    assert result.exit_code == 0
    assert result.stdout == ""
    assert destination.read_text(encoding="utf-8") == README
    assert "Saved to" in result.stderr


def test_the_confirmation_prompt_does_not_reach_stdout(monkeypatch, tmp_path):
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
    _stub_pipeline(monkeypatch)

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path)], input="y\n"
    )

    assert result.exit_code == 0
    assert "Proceed?" in result.stderr
    assert "Proceed?" not in result.stdout
    # CliRunner echoes the simulated keystrokes onto stdout itself, so compare
    # against what the command wrote rather than against the whole stream.
    assert result.stdout.endswith(README)


def test_the_dry_run_report_stays_on_stdout(tmp_path):
    """It is what --dry-run produces, so it is the product, not commentary."""
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--dry-run"]
    )

    assert result.exit_code == 0
    assert "Repository Analysis" in result.stdout
    assert "Files to be processed" in result.stdout
    assert "Dry run complete." in result.stdout


def test_a_failure_message_goes_to_stderr(tmp_path):
    result = CliRunner().invoke(cli_main.main, ["run", "--local", "/nope/not/here"])
    assert "Failed to load repository" in result.stderr
    assert "Failed to load repository" not in result.stdout


def test_printed_readme_is_byte_for_byte_the_generated_one(monkeypatch, tmp_path):
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
    _stub_pipeline(monkeypatch)

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--force"]
    )

    assert result.exit_code == 0
    assert README in result.output


@pytest.mark.parametrize(
    "fragment",
    [
        "- [Installation](#installation)",
        "> [!NOTE]",
        "cfg[key]",
        "items[i] = load(paths[0])",
        "would\nbreak it in half",
    ],
)
def test_printed_readme_keeps_every_bracketed_construct(
    monkeypatch, tmp_path, fragment
):
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
    _stub_pipeline(monkeypatch)

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--force"]
    )

    assert fragment in result.output


def test_readme_printed_after_a_failed_write_is_intact(monkeypatch, tmp_path):
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
    _stub_pipeline(monkeypatch)

    def refuse(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(cli_main, "write_readme", refuse)

    result = CliRunner().invoke(
        cli_main.main,
        ["run", "--local", str(tmp_path), "--force", "--output", str(tmp_path / "OUT.md")],
    )

    assert result.exit_code == 1
    assert "Printing the README instead." in result.output
    assert README in result.output


def test_dry_run_lists_a_path_containing_brackets(tmp_path):
    route = tmp_path / "src" / "[id]"
    route.mkdir(parents=True)
    (route / "page.tsx").write_text("export default () => null\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--dry-run"]
    )

    assert result.exit_code == 0
    assert "src/[id]/page.tsx" in result.output


def test_dry_run_tree_keeps_a_directory_named_with_brackets(tmp_path):
    route = tmp_path / "app" / "[slug]"
    route.mkdir(parents=True)
    (route / "route.ts").write_text("export {}\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--dry-run"]
    )

    assert result.exit_code == 0
    assert "[slug]/" in result.output


def test_a_repository_error_containing_markup_is_reported_not_swallowed(
    monkeypatch, tmp_path
):
    class Exploding:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, return_skip_info=False):
            raise RuntimeError("Failed to clone repository: fatal [red]not found")

    monkeypatch.setattr(cli_main, "RepoLoader", Exploding)

    result = CliRunner().invoke(cli_main.main, ["run", "--local", str(tmp_path)])

    assert "fatal [red]not found" in result.output


def test_summarization_report_keeps_bracketed_paths(monkeypatch, tmp_path):
    from repo2readme.services.reporting import SummaryFailure, build_report_lines

    lines = build_report_lines(
        total=2,
        succeeded=1,
        failures=[SummaryFailure(file_path="src/[id]/page.tsx", reason="rate limit [429]")],
    )
    rendered = "\n".join(lines)

    from rich.console import Console

    buffer = io.StringIO()
    console = Console(file=buffer, width=200)
    for line in lines:
        console.print(line)
    output = buffer.getvalue()

    assert "src/[id]/page.tsx" in output
    assert "rate limit [429]" in output
    assert "[yellow]" not in output  # the CLI's own tags still render
    assert rendered  # the lines themselves are still markup
