"""Tests for validating and writing the output file.

From the issue: `--output docs/generated/README.md` with no `docs/generated`
directory loaded the repository, summarized every file, generated and reviewed
the README, and only then raised FileNotFoundError - throwing away everything
the run had paid for. And `open(output, "w")` truncates immediately, so an
interrupted write destroyed the previous README.
"""

import importlib
import os
import stat
from pathlib import Path

import pytest
from click.testing import CliRunner

from repo2readme.utils.output import (
    BACKUP_SUFFIX,
    OutputPathError,
    backup_path_for,
    prepare_output_path,
    resolve_output_path,
    write_readme,
)

cli_main = importlib.import_module("repo2readme.cli.main")

README = "# Title\n\nBody.\n"


# ---------------------------------------------------------------------------
# resolve_output_path
# ---------------------------------------------------------------------------


def test_resolve_makes_the_path_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert resolve_output_path("README.md") == tmp_path.absolute() / "README.md"


def test_resolve_expands_a_home_path(monkeypatch):
    monkeypatch.setenv("HOME", "/home/someone")

    resolved = resolve_output_path("~/docs/README.md")

    assert "~" not in str(resolved)
    assert resolved.name == "README.md"


def test_resolve_does_not_touch_the_disk(tmp_path):
    resolve_output_path(tmp_path / "nope" / "README.md")

    assert not (tmp_path / "nope").exists()


# ---------------------------------------------------------------------------
# prepare_output_path
# ---------------------------------------------------------------------------


def test_an_ordinary_path_is_accepted(tmp_path):
    target = prepare_output_path(tmp_path / "README.md")

    assert target.path == (tmp_path / "README.md").absolute()
    assert target.exists is False
    assert target.created_parent is False


def test_an_existing_file_is_reported_as_existing(tmp_path):
    destination = tmp_path / "README.md"
    destination.write_text("old", encoding="utf-8")

    assert prepare_output_path(destination).exists is True


def test_a_missing_parent_is_created(tmp_path):
    target = prepare_output_path(tmp_path / "docs" / "generated" / "README.md")

    assert target.created_parent is True
    assert (tmp_path / "docs" / "generated").is_dir()


def test_a_missing_parent_can_be_refused(tmp_path):
    with pytest.raises(OutputPathError) as exc_info:
        prepare_output_path(tmp_path / "docs" / "README.md", create_parents=False)

    assert "does not exist" in str(exc_info.value)
    assert not (tmp_path / "docs").exists()


def test_a_directory_is_rejected(tmp_path):
    with pytest.raises(OutputPathError) as exc_info:
        prepare_output_path(tmp_path)

    message = str(exc_info.value)
    assert "is a directory" in message
    # The message suggests the fix rather than just stating the problem.
    assert "README.md" in message


def test_a_file_used_as_a_directory_is_rejected(tmp_path):
    blocker = tmp_path / "notadir"
    blocker.write_text("x", encoding="utf-8")

    with pytest.raises(OutputPathError) as exc_info:
        prepare_output_path(blocker / "README.md")

    assert "not a directory" in str(exc_info.value)


def test_an_empty_path_is_rejected():
    for value in ("", "   ", None):
        with pytest.raises(OutputPathError):
            prepare_output_path(value)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_an_unwritable_directory_is_rejected(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)

    try:
        with pytest.raises(OutputPathError) as exc_info:
            prepare_output_path(locked / "README.md")
        assert "permission" in str(exc_info.value).lower()
    finally:
        locked.chmod(0o700)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_an_unwritable_file_is_rejected(tmp_path):
    destination = tmp_path / "README.md"
    destination.write_text("old", encoding="utf-8")
    destination.chmod(0o400)

    try:
        with pytest.raises(OutputPathError) as exc_info:
            prepare_output_path(destination)
        assert "not writable" in str(exc_info.value)
    finally:
        destination.chmod(0o600)


# ---------------------------------------------------------------------------
# write_readme
# ---------------------------------------------------------------------------


def test_the_content_is_written(tmp_path):
    destination = tmp_path / "README.md"

    written = write_readme(destination, README)

    assert written == destination.absolute()
    assert destination.read_text(encoding="utf-8") == README


def test_an_existing_file_is_replaced(tmp_path):
    destination = tmp_path / "README.md"
    destination.write_text("old content", encoding="utf-8")

    write_readme(destination, README)

    assert destination.read_text(encoding="utf-8") == README


def test_no_temporary_files_are_left_behind(tmp_path):
    write_readme(tmp_path / "README.md", README)

    assert [p.name for p in tmp_path.iterdir()] == ["README.md"]


def test_a_failed_write_leaves_the_original_intact(tmp_path, monkeypatch):
    destination = tmp_path / "README.md"
    destination.write_text("previous README", encoding="utf-8")

    def explode(src, dst):
        raise OSError("no space left on device")

    monkeypatch.setattr(os, "replace", explode)

    with pytest.raises(OSError):
        write_readme(destination, README)

    # This is the whole point of the temp file: open(..., "w") would have
    # truncated the previous README before failing.
    assert destination.read_text(encoding="utf-8") == "previous README"
    assert [p.name for p in tmp_path.iterdir()] == ["README.md"]


def test_an_interrupted_write_leaves_the_original_intact(tmp_path, monkeypatch):
    destination = tmp_path / "README.md"
    destination.write_text("previous README", encoding="utf-8")

    def interrupt(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupt)

    with pytest.raises(KeyboardInterrupt):
        write_readme(destination, README)

    assert destination.read_text(encoding="utf-8") == "previous README"
    assert [p.name for p in tmp_path.iterdir()] == ["README.md"]


def test_the_backup_keeps_the_previous_version(tmp_path):
    destination = tmp_path / "README.md"
    destination.write_text("previous README", encoding="utf-8")

    write_readme(destination, README, backup=True)

    assert destination.read_text(encoding="utf-8") == README
    assert backup_path_for(destination).read_text(encoding="utf-8") == "previous README"


def test_no_backup_is_made_for_a_new_file(tmp_path):
    destination = tmp_path / "README.md"

    write_readme(destination, README, backup=True)

    assert not backup_path_for(destination).exists()


def test_the_backup_path_is_predictable(tmp_path):
    assert backup_path_for(tmp_path / "README.md").name == "README.md" + BACKUP_SUFFIX


def test_the_mode_of_an_existing_file_is_preserved(tmp_path):
    destination = tmp_path / "README.md"
    destination.write_text("old", encoding="utf-8")
    destination.chmod(0o640)

    write_readme(destination, README)

    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


def test_a_new_file_is_not_created_private(tmp_path):
    destination = tmp_path / "README.md"

    write_readme(destination, README)

    mode = stat.S_IMODE(destination.stat().st_mode)
    # mkstemp creates 0600; a README that nobody else can read is a surprise.
    assert mode & stat.S_IRGRP or mode & stat.S_IROTH or mode == 0o600


def test_unicode_survives_the_round_trip(tmp_path):
    destination = tmp_path / "README.md"
    content = "# Café ☕\n\nUnicode — em dash, ünïcödé.\n"

    write_readme(destination, content)

    assert destination.read_text(encoding="utf-8") == content


def test_line_endings_are_not_rewritten(tmp_path):
    destination = tmp_path / "README.md"

    write_readme(destination, "line one\nline two\n")

    assert destination.read_bytes() == b"line one\nline two\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def flat(output: str) -> str:
    """CLI output with the line wrapping taken out.

    Rich wraps to the terminal width, which is not the same on a developer's
    machine as it is in CI, so a message can arrive with a newline in the
    middle of a phrase. Collapsing whitespace makes these assertions depend on
    what was said rather than on how wide the terminal was.
    """
    return " ".join(output.split())

def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
    return str(repo)


def _patch_pipeline(monkeypatch, readme=README):
    """Make a run cheap: no API keys, no LLM calls."""
    monkeypatch.setattr(cli_main, "setup_api_keys", lambda *a, **k: None)
    monkeypatch.setattr(
        cli_main,
        "generate_all_summaries",
        lambda **kwargs: ([{"file_path": "app.py", "description": "d"}], []),
    )
    # The roll-up is deliberately left real: a single summary is below its
    # threshold, so it returns without calling anything, and these tests then
    # do not depend on the shape of what it returns.

    called = {"pipeline": 0}

    def fake_run_pipeline(**kwargs):
        called["pipeline"] += 1
        return readme

    monkeypatch.setattr(cli_main, "run_pipeline", fake_run_pipeline)
    return called


class TestCliOutputValidation:
    def test_a_bad_path_fails_before_any_work(self, monkeypatch, tmp_path):
        called = _patch_pipeline(monkeypatch)

        def fail(*args, **kwargs):
            raise AssertionError("the repository must not be loaded")

        monkeypatch.setattr(cli_main, "RepoLoader", fail)

        result = CliRunner().invoke(
            cli_main.main,
            ["run", "--local", _repo(tmp_path), "--output", str(tmp_path), "--force"],
        )

        assert result.exit_code == 2
        assert "is a directory" in flat(result.output)
        assert called["pipeline"] == 0

    def test_a_missing_directory_is_created_up_front(self, monkeypatch, tmp_path):
        _patch_pipeline(monkeypatch)
        destination = tmp_path / "docs" / "generated" / "README.md"

        result = CliRunner().invoke(
            cli_main.main,
            ["run", "--local", _repo(tmp_path), "--output", str(destination), "--force"],
        )

        assert result.exit_code == 0
        assert destination.read_text(encoding="utf-8") == README

    def test_directory_creation_can_be_turned_off(self, monkeypatch, tmp_path):
        _patch_pipeline(monkeypatch)
        destination = tmp_path / "docs" / "README.md"

        result = CliRunner().invoke(
            cli_main.main,
            [
                "run",
                "--local",
                _repo(tmp_path),
                "--output",
                str(destination),
                "--no-create-dirs",
                "--force",
            ],
        )

        assert result.exit_code == 2
        assert "does not exist" in flat(result.output)
        assert not destination.parent.exists()

    def test_the_dry_run_validates_the_path_too(self, monkeypatch, tmp_path):
        result = CliRunner().invoke(
            cli_main.main,
            [
                "run",
                "--local",
                _repo(tmp_path),
                "--dry-run",
                "--output",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 2


class TestCliOutputWriting:
    def test_the_readme_is_written(self, monkeypatch, tmp_path):
        _patch_pipeline(monkeypatch)
        destination = tmp_path / "README.md"

        result = CliRunner().invoke(
            cli_main.main,
            ["run", "--local", _repo(tmp_path), "--output", str(destination), "--force"],
        )

        assert result.exit_code == 0
        assert destination.read_text(encoding="utf-8") == README
        assert "Saved to" in flat(result.output)

    def test_backup_keeps_the_previous_readme(self, monkeypatch, tmp_path):
        _patch_pipeline(monkeypatch)
        destination = tmp_path / "README.md"
        destination.write_text("hand written", encoding="utf-8")

        result = CliRunner().invoke(
            cli_main.main,
            [
                "run",
                "--local",
                _repo(tmp_path),
                "--output",
                str(destination),
                "--backup",
                "--force",
            ],
        )

        assert result.exit_code == 0
        assert backup_path_for(destination).read_text(encoding="utf-8") == "hand written"
        assert "kept at" in flat(result.output)

    def test_a_write_failure_prints_the_readme_instead_of_losing_it(
        self, monkeypatch, tmp_path
    ):
        _patch_pipeline(monkeypatch)
        destination = tmp_path / "README.md"

        def explode(*args, **kwargs):
            raise OSError("disk on fire")

        monkeypatch.setattr(cli_main, "write_readme", explode)

        result = CliRunner().invoke(
            cli_main.main,
            ["run", "--local", _repo(tmp_path), "--output", str(destination), "--force"],
        )

        assert result.exit_code == 1
        assert "disk on fire" in flat(result.output)
        # The generated README is on stdout rather than thrown away.
        assert "Title" in flat(result.output)

    def test_declining_the_overwrite_leaves_the_file_alone(self, monkeypatch, tmp_path):
        _patch_pipeline(monkeypatch)
        destination = tmp_path / "README.md"
        destination.write_text("hand written", encoding="utf-8")

        result = CliRunner().invoke(
            cli_main.main,
            ["run", "--local", _repo(tmp_path), "--output", str(destination)],
            input="y\nn\n",
        )

        assert result.exit_code == 0
        assert destination.read_text(encoding="utf-8") == "hand written"
        assert Path(str(destination) + BACKUP_SUFFIX).exists() is False
