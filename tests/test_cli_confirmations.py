"""
Tests for the two confirmations `run` asks for (issue #130).

Covers:
- --yes accepts the token estimate and nothing else
- --force still does both, so existing command lines keep working
- neither flag skips the other's guard by accident
- a confirmation nobody can answer names the flag that would have answered it
- the helpers on their own
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch

import pytest
from click.testing import CliRunner

# repo2readme.cli re-exports the click group as ``main``, which shadows the
# submodule for a plain ``import ... as``; the rest of the suite resolves it
# the same way.
cli_main = importlib.import_module("repo2readme.cli.main")

NonInteractiveError = cli_main.NonInteractiveError
confirm_estimate = cli_main.confirm_estimate
confirm_overwrite = cli_main.confirm_overwrite
run = cli_main.run

EXISTING_README = "# Written by a human\n\nPlease do not clobber me.\n"
GENERATED_README = "# Generated\n\ntext\n"


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "app.py").write_text("print('hi')\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def invoke(repo):
    """Run the CLI with every API-touching stage replaced, counting the calls."""
    calls = {"summarize": 0}

    def fake_summaries(documents, **kwargs):
        calls["summarize"] += 1
        return (
            [
                {"file_path": doc["metadata"]["file_path"], "description": "x"}
                for doc in documents
            ],
            [],
        )

    def run_cli(args, stdin=""):
        with patch.object(
            cli_main, "generate_all_summaries", side_effect=fake_summaries
        ), patch.object(
            cli_main,
            "generate_hierarchical_summaries",
            side_effect=lambda file_summaries, **kwargs: file_summaries,
        ), patch.object(
            cli_main, "run_pipeline", return_value=GENERATED_README
        ), patch.object(
            cli_main, "setup_api_keys"
        ):
            return CliRunner().invoke(
                run, ["--local", str(repo)] + args, input=stdin
            )

    run_cli.calls = calls
    return run_cli


def _output(repo):
    return str(repo / "README.md")


def _write_existing(repo):
    path = repo / "README.md"
    path.write_text(EXISTING_README, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# --yes
# ---------------------------------------------------------------------------


class TestYesFlag:
    def test_accepts_the_estimate_without_a_prompt(self, invoke, repo):
        result = invoke(["--yes", "-o", _output(repo)])
        assert result.exit_code == 0
        assert "Proceed?" not in result.output
        assert invoke.calls["summarize"] == 1

    def test_short_form(self, invoke, repo):
        result = invoke(["-y", "-o", _output(repo)])
        assert result.exit_code == 0
        assert invoke.calls["summarize"] == 1

    def test_does_not_overwrite_an_existing_file(self, invoke, repo):
        """The whole point: approve the spend, keep the guard on the file."""
        path = _write_existing(repo)
        result = invoke(["--yes", "-o", path], stdin="n\n")

        assert "already exists" in result.output
        assert open(path, encoding="utf-8").read() == EXISTING_README

    def test_overwrites_when_the_prompt_is_answered(self, invoke, repo):
        path = _write_existing(repo)
        result = invoke(["--yes", "-o", path], stdin="y\n")

        assert result.exit_code == 0
        assert open(path, encoding="utf-8").read() == GENERATED_README

    def test_writes_a_new_file_without_any_prompt(self, invoke, repo):
        path = _output(repo)
        result = invoke(["--yes", "-o", path])

        assert result.exit_code == 0
        assert "already exists" not in result.output
        assert open(path, encoding="utf-8").read() == GENERATED_README


# ---------------------------------------------------------------------------
# --force
# ---------------------------------------------------------------------------


class TestForceFlag:
    def test_still_answers_both_confirmations(self, invoke, repo):
        """Existing command lines must keep working exactly as before."""
        path = _write_existing(repo)
        result = invoke(["--force", "-o", path])

        assert result.exit_code == 0
        assert "Proceed?" not in result.output
        assert "already exists" not in result.output
        assert open(path, encoding="utf-8").read() == GENERATED_README

    def test_short_form(self, invoke, repo):
        path = _write_existing(repo)
        result = invoke(["-f", "-o", path])
        assert result.exit_code == 0
        assert open(path, encoding="utf-8").read() == GENERATED_README

    def test_help_mentions_both_effects(self):
        """The old text described only the overwrite, which was half the story."""
        help_text = " ".join(CliRunner().invoke(run, ["--help"]).output.split())
        assert "Overwrite the output file without confirmation" in help_text
        assert "Implies --yes" in help_text
        assert "token estimate is not confirmed" in help_text

    def test_help_lists_the_yes_flag(self):
        result = CliRunner().invoke(run, ["--help"])
        assert "--yes" in result.output
        assert "-y," in result.output or "-y " in result.output


# ---------------------------------------------------------------------------
# Neither flag
# ---------------------------------------------------------------------------


class TestWithoutFlags:
    def test_declining_the_estimate_makes_no_api_call(self, invoke, repo):
        result = invoke(["-o", _output(repo)], stdin="n\n")

        assert result.exit_code == 0
        assert invoke.calls["summarize"] == 0
        assert "Operation cancelled" in result.output
        assert not os.path.exists(_output(repo))

    def test_accepting_both_prompts_writes_the_file(self, invoke, repo):
        path = _write_existing(repo)
        result = invoke(["-o", path], stdin="y\ny\n")

        assert result.exit_code == 0
        assert open(path, encoding="utf-8").read() == GENERATED_README

    def test_piped_input_still_works(self, invoke, repo):
        """`echo y | repo2readme run ...` is a legitimate way to automate this."""
        result = invoke(["-o", _output(repo)], stdin="y\n")
        assert result.exit_code == 0
        assert invoke.calls["summarize"] == 1


# ---------------------------------------------------------------------------
# Nothing on stdin
# ---------------------------------------------------------------------------


class TestUnanswerableConfirmations:
    def test_estimate_names_the_flag_and_spends_nothing(self, invoke, repo):
        result = invoke(["-o", _output(repo)])

        assert result.exit_code != 0
        assert invoke.calls["summarize"] == 0
        assert "--yes" in result.output
        assert not os.path.exists(_output(repo))

    def test_overwrite_names_the_flag_and_keeps_the_file(self, invoke, repo):
        path = _write_existing(repo)
        result = invoke(["--yes", "-o", path])

        assert result.exit_code != 0
        assert "--force" in result.output
        assert open(path, encoding="utf-8").read() == EXISTING_README

    def test_the_message_is_not_a_bare_abort(self, invoke, repo):
        result = invoke(["-o", _output(repo)])
        assert "Error:" in result.output
        assert "estimate" in result.output


# ---------------------------------------------------------------------------
# The helpers on their own
# ---------------------------------------------------------------------------


class TestConfirmEstimate:
    def test_assume_yes_short_circuits(self):
        with patch("click.confirm") as confirm:
            assert confirm_estimate(True) is True
        confirm.assert_not_called()

    def test_yes(self):
        with patch("click.confirm", return_value=True):
            assert confirm_estimate(False) is True

    def test_no(self):
        with patch("click.confirm", return_value=False):
            assert confirm_estimate(False) is False

    def test_eof_becomes_an_actionable_error(self):
        import click

        with patch("click.confirm", side_effect=click.Abort()):
            with pytest.raises(NonInteractiveError) as excinfo:
                confirm_estimate(False)

        assert "--yes" in str(excinfo.value)


class TestConfirmOverwrite:
    def test_yes(self):
        with patch("click.confirm", return_value=True):
            assert confirm_overwrite("README.md") is True

    def test_no(self):
        with patch("click.confirm", return_value=False):
            assert confirm_overwrite("README.md") is False

    def test_eof_becomes_an_actionable_error(self):
        import click

        with patch("click.confirm", side_effect=click.Abort()):
            with pytest.raises(NonInteractiveError) as excinfo:
                confirm_overwrite("README.md")

        message = str(excinfo.value)
        assert "--force" in message
        assert "README.md" in message

    def test_the_path_is_named_in_the_question(self):
        with patch("click.confirm", return_value=True) as confirm:
            confirm_overwrite("docs/README.md")
        assert "docs/README.md" in confirm.call_args[0][0]


# ---------------------------------------------------------------------------
# Interaction with the other flags
# ---------------------------------------------------------------------------


class TestOtherFlagsAreUnaffected:
    def test_dry_run_asks_for_nothing(self, invoke, repo):
        result = invoke(["--dry-run"])

        assert result.exit_code == 0
        assert "Proceed?" not in result.output
        assert invoke.calls["summarize"] == 0
        assert "No API requests were made" in result.output

    def test_stdout_output_never_asks_about_overwriting(self, invoke, repo):
        result = invoke(["--yes"])

        assert result.exit_code == 0
        assert "already exists" not in result.output
