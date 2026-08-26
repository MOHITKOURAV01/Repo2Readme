"""A failed run has to tell the shell it failed.

From the issue: every early exit from ``run()`` was a bare ``return``, so a
missing source, a repository that would not load and an API key that would not
configure all ended with status 0. A CI job that regenerates a README then
commits an unchanged file and stays green. The clone directory was leaked on
the same path, because the loader raised before the CLI was ever handed the
object that knows how to remove it.
"""

import glob
import importlib
import os
import subprocess
import tempfile

import pytest
from click.testing import CliRunner

from repo2readme.cli.exit_codes import (
    EXIT_CODE_MEANINGS,
    ExitCode,
    describe,
    fail,
)
from repo2readme.loaders.loader import UrlRepoLoader

cli_main = importlib.import_module("repo2readme.cli.main")


# ---------------------------------------------------------------------------
# The codes themselves
# ---------------------------------------------------------------------------


def test_the_three_codes_are_the_conventional_ones():
    assert int(ExitCode.SUCCESS) == 0
    assert int(ExitCode.FAILURE) == 1
    assert int(ExitCode.USAGE) == 2


def test_only_success_counts_as_success():
    assert ExitCode.SUCCESS.succeeded
    assert not ExitCode.FAILURE.succeeded
    assert not ExitCode.USAGE.succeeded


def test_every_code_has_a_meaning():
    assert set(EXIT_CODE_MEANINGS) == set(ExitCode)
    for code in ExitCode:
        assert describe(int(code)) == EXIT_CODE_MEANINGS[code]


def test_describe_tolerates_an_unknown_status():
    assert describe(77) == "unrecognised exit status"


# ---------------------------------------------------------------------------
# fail()
# ---------------------------------------------------------------------------


def test_fail_raises_system_exit_with_the_given_code():
    printed = []
    with pytest.raises(SystemExit) as excinfo:
        fail("something broke", ExitCode.USAGE, printer=printed.append)
    assert excinfo.value.code == 2
    assert "something broke" in printed[0]


def test_fail_defaults_to_the_generic_failure():
    with pytest.raises(SystemExit) as excinfo:
        fail("broke", printer=lambda _: None)
    assert excinfo.value.code == 1


def test_fail_refuses_to_report_success_as_a_failure():
    with pytest.raises(ValueError):
        fail("fine", ExitCode.SUCCESS, printer=lambda _: None)


def test_fail_escapes_the_message():
    printed = []
    with pytest.raises(SystemExit):
        fail("cannot read src/[id]/page.tsx", printer=printed.append)
    # Rendering the line must not swallow the path.
    import io

    from rich.console import Console

    buffer = io.StringIO()
    Console(file=buffer, width=200).print(printed[0])
    assert "src/[id]/page.tsx" in buffer.getvalue()


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def test_no_source_is_a_usage_error():
    result = CliRunner().invoke(cli_main.main, ["run"])
    assert result.exit_code == 2
    assert "Provide either --url or --local" in result.output


def test_a_missing_local_repository_fails():
    result = CliRunner().invoke(cli_main.main, ["run", "--local", "/nope/not/here"])
    assert result.exit_code == 1
    assert "Failed to load repository" in result.output


def test_a_failed_clone_fails(monkeypatch, tmp_path):
    class Exploding:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, return_skip_info=False):
            raise RuntimeError("Failed to clone repository: Authentication failed")

    monkeypatch.setattr(cli_main, "RepoLoader", Exploding)

    result = CliRunner().invoke(cli_main.main, ["run", "--url", "https://x/y.git"])
    assert result.exit_code == 1
    assert "Authentication failed" in result.output


def test_an_unwritable_output_path_is_a_usage_error(tmp_path):
    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--output", str(tmp_path)]
    )
    assert result.exit_code == 2


def test_failing_api_key_setup_fails(monkeypatch, tmp_path):
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")

    def refuse(provider):
        raise RuntimeError("Invalid Groq API key.")

    monkeypatch.setattr(cli_main, "setup_api_keys", refuse)

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--force"]
    )
    assert result.exit_code == 1
    assert "Failed to configure API keys" in result.output


def test_declining_the_estimate_is_not_a_failure(tmp_path):
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path)], input="n\n"
    )
    assert result.exit_code == 0
    assert "Operation cancelled." in result.output


def test_declining_the_overwrite_is_not_a_failure(monkeypatch, tmp_path):
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
    destination = tmp_path / "OUT.md"
    destination.write_text("keep me\n", encoding="utf-8")

    monkeypatch.setattr(cli_main, "setup_api_keys", lambda provider: None)
    monkeypatch.setattr(
        cli_main, "generate_all_summaries", lambda **kwargs: ([{"description": "d"}], [])
    )
    monkeypatch.setattr(
        cli_main, "generate_hierarchical_summaries", lambda **kwargs: [{"description": "d"}]
    )
    monkeypatch.setattr(cli_main, "run_pipeline", lambda **kwargs: "# Title\n")

    result = CliRunner().invoke(
        cli_main.main,
        ["run", "--local", str(tmp_path), "--output", str(destination)],
        input="y\nn\n",
    )
    assert result.exit_code == 0
    assert destination.read_text(encoding="utf-8") == "keep me\n"


def test_a_successful_run_succeeds(monkeypatch, tmp_path):
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")

    monkeypatch.setattr(cli_main, "setup_api_keys", lambda provider: None)
    monkeypatch.setattr(
        cli_main, "generate_all_summaries", lambda **kwargs: ([{"description": "d"}], [])
    )
    monkeypatch.setattr(
        cli_main, "generate_hierarchical_summaries", lambda **kwargs: [{"description": "d"}]
    )
    monkeypatch.setattr(cli_main, "run_pipeline", lambda **kwargs: "# Title\n")

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--force"]
    )
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# The leaked clone directory
# ---------------------------------------------------------------------------


def _clone_dirs() -> set[str]:
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "repo2readme-*")))


def test_a_failed_clone_leaves_nothing_behind(monkeypatch):
    def refuse(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=128, cmd=["git", "clone"], stderr="fatal: Authentication failed"
        )

    monkeypatch.setattr(subprocess, "run", refuse)

    before = _clone_dirs()
    loader = UrlRepoLoader("https://example.invalid/x.git")

    with pytest.raises(RuntimeError, match="Failed to clone repository"):
        loader.load()

    assert _clone_dirs() == before
    assert loader.temp_dir is None


def test_a_missing_git_binary_says_so(monkeypatch):
    def refuse(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(subprocess, "run", refuse)

    before = _clone_dirs()
    loader = UrlRepoLoader("https://example.invalid/x.git")

    with pytest.raises(RuntimeError, match="git was not found on PATH"):
        loader.load()

    assert _clone_dirs() == before


def test_a_failing_traversal_also_cleans_up(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)

    import repo2readme.loaders.loader as loader_module

    def explode(*args, **kwargs):
        raise OSError("filesystem went away")

    monkeypatch.setattr(loader_module, "_run_traversal", explode)

    before = _clone_dirs()
    loader = UrlRepoLoader("https://example.invalid/x.git")

    with pytest.raises(OSError):
        loader.load()

    assert _clone_dirs() == before


def test_an_interrupted_clone_cleans_up(monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", interrupt)

    before = _clone_dirs()
    loader = UrlRepoLoader("https://example.invalid/x.git")

    with pytest.raises(KeyboardInterrupt):
        loader.load()

    assert _clone_dirs() == before


def test_a_destination_the_caller_chose_is_left_alone(monkeypatch, tmp_path):
    """Only a directory the loader made is the loader's to remove."""

    def refuse(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=128, cmd=["git", "clone"], stderr="fatal: nope"
        )

    monkeypatch.setattr(subprocess, "run", refuse)

    destination = tmp_path / "checkout"
    loader = UrlRepoLoader("https://example.invalid/x.git")
    loader.temp_dir = str(destination)

    with pytest.raises(RuntimeError):
        loader.load()

    assert destination.exists()
    assert loader.temp_dir == str(destination)


def test_the_cli_leaks_nothing_when_a_clone_fails(monkeypatch):
    def refuse(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=128, cmd=["git", "clone"], stderr="fatal: Authentication failed"
        )

    monkeypatch.setattr(subprocess, "run", refuse)

    before = _clone_dirs()
    result = CliRunner().invoke(
        cli_main.main, ["run", "--url", "https://example.invalid/x.git"]
    )

    assert result.exit_code == 1
    assert _clone_dirs() == before
