"""Tests for cloning a remote repository.

The behaviour under test is the one described in the issue: ``--branch``
defaulted to the literal ``"main"`` and was always passed to ``git clone``, so
any repository whose default branch is ``master`` (or ``develop``, or
``trunk``) failed with "Remote branch main not found in upstream origin", and
the temporary clone directory created before the failed clone was never
removed.
"""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from repo2readme.loaders.clone import (
    CLONE_TIMEOUT_SECONDS,
    DEFAULT_CLONE_DEPTH,
    CloneError,
    CloneFailure,
    build_clone_command,
    classify_failure,
    clone_repository,
    describe_failure,
    parse_symref,
    resolve_default_branch,
)
from repo2readme.loaders.loader import UrlRepoLoader
from repo2readme.loaders.repo_loader import RepoLoader

REPO_URL = "https://github.com/acme/app.git"


def completed(stdout="", stderr=""):
    """A stand-in for a successful ``subprocess.run`` result."""
    result = MagicMock()
    result.returncode = 0
    result.stdout = stdout
    result.stderr = stderr
    return result


def failed(stderr, returncode=128):
    return subprocess.CalledProcessError(
        returncode=returncode, cmd=["git", "clone"], stderr=stderr
    )


class RecordingRunner:
    """A ``subprocess.run`` replacement that records the commands it is given.

    ``responses`` maps the git subcommand (``ls-remote``, ``clone``) to either
    a result or an exception to raise.
    """

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        self.kwargs = kwargs
        subcommand = command[1] if len(command) > 1 else ""
        response = self.responses.get(subcommand, completed())
        if isinstance(response, Exception):
            raise response
        return response

    def command_for(self, subcommand):
        for command in self.commands:
            if len(command) > 1 and command[1] == subcommand:
                return command
        return None


# ---------------------------------------------------------------------------
# parse_symref
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output, expected",
    [
        ("ref: refs/heads/master\tHEAD\ndeadbeef\tHEAD\n", "master"),
        ("ref: refs/heads/main\tHEAD\n", "main"),
        ("ref: refs/heads/develop\tHEAD\n", "develop"),
        ("ref: refs/heads/release/2.0\tHEAD\n", "release/2.0"),
        ("ref: refs/heads/feature-x\tHEAD\nabc123\tHEAD\n", "feature-x"),
        ("deadbeef\tHEAD\n", None),
        ("", None),
        (None, None),
        ("ref: refs/tags/v1\tHEAD\n", None),
        ("total nonsense", None),
    ],
)
def test_parse_symref(output, expected):
    assert parse_symref(output) == expected


def test_parse_symref_ignores_other_refs():
    output = (
        "ref: refs/heads/trunk\tHEAD\n"
        "1111111111111111111111111111111111111111\tHEAD\n"
        "2222222222222222222222222222222222222222\trefs/heads/main\n"
    )
    assert parse_symref(output) == "trunk"


# ---------------------------------------------------------------------------
# resolve_default_branch
# ---------------------------------------------------------------------------


def test_resolve_default_branch_reads_the_remote_head():
    runner = RecordingRunner(
        {"ls-remote": completed("ref: refs/heads/master\tHEAD\n")}
    )

    assert resolve_default_branch(REPO_URL, runner=runner) == "master"
    assert runner.command_for("ls-remote") == [
        "git",
        "ls-remote",
        "--symref",
        REPO_URL,
        "HEAD",
    ]


def test_resolve_default_branch_returns_none_when_the_remote_fails():
    runner = RecordingRunner({"ls-remote": failed("fatal: repository not found")})

    # A probe must not turn into the error the user sees; the clone reports it.
    assert resolve_default_branch(REPO_URL, runner=runner) is None


def test_resolve_default_branch_returns_none_for_unparsable_output():
    runner = RecordingRunner({"ls-remote": completed("no symref here")})

    assert resolve_default_branch(REPO_URL, runner=runner) is None


def test_resolve_default_branch_survives_an_unexpected_runner_error():
    def explode(command, **kwargs):
        raise OSError("something odd")

    assert resolve_default_branch(REPO_URL, runner=explode) is None


def test_resolve_default_branch_reports_missing_git():
    def no_git(command, **kwargs):
        raise FileNotFoundError("git")

    with pytest.raises(CloneError) as exc_info:
        resolve_default_branch(REPO_URL, runner=no_git)

    assert exc_info.value.kind is CloneFailure.GIT_MISSING


# ---------------------------------------------------------------------------
# build_clone_command
# ---------------------------------------------------------------------------


def test_build_clone_command_omits_branch_when_none():
    command = build_clone_command(REPO_URL, "/tmp/dest", branch=None)

    assert "--branch" not in command
    assert command == ["git", "clone", "--depth", "1", REPO_URL, "/tmp/dest"]


def test_build_clone_command_includes_an_explicit_branch():
    command = build_clone_command(REPO_URL, "/tmp/dest", branch="develop")

    assert command[command.index("--branch") + 1] == "develop"


@pytest.mark.parametrize("depth", [None, 0])
def test_build_clone_command_can_do_a_full_clone(depth):
    command = build_clone_command(REPO_URL, "/tmp/dest", depth=depth)

    assert "--depth" not in command


def test_build_clone_command_honours_a_custom_depth():
    command = build_clone_command(REPO_URL, "/tmp/dest", depth=25)

    assert command[command.index("--depth") + 1] == "25"


# ---------------------------------------------------------------------------
# clone_repository
# ---------------------------------------------------------------------------


def test_clone_without_a_branch_uses_the_remote_default():
    runner = RecordingRunner(
        {"ls-remote": completed("ref: refs/heads/master\tHEAD\n")}
    )

    used = clone_repository(REPO_URL, "/tmp/dest", runner=runner)

    assert used == "master"
    clone = runner.command_for("clone")
    assert clone[clone.index("--branch") + 1] == "master"


def test_clone_falls_back_to_git_s_own_default_when_the_remote_is_silent():
    runner = RecordingRunner({"ls-remote": completed("")})

    used = clone_repository(REPO_URL, "/tmp/dest", runner=runner)

    assert used is None
    assert "--branch" not in runner.command_for("clone")


def test_an_explicit_branch_skips_the_probe():
    runner = RecordingRunner()

    used = clone_repository(REPO_URL, "/tmp/dest", branch="develop", runner=runner)

    assert used == "develop"
    assert runner.command_for("ls-remote") is None
    clone = runner.command_for("clone")
    assert clone[clone.index("--branch") + 1] == "develop"


def test_clone_passes_a_timeout():
    runner = RecordingRunner()

    clone_repository(REPO_URL, "/tmp/dest", branch="main", runner=runner)

    assert runner.kwargs["timeout"] == CLONE_TIMEOUT_SECONDS
    assert runner.kwargs["check"] is True
    assert runner.kwargs["capture_output"] is True


def test_clone_timeout_becomes_a_clone_error():
    def slow(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd=command, timeout=1)

    with pytest.raises(CloneError) as exc_info:
        clone_repository(REPO_URL, "/tmp/dest", branch="main", runner=slow)

    assert exc_info.value.kind is CloneFailure.TIMEOUT


def test_clone_uses_the_default_depth():
    runner = RecordingRunner()

    clone_repository(REPO_URL, "/tmp/dest", branch="main", runner=runner)

    clone = runner.command_for("clone")
    assert clone[clone.index("--depth") + 1] == str(DEFAULT_CLONE_DEPTH)


# ---------------------------------------------------------------------------
# Failure classification and messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stderr, expected",
    [
        (
            "fatal: Remote branch main not found in upstream origin",
            CloneFailure.BRANCH_NOT_FOUND,
        ),
        (
            "error: pathspec 'release' did not match any file(s) known to git",
            CloneFailure.BRANCH_NOT_FOUND,
        ),
        ("remote: Repository not found.", CloneFailure.REPOSITORY_NOT_FOUND),
        (
            "fatal: 'x' does not appear to be a git repository",
            CloneFailure.REPOSITORY_NOT_FOUND,
        ),
        ("fatal: Authentication failed for 'https://...'", CloneFailure.AUTHENTICATION),
        ("git@github.com: Permission denied (publickey).", CloneFailure.AUTHENTICATION),
        (
            "fatal: could not read Username for 'https://github.com'",
            CloneFailure.AUTHENTICATION,
        ),
        ("fatal: unable to access '...': Could not resolve host: github.com", CloneFailure.NETWORK),
        ("ssh: connect to host github.com port 22: Connection refused", CloneFailure.NETWORK),
        ("fatal: destination path 'x' already exists and is not an empty directory.", CloneFailure.DESTINATION),
        ("fatal: write error: No space left on device", CloneFailure.DESTINATION),
        ("something nobody has seen before", CloneFailure.UNKNOWN),
        ("", CloneFailure.UNKNOWN),
    ],
)
def test_classify_failure(stderr, expected):
    assert classify_failure(stderr) is expected


def test_describe_failure_names_the_repository_and_the_branch():
    message = describe_failure(
        CloneFailure.BRANCH_NOT_FOUND,
        REPO_URL,
        "main",
        "fatal: Remote branch main not found in upstream origin",
    )

    assert "Failed to clone repository" in message
    assert REPO_URL in message
    assert "branch main" in message
    assert "--branch" in message
    assert "git said:" in message


def test_describe_failure_without_a_branch():
    message = describe_failure(CloneFailure.NETWORK, REPO_URL, None, "boom")

    assert "branch" not in message.split("git said:")[0].lower()
    assert "network" in message.lower()


def test_clone_error_keeps_the_historical_prefix():
    runner = RecordingRunner(
        {"clone": failed("fatal: Authentication failed for 'https://...'")}
    )

    with pytest.raises(CloneError) as exc_info:
        clone_repository(REPO_URL, "/tmp/dest", branch="main", runner=runner)

    error = exc_info.value
    assert str(error).startswith("Failed to clone repository")
    assert error.kind is CloneFailure.AUTHENTICATION
    assert "personal access token" in str(error)
    # Still a RuntimeError, so callers that only catch RuntimeError keep working.
    assert isinstance(error, RuntimeError)


def test_missing_git_is_reported_as_such():
    def no_git(command, **kwargs):
        raise FileNotFoundError("git")

    with pytest.raises(CloneError) as exc_info:
        clone_repository(REPO_URL, "/tmp/dest", branch="main", runner=no_git)

    assert exc_info.value.kind is CloneFailure.GIT_MISSING
    assert "not installed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# UrlRepoLoader
# ---------------------------------------------------------------------------


def test_loader_defaults_to_no_branch():
    assert UrlRepoLoader(REPO_URL).branch is None
    assert RepoLoader(REPO_URL).branch is None


@patch("repo2readme.loaders.loader.subprocess.run")
def test_loader_does_not_force_main(mock_run, tmp_path):
    mock_run.return_value = completed("ref: refs/heads/master\tHEAD\n")

    loader = UrlRepoLoader(REPO_URL)
    loader.temp_dir = str(tmp_path / "clone")

    try:
        loader.load()
    finally:
        loader.cleanup()

    commands = [call[0][0] for call in mock_run.call_args_list]
    clone = next(c for c in commands if c[1] == "clone")
    assert clone[clone.index("--branch") + 1] == "master"
    assert loader.cloned_branch == "master"


@patch("repo2readme.loaders.loader.subprocess.run")
def test_loader_still_honours_an_explicit_branch(mock_run, tmp_path):
    mock_run.return_value = completed()

    loader = UrlRepoLoader(REPO_URL, branch="develop")
    loader.temp_dir = str(tmp_path / "clone")

    try:
        loader.load()
    finally:
        loader.cleanup()

    commands = [call[0][0] for call in mock_run.call_args_list]
    assert not any(c[1] == "ls-remote" for c in commands)
    clone = commands[0]
    assert clone[clone.index("--branch") + 1] == "develop"


@patch("repo2readme.loaders.loader.subprocess.run")
def test_failed_clone_removes_the_directory_it_created(mock_run):
    mock_run.side_effect = failed("fatal: Repository not found")

    loader = UrlRepoLoader(REPO_URL, branch="main")

    with pytest.raises(CloneError):
        loader.load()

    # The directory is created before git runs, so it has to be cleaned up on
    # the error path: the CLI returns without ever calling cleanup().
    assert loader.temp_dir is None


@patch("repo2readme.loaders.loader.subprocess.run")
def test_failed_clone_leaves_a_caller_supplied_directory_alone(mock_run, tmp_path):
    mock_run.side_effect = failed("fatal: Repository not found")

    destination = tmp_path / "mine"
    destination.mkdir()
    loader = UrlRepoLoader(REPO_URL, branch="main")
    loader.temp_dir = str(destination)

    with pytest.raises(CloneError):
        loader.load()

    assert os.path.isdir(destination)


@patch("repo2readme.loaders.loader.subprocess.run")
def test_successful_clone_keeps_the_directory(mock_run, tmp_path):
    mock_run.return_value = completed()

    loader = UrlRepoLoader(REPO_URL, branch="main")
    loader.temp_dir = str(tmp_path / "clone")

    try:
        loader.load()
        assert os.path.isdir(loader.temp_dir)
    finally:
        loader.cleanup()


@patch("repo2readme.loaders.loader.subprocess.run")
def test_repo_loader_passes_the_branch_through(mock_run, tmp_path):
    mock_run.return_value = completed()

    loader = RepoLoader(REPO_URL, branch="release/1.0")
    built = loader._build_loader()

    assert built.branch == "release/1.0"
