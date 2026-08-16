"""Cloning a remote repository.

``UrlRepoLoader`` used to build the git command inline::

    subprocess.run([
        "git", "clone", "--branch", self.branch, "--depth", "1",
        self.clone_url, self.temp_dir,
    ], check=True, capture_output=True, text=True)

with ``self.branch`` defaulting to the literal ``"main"``. That default is
wrong for every repository that still uses ``master``, and for anything on
``develop`` or ``trunk``: the clone fails with "Remote branch main not found in
upstream origin" even though plain ``git clone`` would have worked, because git
follows the remote's ``HEAD`` when no branch is requested.

This module owns the clone. When no branch is requested it asks the remote what
its default branch is instead of guessing, and it turns git's stderr into a
message that says what to do about the failure - "repository not found" and
"authentication failed" need very different responses from the user, and both
used to arrive as the same wall of git output.
"""

from __future__ import annotations

import logging
import re
import subprocess
from enum import Enum

logger = logging.getLogger(__name__)

# A shallow clone is enough: nothing in the project reads history.
DEFAULT_CLONE_DEPTH = 1

# ``git ls-remote`` is a single round trip against the remote; the clone itself
# can legitimately take minutes on a large repository.
LS_REMOTE_TIMEOUT_SECONDS = 30
CLONE_TIMEOUT_SECONDS = 900

# ``git ls-remote --symref <url> HEAD`` answers with, for example:
#     ref: refs/heads/master\tHEAD
#     a1b2c3...\tHEAD
_SYMREF_RE = re.compile(r"^ref:\s+refs/heads/(?P<branch>\S+)\s+HEAD\s*$", re.MULTILINE)


class CloneFailure(str, Enum):
    """Why a clone failed, as far as it can be told from git's output."""

    AUTHENTICATION = "authentication"
    REPOSITORY_NOT_FOUND = "repository-not-found"
    BRANCH_NOT_FOUND = "branch-not-found"
    NETWORK = "network"
    GIT_MISSING = "git-missing"
    TIMEOUT = "timeout"
    DESTINATION = "destination"
    UNKNOWN = "unknown"


class CloneError(RuntimeError):
    """A clone that did not produce a working tree.

    Subclasses ``RuntimeError`` and keeps the historical
    ``"Failed to clone repository: ..."`` prefix, so existing callers and
    tests that only look at the message keep working. ``kind`` and ``stderr``
    are there for callers that want to react to the specific failure.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: CloneFailure = CloneFailure.UNKNOWN,
        stderr: str = "",
    ):
        super().__init__(message)
        self.kind = kind
        self.stderr = stderr


# Ordered most specific first: "could not read Username" is an authentication
# problem even though it also mentions the repository.
_FAILURE_SIGNATURES: tuple[tuple[CloneFailure, tuple[str, ...]], ...] = (
    (
        CloneFailure.BRANCH_NOT_FOUND,
        (
            "remote branch",
            "not found in upstream",
            "pathspec",
        ),
    ),
    (
        CloneFailure.AUTHENTICATION,
        (
            "authentication failed",
            "could not read username",
            "could not read password",
            "permission denied (publickey)",
            "invalid username or password",
            "access denied",
            "terminal prompts disabled",
            "support for password authentication was removed",
        ),
    ),
    (
        CloneFailure.REPOSITORY_NOT_FOUND,
        (
            "repository not found",
            "does not appear to be a git repository",
            "not found: ",
            "remote: not found",
        ),
    ),
    (
        CloneFailure.NETWORK,
        (
            "could not resolve host",
            "could not resolve hostname",
            "connection timed out",
            "connection refused",
            "network is unreachable",
            "failed to connect",
            "ssl certificate problem",
            "temporary failure in name resolution",
            "operation timed out",
        ),
    ),
    (
        CloneFailure.DESTINATION,
        (
            "already exists and is not an empty directory",
            "permission denied",
            "no space left on device",
            "read-only file system",
        ),
    ),
)


_REMEDIES: dict[CloneFailure, str] = {
    CloneFailure.BRANCH_NOT_FOUND: (
        "The branch does not exist on the remote. Run "
        "'git ls-remote --heads <url>' to list the branches it does have, or "
        "drop --branch to use the repository's default branch."
    ),
    CloneFailure.AUTHENTICATION: (
        "The remote refused the credentials. For a private repository over "
        "HTTPS use a personal access token, or clone over SSH with a key the "
        "remote knows about."
    ),
    CloneFailure.REPOSITORY_NOT_FOUND: (
        "The remote reports no such repository. Check the URL for a typo - a "
        "private repository also reports this when the credentials in use "
        "cannot see it."
    ),
    CloneFailure.NETWORK: (
        "The remote could not be reached. Check the network connection, the "
        "host name, and any proxy settings."
    ),
    CloneFailure.GIT_MISSING: (
        "git is not installed, or not on PATH. Install git, or point at an "
        "already-cloned copy with --local."
    ),
    CloneFailure.TIMEOUT: (
        "The clone did not finish in time. Very large repositories can need "
        "longer than the default timeout."
    ),
    CloneFailure.DESTINATION: (
        "The clone destination could not be written. Check the permissions "
        "and free space of the temporary directory."
    ),
}


def classify_failure(stderr: str) -> CloneFailure:
    """Map git's stderr onto a :class:`CloneFailure`."""
    haystack = (stderr or "").lower()
    if not haystack.strip():
        return CloneFailure.UNKNOWN

    for kind, needles in _FAILURE_SIGNATURES:
        if any(needle in haystack for needle in needles):
            return kind

    return CloneFailure.UNKNOWN


def describe_failure(
    kind: CloneFailure,
    clone_url: str,
    branch: str | None,
    stderr: str,
) -> str:
    """Build the message a user actually sees for a failed clone."""
    target = f"{clone_url} (branch {branch})" if branch else clone_url
    parts = [f"Failed to clone repository: {target}"]

    remedy = _REMEDIES.get(kind)
    if remedy:
        parts.append(remedy)

    detail = (stderr or "").strip()
    if detail:
        parts.append(f"git said: {detail}")

    return "\n".join(parts)


def _invoke(command: list[str], runner, timeout: int):
    """Run ``command``, translating the two failures git itself cannot report."""
    try:
        return runner(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise CloneError(
            f"Failed to clone repository: {_REMEDIES[CloneFailure.GIT_MISSING]}",
            kind=CloneFailure.GIT_MISSING,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise CloneError(
            f"Failed to clone repository: {_REMEDIES[CloneFailure.TIMEOUT]}",
            kind=CloneFailure.TIMEOUT,
        ) from exc


def parse_symref(output: str) -> str | None:
    """Read the branch name out of ``git ls-remote --symref`` output."""
    match = _SYMREF_RE.search(str(output or ""))
    if match is None:
        return None
    branch = match.group("branch").strip()
    return branch or None


def resolve_default_branch(clone_url: str, runner=None) -> str | None:
    """
    Ask the remote which branch its ``HEAD`` points at.

    Returns ``None`` when the remote cannot be asked - an old git without
    ``--symref``, a transport that does not report it, or any error at all.
    ``None`` is not a failure: the caller then clones without ``--branch``,
    which is exactly what a bare ``git clone`` does, and the real error (if
    there is one) is reported by the clone itself rather than by this probe.
    """
    runner = runner or subprocess.run

    try:
        result = _invoke(
            ["git", "ls-remote", "--symref", clone_url, "HEAD"],
            runner,
            LS_REMOTE_TIMEOUT_SECONDS,
        )
    except CloneError:
        raise
    except subprocess.CalledProcessError as exc:
        logger.debug(
            "Could not read the default branch of %s: %s",
            clone_url,
            (exc.stderr or "").strip(),
        )
        return None
    except Exception as exc:  # noqa: BLE001 - a probe must never be fatal
        logger.debug("Could not read the default branch of %s: %s", clone_url, exc)
        return None

    branch = parse_symref(getattr(result, "stdout", "") or "")
    if branch:
        logger.debug("Default branch of %s is %s", clone_url, branch)
    return branch


def build_clone_command(
    clone_url: str,
    destination: str,
    branch: str | None = None,
    depth: int | None = DEFAULT_CLONE_DEPTH,
) -> list[str]:
    """
    The git command for a clone.

    ``branch`` is omitted entirely when it is ``None``, which is what makes git
    follow the remote's own default branch. ``depth`` of ``None`` or ``0``
    means a full clone.
    """
    command = ["git", "clone"]

    if branch:
        command += ["--branch", branch]

    if depth:
        command += ["--depth", str(int(depth))]

    command += [clone_url, destination]
    return command


def clone_repository(
    clone_url: str,
    destination: str,
    branch: str | None = None,
    depth: int | None = DEFAULT_CLONE_DEPTH,
    runner=None,
    timeout: int = CLONE_TIMEOUT_SECONDS,
) -> str | None:
    """
    Clone ``clone_url`` into ``destination``.

    When ``branch`` is ``None`` the remote's default branch is resolved first,
    purely so that the branch that was used can be reported back; if the remote
    does not answer, the clone runs without ``--branch`` and git picks the
    default itself.

    Returns the branch that was cloned, or ``None`` when git chose it.

    Raises
    ------
    CloneError
        With a message that names the likely cause and what to do about it.
    """
    runner = runner or subprocess.run

    requested = branch
    if requested is None:
        requested = resolve_default_branch(clone_url, runner=runner)

    command = build_clone_command(clone_url, destination, requested, depth)

    try:
        _invoke(command, runner, timeout)
    except subprocess.CalledProcessError as exc:
        stderr = (getattr(exc, "stderr", "") or "")
        kind = classify_failure(stderr)
        raise CloneError(
            describe_failure(kind, clone_url, requested, stderr),
            kind=kind,
            stderr=stderr,
        ) from exc

    return requested
