"""What ``repo2readme`` returns to the shell, in one place.

Every early exit from ``run()`` used to be a bare ``return``, which ends the
process with status ``0``. A missing ``--url``/``--local``, a repository that
could not be found, a clone that failed authentication - all of them printed a
red message and told the shell the run had succeeded::

    $ repo2readme run --url https://github.com/acme/private.git
    Failed to load repository: ... Authentication failed
    $ echo $?
    0

which is invisible to anything that checks the status, and a CI job that
regenerates a README would commit an unchanged file and stay green.

The failures that *did* exit non-zero used bare literals (``SystemExit(1)``,
``SystemExit(2)``) spread through ``main.py``, so there was nothing to read to
find out what a status meant. The codes are named here instead, and
:func:`fail` is the one way to end a run badly.

The convention is the usual one for a command line tool:

======  ============================================================
Status  Meaning
======  ============================================================
``0``   The run did what was asked. Declining a prompt counts: the
        user was asked and said no, which is not a failure.
``1``   The run was understood and could not be completed - the
        repository would not load, the keys would not configure, the
        README could not be generated or written.
``2``   The invocation itself was wrong - no source given, an
        ``--output`` path that cannot be written to. Nothing was
        attempted.
======  ============================================================
"""

from __future__ import annotations

from enum import IntEnum
from typing import Callable, NoReturn

from rich.markup import escape


class ExitCode(IntEnum):
    """Statuses ``repo2readme`` can end with."""

    #: The run finished, or the user declined a prompt.
    SUCCESS = 0

    #: The run was understood but could not be completed.
    FAILURE = 1

    #: The command line itself was wrong; nothing was attempted.
    USAGE = 2

    @property
    def succeeded(self) -> bool:
        return self is ExitCode.SUCCESS

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return str(int(self))


#: Human readable names, for help text and for the docs table.
EXIT_CODE_MEANINGS: dict[ExitCode, str] = {
    ExitCode.SUCCESS: "README generated, or the user declined a prompt",
    ExitCode.FAILURE: "the run could not be completed",
    ExitCode.USAGE: "the command line was wrong; nothing was attempted",
}


def describe(code: int) -> str:
    """One line explaining a status, for a message or a test failure."""
    try:
        return EXIT_CODE_MEANINGS[ExitCode(code)]
    except ValueError:
        return "unrecognised exit status"


def fail(
    message: str,
    code: ExitCode = ExitCode.FAILURE,
    printer: Callable[[str], None] | None = None,
) -> NoReturn:
    """Report ``message`` and end the run with ``code``.

    ``printer`` defaults to ``rich.print`` and is injectable so the helper can
    be exercised without capturing a console. The message is styled here rather
    than at the call site so every failure looks the same, and it is escaped
    here too: these messages quote paths and provider errors, and a stray
    ``[...]`` in one of those is a value, not a style.

    Raises
    ------
    SystemExit
        Always. The return type is ``NoReturn`` so a caller writing
        ``fail(...)`` instead of ``return fail(...)`` still type-checks.
    """
    if code is ExitCode.SUCCESS:
        raise ValueError("fail() is for failures; SUCCESS is not one.")

    if printer is None:  # pragma: no cover - trivial default
        from rich import print as printer  # type: ignore[assignment]

    printer(f"[red]{escape(str(message))}[/red]")
    raise SystemExit(int(code))
