"""Console and file logging setup for the CLI.

Three modules log through ``logging`` (``summarize.summary``,
``summarize.directory_summary`` and ``cache``) but nothing ever configured a
handler. Python fell back to ``logging.lastResort``, which writes the bare
message to stderr while ``rich.progress.Progress`` is redrawing - so warnings
shredded the progress bar, carried no context, and ``debug``/``info`` calls
were unreachable.

Logging goes through Rich here so it cooperates with the progress display, and
the verbosity flags are wired as eager Click callbacks: they configure logging
as a side effect and never reach the command function.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable

import click
from rich.console import Console
from rich.logging import RichHandler

# Handlers we installed are tagged so repeated configuration replaces them
# instead of stacking - CliRunner invokes the CLI many times per test session.
_HANDLER_TAG = "_repo2readme_handler"

_CONTEXT_KEY = "repo2readme.logging"

# Third-party loggers that are chatty at INFO/DEBUG. Damped unless the user
# explicitly asks for -vv.
NOISY_LOGGERS: tuple[str, ...] = (
    "langchain",
    "langchain_core",
    "langchain_community",
    "httpx",
    "httpcore",
    "urllib3",
    "openai",
    "groq",
    "google",
    "google_genai",
)

FILE_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def resolve_level(verbosity: int = 0, quiet: bool = False) -> int:
    """Map ``-v`` count and ``--quiet`` onto a logging level."""
    if quiet:
        return logging.ERROR
    if verbosity <= 0:
        return logging.WARNING
    if verbosity == 1:
        return logging.INFO
    return logging.DEBUG


def reset_logging(root: logging.Logger | None = None) -> None:
    """Remove handlers this module installed. Foreign handlers are left alone."""
    root = root or logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)
            # Closing must never take a run down.
            with contextlib.suppress(Exception):
                handler.close()


def _damp_noisy_loggers(verbosity: int, names: Iterable[str] = NOISY_LOGGERS) -> None:
    """Keep third-party libraries quiet unless the user asked for -vv."""
    level = logging.DEBUG if verbosity >= 2 else logging.WARNING
    for name in names:
        logging.getLogger(name).setLevel(level)


def configure_logging(
    verbosity: int = 0,
    quiet: bool = False,
    log_file: str | None = None,
    console: Console | None = None,
) -> int:
    """Configure console (and optionally file) logging. Returns the console level.

    Safe to call repeatedly: previously installed handlers are replaced, not
    duplicated. A file log always records DEBUG regardless of console
    verbosity, which is what makes it useful in a bug report.
    """
    level = resolve_level(verbosity, quiet)
    root = logging.getLogger()

    reset_logging(root)

    console_handler = RichHandler(
        console=console or Console(stderr=True),
        show_path=False,
        show_time=level <= logging.DEBUG,
        rich_tracebacks=True,
        markup=False,
    )
    console_handler.setLevel(level)
    setattr(console_handler, _HANDLER_TAG, True)
    root.addHandler(console_handler)

    root_level = level
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
        except OSError as error:
            # A bad --log-file path must not take the whole run down.
            click.echo(f"Could not open log file {log_file!r}: {error}", err=True)
        else:
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT))
            setattr(file_handler, _HANDLER_TAG, True)
            root.addHandler(file_handler)
            root_level = logging.DEBUG

    root.setLevel(root_level)
    _damp_noisy_loggers(verbosity)

    return level


def _apply_from_context(ctx: click.Context) -> None:
    """Re-configure logging from whatever flags have been parsed so far."""
    settings = ctx.meta.setdefault(_CONTEXT_KEY, {})

    verbosity = settings.get("verbosity", 0)
    quiet = settings.get("quiet", False)

    if quiet and verbosity:
        raise click.UsageError("--quiet and --verbose cannot be used together.")

    configure_logging(
        verbosity=verbosity,
        quiet=quiet,
        log_file=settings.get("log_file"),
    )


def _store(ctx: click.Context, key: str, value) -> None:
    ctx.meta.setdefault(_CONTEXT_KEY, {})[key] = value
    _apply_from_context(ctx)


def _verbose_callback(ctx, param, value):
    _store(ctx, "verbosity", value or 0)


def _quiet_callback(ctx, param, value):
    _store(ctx, "quiet", bool(value))


def _log_file_callback(ctx, param, value):
    _store(ctx, "log_file", value)


def logging_options(func):
    """Add ``-v``/``--quiet``/``--log-file`` to a command.

    The options configure logging through their callbacks and are not passed to
    the command function, so adding them does not change its signature.
    """
    func = click.option(
        "--log-file",
        default=None,
        type=click.Path(dir_okay=False, writable=True),
        expose_value=False,
        callback=_log_file_callback,
        help="Write a full debug log to this file, regardless of console verbosity.",
    )(func)
    func = click.option(
        "--quiet",
        "-q",
        is_flag=True,
        default=False,
        expose_value=False,
        callback=_quiet_callback,
        help="Only report errors.",
    )(func)
    func = click.option(
        "--verbose",
        "-v",
        count=True,
        expose_value=False,
        callback=_verbose_callback,
        help="Increase log verbosity (-v for info, -vv for debug).",
    )(func)
    return func
