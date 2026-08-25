"""How long a single request is allowed to take.

``create_llm`` built every chat model without a deadline, and none of the four
call sites supplied one, so the underlying HTTP client fell back to its own
default - which for several of these is "wait indefinitely".

Nothing in the project could rescue that. ``utils.retry`` classifies
``TimeoutError`` as retryable and matches "timeout" and "timed out" in provider
messages, but a request that never returns never raises, so none of it is
reachable. Summarization runs in a four-thread pool, so four stalled requests
end the run with the progress bar frozen at its last count; Ctrl-C is absorbed
by ``ThreadPoolExecutor.shutdown(wait=True)``, which cannot interrupt a thread
blocked in a socket read; and the cache flushes in the ``finally`` block of
``run()``, which is never reached, so the summaries already produced go with it.

The four provider clients spell the parameter four different ways
(``request_timeout``, ``default_request_timeout``, ``timeout``, and Ollama only
through ``client_kwargs``), which is why the name lives in the provider
registry and the value is translated in the factory.
"""

from __future__ import annotations

# Seconds. Generous enough that a slow model finishes a large file, short
# enough that a dead connection is noticed within a coffee break.
DEFAULT_TIMEOUT_SECONDS = 120.0

# The README stage writes a whole document from every summary in the
# repository, so it legitimately takes longer than describing one file. One
# flag scales rather than applying flat.
README_TIMEOUT_MULTIPLIER = 3.0

# What the user types to turn the deadline off, for anyone deliberately driving
# a slow local model.
NO_TIMEOUT = 0


class InvalidTimeoutError(ValueError):
    """The requested timeout cannot be used, with a reason the user can act on."""


def validate_timeout(value: float | int | None) -> float | None:
    """Return ``value`` as a usable timeout.

    ``None`` means "use the default" and is passed through. ``0`` means "no
    deadline" and becomes ``None`` at the client, which is what every one of
    these libraries reads as unlimited.

    Raises
    ------
    InvalidTimeoutError
        If the value is negative. Validated in the CLI, before the clone, so it
        cannot surface at the first request the way ``--max-workers`` used to.
    """
    if value is None:
        return None

    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidTimeoutError(f"must be a number of seconds, got {value!r}") from exc

    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        raise InvalidTimeoutError(f"must be a finite number of seconds, got {value!r}")

    if seconds < 0:
        raise InvalidTimeoutError(
            f"must be 0 or greater, got {value}. Use 0 for no timeout."
        )

    return seconds


def resolve_timeout(
    requested: float | None,
    multiplier: float = 1.0,
    default: float = DEFAULT_TIMEOUT_SECONDS,
) -> float | None:
    """The deadline for one request, or ``None`` for no deadline.

    ``requested`` is what the user asked for: ``None`` for the default, ``0``
    for unlimited. ``multiplier`` scales it for the stages that need longer.
    """
    seconds = default if requested is None else float(requested)

    if seconds <= 0:
        return None

    return seconds * multiplier


def readme_timeout(requested: float | None) -> float | None:
    """The deadline for the README and review stages."""
    return resolve_timeout(requested, multiplier=README_TIMEOUT_MULTIPLIER)
