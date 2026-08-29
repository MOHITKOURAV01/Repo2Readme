"""Retry helper for transient LLM failures.

Every LLM call in the project was a single attempt, so one 429 during a
parallel run permanently lost that file's summary. This module adds a small,
dependency-free retry with exponential backoff and jitter.

The important part is the classification: rate limits, timeouts and transient
server errors are worth retrying, while a bad API key or an unsupported
provider is not - retrying those only makes the failure slower.

A response the output parser could not read is also worth retrying. The chains
sample the model, so the next attempt produces different text, and a truncated
object or a stray code fence usually parses on the second ask.

When the provider says how long to wait, that answer is used as given. It is a
fact about when the next request can succeed, which is not something a backoff
curve can work out - so it is bounded by its own limit, ``max_retry_after``,
rather than by the ceiling on the delays this module invents. A hint longer
than that limit is not a reason to retry sooner; it means no further attempt
will land inside the window, and the failure is raised immediately instead of
spending the remaining attempts on requests that cannot succeed.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 2
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0
DEFAULT_JITTER = 0.25

# Longest wait accepted from a provider's own hint. Deliberately larger than
# DEFAULT_MAX_DELAY: that bounds a delay this module guessed at, while a hint
# is the provider stating when the next request can succeed. Five minutes
# covers the per-minute token limits that large repositories hit, and stops
# short of the hour-scale waits a daily quota reports.
DEFAULT_MAX_RETRY_AFTER = 300.0

ENV_MAX_RETRIES = "REPO2README_MAX_RETRIES"
ENV_BASE_DELAY = "REPO2README_RETRY_BASE_DELAY"
ENV_MAX_RETRY_AFTER = "REPO2README_MAX_RETRY_AFTER"

# Statuses worth another attempt: rate limiting, request timeout and the
# transient 5xx family.
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Statuses that will never succeed on a retry with the same input.
NON_RETRYABLE_STATUS_CODES = frozenset({400, 401, 403, 404, 413, 422})

RETRYABLE_MESSAGE_PATTERNS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "connection error",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "overloaded",
    "try again",
)

NON_RETRYABLE_MESSAGE_PATTERNS = (
    "invalid api key",
    "incorrect api key",
    "api key not found",
    "authentication",
    "unauthorized",
    "permission denied",
    "unsupported provider",
    "context length",
    "maximum context",
    "model not found",
    "does not exist",
)

# Exception classes raised when the model's answer could not be parsed. Matched
# against every class name in the exception's MRO rather than just its own name:
# the provider integrations subclass these with names of their own, and
# LangChain's OutputParserException is itself a ValueError, which the
# programming-error rejection below would otherwise swallow.
PARSER_EXCEPTION_MARKERS = (
    "outputparserexception",
    "outputparsererror",
    "parsererror",
    "parseexception",
    "validationerror",
    "jsondecodeerror",
)

_STATUS_IN_MESSAGE = re.compile(r"\berror code:\s*(\d{3})\b", re.IGNORECASE)

# Where a message hint starts. What follows it is a duration, parsed below.
_RETRY_AFTER_PREFIX = re.compile(r"try again in\s+", re.IGNORECASE)

# One value-and-unit pair. Providers write these back to back and without a
# separator - Groq reports a per-minute limit as "2m59.56s" - so a hint is any
# number of these in a row, and the parser reads them one at a time.
#
# The alternation is ordered longest-first within each unit, because Python
# takes the first branch that matches: with "m" ahead of "minutes", "minutes"
# would parse as one minute followed by the unreadable "inutes". The trailing
# lookahead is what a `\b` cannot do here - it rejects a unit that is really
# the start of a word, while still allowing the digit that begins the next pair.
_DURATION_PART = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(milliseconds?|msecs?|ms"
    r"|seconds?|secs?|s"
    r"|minutes?|mins?|m"
    r"|hours?|hrs?|h)"
    r"(?![a-z])",
    re.IGNORECASE,
)

_UNIT_SECONDS = {
    "ms": 0.001,
    "msec": 0.001,
    "msecs": 0.001,
    "millisecond": 0.001,
    "milliseconds": 0.001,
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}


@dataclass(frozen=True)
class RetryConfig:
    """How hard to try. ``max_retries=0`` restores single-attempt behaviour."""

    max_retries: int = DEFAULT_MAX_RETRIES
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = DEFAULT_MAX_DELAY
    jitter: float = DEFAULT_JITTER
    max_retry_after: float = DEFAULT_MAX_RETRY_AFTER

    @property
    def max_attempts(self) -> int:
        return self.max_retries + 1

    @classmethod
    def from_env(cls, env: dict | None = None) -> RetryConfig:
        """Build a config from the environment, ignoring unusable values."""
        source = os.environ if env is None else env

        return cls(
            max_retries=_read_int(
                source, ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES, minimum=0
            ),
            base_delay=_read_float(
                source, ENV_BASE_DELAY, DEFAULT_BASE_DELAY, minimum=0.0
            ),
            max_retry_after=_read_float(
                source, ENV_MAX_RETRY_AFTER, DEFAULT_MAX_RETRY_AFTER, minimum=0.0
            ),
        )


def _read_int(source, name: str, default: int, minimum: int) -> int:
    raw = source.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        logger.warning("Ignoring %s=%r: not an integer", name, raw)
        return default
    if value < minimum:
        logger.warning("Ignoring %s=%r: must be >= %d", name, raw, minimum)
        return default
    return value


def _read_float(source, name: str, default: float, minimum: float) -> float:
    raw = source.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).strip())
    except ValueError:
        logger.warning("Ignoring %s=%r: not a number", name, raw)
        return default
    if value < minimum:
        logger.warning("Ignoring %s=%r: must be >= %s", name, raw, minimum)
        return default
    return value


def status_code_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction across the provider SDKs."""
    for attribute in ("status_code", "http_status", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value

    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value

    match = _STATUS_IN_MESSAGE.search(str(exc))
    if match:
        return int(match.group(1))

    return None


def exception_class_names(exc: BaseException) -> tuple[str, ...]:
    """Lowercased names of ``exc``'s class and every class it inherits from."""
    return tuple(cls.__name__.lower() for cls in type(exc).__mro__)


def is_parse_error(exc: BaseException) -> bool:
    """Whether ``exc`` means the model's answer could not be parsed.

    Covers the parser exceptions raised by the JSON and Pydantic parsers on the
    summarization and review chains, including provider-specific subclasses.
    """
    names = exception_class_names(exc)
    return any(
        marker in name for name in names for marker in PARSER_EXCEPTION_MARKERS
    )


def is_retryable(exc: BaseException) -> bool:
    """Whether another attempt could plausibly succeed."""
    # Checked before the programming-error rejection below, because
    # OutputParserException subclasses ValueError and would never reach a later
    # branch. The message is not consulted either: it embeds the model's own
    # output, so scanning it for phrases like "authentication" reads the answer
    # as if it were an error report.
    if is_parse_error(exc):
        return True

    # Programming and configuration errors: never worth a retry.
    if isinstance(exc, (TypeError, KeyError, AttributeError, ImportError,
                        NotImplementedError, ValueError)):
        return False

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    status = status_code_of(exc)
    if status is not None:
        if status in NON_RETRYABLE_STATUS_CODES:
            return False
        if status in RETRYABLE_STATUS_CODES:
            return True

    message = str(exc).lower()

    if any(pattern in message for pattern in NON_RETRYABLE_MESSAGE_PATTERNS):
        return False

    return any(pattern in message for pattern in RETRYABLE_MESSAGE_PATTERNS)


def parse_duration(text: str) -> float | None:
    """Seconds described by a run of value-and-unit pairs at the start of ``text``.

    ``"20s"``, ``"500ms"`` and ``"2m"`` are one pair. ``"2m59.56s"``,
    ``"1m20s"`` and ``"1h30m"`` are two, written without a separator, and are
    the forms providers use for the waits worth honouring - the shorter a limit
    is, the more likely it is to be reported in a single unit.

    Parsing stops at the first thing that is not a pair, so the trailing prose
    of a real error message ("Please try again in 5 minutes or reduce your
    request rate") does not prevent the part before it from being read.

    Returns ``None`` when ``text`` does not begin with a pair at all, which is
    what separates "no hint" from a hint of zero.
    """
    total = 0.0
    position = 0
    found = False

    while True:
        match = _DURATION_PART.match(text, position)
        if match is None:
            break
        total += float(match.group(1)) * _UNIT_SECONDS[match.group(2).lower()]
        position = match.end()
        found = True
        # Pairs are usually adjacent ("2m59.56s") but may be spaced ("2m 30s").
        while position < len(text) and text[position].isspace():
            position += 1

    return total if found else None


def _seconds_until(http_date: str) -> float | None:
    """Seconds from now until an HTTP-date, or ``None`` if it is not one.

    RFC 7231 lets ``Retry-After`` carry either a delay in seconds or a date,
    and both spellings are in use. A date already in the past means the wait is
    over, which is zero rather than a negative delay.
    """
    try:
        deadline = parsedate_to_datetime(http_date)
    except (TypeError, ValueError):
        return None
    if deadline is None:
        return None

    # A date with no zone is UTC by convention; parsedate_to_datetime leaves it
    # naive, and subtracting a naive from an aware datetime raises.
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)

    return max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())


def _header_retry_after(exc: BaseException) -> float | None:
    """The ``Retry-After`` header's value in seconds, if the response has one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    for key in ("retry-after", "Retry-After"):
        try:
            raw = headers.get(key)
        except AttributeError:
            continue
        if raw is None:
            continue

        text = str(raw).strip()
        try:
            value = float(text)
        except ValueError:
            seconds = _seconds_until(text)
            if seconds is not None:
                return seconds
            continue
        if value >= 0:
            return value

    return None


def retry_after_seconds(exc: BaseException) -> float | None:
    """Read the provider's own backoff hint, if it gave one.

    The header is preferred over the message: it is structured, and it is what
    the provider's own client would read.
    """
    header = _header_retry_after(exc)
    if header is not None:
        return header

    message = str(exc)
    prefix = _RETRY_AFTER_PREFIX.search(message)
    if prefix is None:
        return None

    return parse_duration(message[prefix.end():])


def compute_delay(
    attempt: int,
    config: RetryConfig,
    retry_after: float | None = None,
    rng: Callable[[], float] = random.random,
) -> float:
    """Delay before the next attempt (``attempt`` is 0-based).

    A provider-supplied ``retry_after`` wins over the computed backoff, because
    guessing shorter than the server asked for just earns another 429. It is
    bounded by ``max_retry_after`` and not by ``max_delay``: the latter caps a
    delay this module chose, and applying it to the provider's answer produced
    exactly the early retry the hint exists to prevent.

    A hint beyond that bound is still clamped here, but callers are expected to
    stop instead - see :func:`call_with_retry`.
    """
    if retry_after is not None:
        return min(max(retry_after, 0.0), config.max_retry_after)

    delay = min(config.base_delay * (2 ** attempt), config.max_delay)
    if config.jitter:
        delay += delay * config.jitter * rng()
    return min(delay, config.max_delay)


def call_with_retry(
    func: Callable[[], Any],
    *,
    config: RetryConfig | None = None,
    description: str = "LLM call",
    retryable: Callable[[BaseException], bool] = is_retryable,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> Any:
    """Call ``func`` retrying transient failures.

    The last exception is re-raised once the attempts are exhausted, so callers
    keep their existing error handling.
    """
    config = config or RetryConfig.from_env()
    attempts = max(1, config.max_attempts)
    last_exc: BaseException | None = None

    for attempt in range(attempts):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - re-raised below
            last_exc = exc
            is_last_attempt = attempt == attempts - 1

            if is_last_attempt or not retryable(exc):
                raise

            retry_after = retry_after_seconds(exc)

            # The provider has said the next request cannot succeed for longer
            # than this run is willing to wait. Retrying earlier fails by
            # definition, and spends an attempt doing it, so stop now and let
            # the caller report the provider's own reason.
            if retry_after is not None and retry_after > config.max_retry_after:
                logger.debug(
                    "%s failed (attempt %d/%d): %s. Provider asked for %.0fs, "
                    "which is beyond the %.0fs limit; not retrying",
                    description,
                    attempt + 1,
                    attempts,
                    exc,
                    retry_after,
                    config.max_retry_after,
                )
                raise

            delay = compute_delay(
                attempt, config, retry_after=retry_after, rng=rng
            )
            logger.debug(
                "%s failed (attempt %d/%d): %s. Retrying in %.2fs",
                description,
                attempt + 1,
                attempts,
                exc,
                delay,
            )
            sleep(delay)

    # Only reachable if attempts is 0, which max(1, ...) prevents.
    raise last_exc  # pragma: no cover
