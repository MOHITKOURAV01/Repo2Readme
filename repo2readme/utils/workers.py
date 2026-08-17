"""One definition of what ``--max-workers`` means.

The flag was consumed in two places that disagreed. The traversal pipeline
clamped it to at least one and ignored the amount of work
(``max(1, self.max_workers)``), while the summarization stage capped it at the
document count and treated zero as "unset"
(``min(max_workers or 4, total_documents)``) - so ``0`` silently became four and
a negative value reached ``ThreadPoolExecutor``, which rejects it:

    ValueError: max_workers must be greater than 0

That crash landed part way through the summarization progress bar, after the
repository had been loaded and the token estimate confirmed. The value is now
validated in the CLI, before any of that, and resolved here for both stages.
"""

from __future__ import annotations

DEFAULT_MAX_WORKERS = 4

MIN_MAX_WORKERS = 1


def validate_max_workers(value: int | None) -> int | None:
    """Return ``value`` if it is a usable worker count.

    ``None`` means "decide for me" and is passed through.

    Raises
    ------
    ValueError
        If the value is below :data:`MIN_MAX_WORKERS`. Zero is rejected rather
        than silently read as the default: nobody who types ``0`` expects four
        threads.
    """
    if value is None:
        return None

    if value < MIN_MAX_WORKERS:
        raise ValueError(
            f"must be {MIN_MAX_WORKERS} or greater, got {value}"
        )

    return value


def resolve_worker_count(
    requested: int | None,
    total_items: int,
    default: int = DEFAULT_MAX_WORKERS,
) -> int:
    """How many worker threads to start for ``total_items`` pieces of work.

    Never more than there is work to do, never fewer than one - a pool of zero
    threads never finishes, and ``ThreadPoolExecutor`` refuses a non-positive
    size. With no request, ``default`` applies.
    """
    wanted = default if requested is None else requested
    wanted = max(MIN_MAX_WORKERS, wanted)

    if total_items <= 0:
        return MIN_MAX_WORKERS

    return min(wanted, total_items)
