"""Fit a file into one request instead of losing it.

``--max-file-size-kb`` was the only thing standing between a large file and the
model, and all it decided was whether the file was read at all. Anything under
the limit went whole::

    chain.invoke({"file_path": ..., "language": ..., "content": content})

A file at the default 200 KB limit is 60-70 thousand tokens in a single
request. That is over the per-request limit of most of the models in the
registry and well over the free-tier per-minute token limits of the two
providers a default run uses. The failure is then classified - correctly - as
permanent::

    NON_RETRYABLE_STATUS_CODES = frozenset({400, 401, 403, 404, 413, 422})
    NON_RETRYABLE_MESSAGE_PATTERNS = (..., "context length", "maximum context")

so the file is dropped, after being paid for on the way out. Lowering the size
limit is not a workaround: it drops the same file earlier.

The files this loses are the ones worth having - a generated API client, a
vendored parser, a schema module, ``views.py`` in a mature Django app. This
module keeps the head and the tail and drops the middle, with a marker saying
so, because imports and module docstrings live at the top, the public entry
points and the ``main`` guard at the bottom, and the middle is the most
repetitive part.
"""

from __future__ import annotations

from dataclasses import dataclass

# Characters, not tokens: the conversion is provider-specific and this only has
# to be safe, not exact. At roughly three characters per token this is about
# 13k tokens of content, which leaves room for the prompt and the answer inside
# a 16k context and is comfortably inside a 32k one.
DEFAULT_MAX_CONTENT_CHARS = 40_000

# Below this a budget cannot produce anything useful - the marker alone is
# longer than the excerpt.
MIN_MAX_CONTENT_CHARS = 500

# How much of the budget goes to the top of the file. Imports, the module
# docstring and the class definitions that follow are worth more than the same
# number of characters from the middle.
HEAD_SHARE = 0.6

# Wording stays neutral about what is being cut: the same budget is applied to
# the roll-up's JSON blob, which is not a file.
MARKER_TEMPLATE = "\n\n... {omitted:,} line(s) omitted from the middle ...\n\n"


class InvalidContentBudgetError(ValueError):
    """The requested budget cannot be used, with a reason the user can act on."""


def validate_max_content_chars(value: int | None) -> int | None:
    """Return ``value`` if it is a usable budget.

    ``None`` means "use the default" and is passed through. ``0`` means "send
    the file whole", which is the behaviour this module replaced and is kept
    for anyone on a very large context window.

    Raises
    ------
    InvalidContentBudgetError
        If the value is negative, or too small to hold an excerpt.
    """
    if value is None:
        return None

    try:
        chars = int(value)
    except (TypeError, ValueError) as exc:
        raise InvalidContentBudgetError(
            f"must be a whole number of characters, got {value!r}"
        ) from exc

    if chars == 0:
        return 0

    if chars < 0:
        raise InvalidContentBudgetError(
            f"must be 0 or greater, got {chars}. Use 0 to send files whole."
        )

    if chars < MIN_MAX_CONTENT_CHARS:
        raise InvalidContentBudgetError(
            f"must be at least {MIN_MAX_CONTENT_CHARS} characters, got {chars}."
        )

    return chars


@dataclass(frozen=True)
class BudgetedContent:
    """What is being sent, and what was left out to make it fit."""

    text: str
    truncated: bool = False
    original_chars: int = 0
    original_lines: int = 0
    omitted_lines: int = 0

    @property
    def kept_chars(self) -> int:
        return len(self.text)


def _take_head(lines: list[str], budget: int) -> int:
    """How many leading lines fit in ``budget`` characters."""
    used = 0
    for index, line in enumerate(lines):
        cost = len(line) + 1
        if used + cost > budget:
            return index
        used += cost
    return len(lines)


def _take_tail(lines: list[str], budget: int) -> int:
    """How many trailing lines fit in ``budget`` characters."""
    used = 0
    for count, line in enumerate(reversed(lines), start=1):
        cost = len(line) + 1
        if used + cost > budget:
            return count - 1
        used += cost
    return len(lines)


def apply_content_budget(
    content: str | None,
    max_chars: int | None = None,
) -> BudgetedContent:
    """Cut ``content`` down to ``max_chars``, keeping the head and the tail.

    The cut is made on line boundaries, so the excerpt is never spliced through
    the middle of a statement, and the gap is marked explicitly so the model
    knows it is reading an excerpt and does not describe the file as ending
    where the excerpt ends.

    ``max_chars`` of ``0`` sends the content whole.
    """
    content = content or ""

    if max_chars is None:
        max_chars = DEFAULT_MAX_CONTENT_CHARS

    if max_chars <= 0 or len(content) <= max_chars:
        return BudgetedContent(
            text=content,
            original_chars=len(content),
            original_lines=content.count("\n") + 1 if content else 0,
        )

    lines = content.splitlines()
    marker_cost = len(MARKER_TEMPLATE.format(omitted=len(lines)))
    available = max(0, max_chars - marker_cost)

    head_budget = int(available * HEAD_SHARE)
    head_count = _take_head(lines, head_budget)
    # Whatever the head did not use goes to the tail rather than being wasted.
    tail_budget = available - sum(len(line) + 1 for line in lines[:head_count])
    tail_count = _take_tail(lines[head_count:], tail_budget)

    omitted = len(lines) - head_count - tail_count
    if omitted <= 0:
        # Everything fits after all; nothing was worth cutting.
        return BudgetedContent(
            text=content,
            original_chars=len(content),
            original_lines=len(lines),
        )

    head = lines[:head_count]
    tail = lines[len(lines) - tail_count:] if tail_count else []

    text = "\n".join(head) + MARKER_TEMPLATE.format(omitted=omitted) + "\n".join(tail)

    return BudgetedContent(
        text=text,
        truncated=True,
        original_chars=len(content),
        original_lines=len(lines),
        omitted_lines=omitted,
    )


def annotate_truncation(summary, budgeted: BudgetedContent):
    """Record on a summary that it describes an excerpt.

    An invisible loss of fidelity is worse than a visible one: a downstream
    reader - the roll-up, the README prompt, a human looking at the cache -
    should be able to tell that a description was written from part of a file.
    """
    if not budgeted.truncated or not isinstance(summary, dict):
        return summary

    if "error" in summary:
        return summary

    annotated = dict(summary)
    annotated["truncated"] = True
    annotated["original_lines"] = budgeted.original_lines
    annotated["omitted_lines"] = budgeted.omitted_lines
    return annotated
