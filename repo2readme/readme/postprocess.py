"""Clean up and check the generated README before it is written to disk.

The generation prompt asks the model to return only valid Markdown, to keep
table-of-contents anchors matching the headings and to avoid placeholder
images. Those are requests, not guarantees, and the result used to be written
byte-for-byte. This module fixes what can be fixed mechanically and reports the
rest instead of silently rewriting the model's content.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

# A fence that opens the document and closes it at the very end, e.g. the model
# answering with ```markdown ... ```.
_FENCE_LINE = re.compile(r"^\s*(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)\s*$")

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")
_HTML_IMAGE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_HTML_SRC = re.compile(r"""\bsrc\s*=\s*["']([^"']*)["']""", re.IGNORECASE)

# Characters GitHub strips when building a heading anchor.
_ANCHOR_STRIP = re.compile(r"[^\w\- ]", re.UNICODE)

# Targets that render as a broken image on GitHub.
PLACEHOLDER_PATTERNS = (
    "path/to",
    "path_to",
    "your-",
    "your_",
    "yourusername",
    "your-username",
    "example.com",
    "<url>",
    "url-here",
    "insert-",
    "todo",
    "placeholder",
)

BLANK_LINE_LIMIT = 2


@dataclass(frozen=True)
class ValidationIssue:
    """Something worth telling the user about, but not worth rewriting."""

    kind: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.kind}: {self.message}"


def github_anchor(heading: str) -> str:
    """Slug for a heading, following GitHub's rules.

    Lowercase, punctuation and emoji dropped, spaces turned into hyphens.
    """
    text = heading.strip().lower()
    # Inline links in a heading contribute only their label.
    text = _MD_LINK.sub(r"\1", text)
    text = text.replace("`", "")
    text = _ANCHOR_STRIP.sub("", text)
    return text.strip().replace(" ", "-")


def fenced_flags(lines: Sequence[str]) -> list[bool]:
    """Mark every line that belongs to a fenced code block.

    The fence delimiters count as part of the block, so a ``True`` run covers
    the opening line, the code and the closing line. A fence is closed only by
    the marker character that opened it, which is how a ``~~~`` block can
    contain a line of backticks.

    An unclosed fence runs to the end of the document. That is deliberate:
    everything after it is code as far as anything can tell, and the validators
    have always read it that way.
    """
    flags: list[bool] = []
    fence: str | None = None

    for line in lines:
        match = _FENCE_LINE.match(line)
        if match:
            marker = match.group(1)[0]
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            flags.append(True)
            continue

        flags.append(fence is not None)

    return flags


def iter_prose_lines(text: str) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, line)`` for lines outside fenced code blocks.

    A usage section full of example Markdown must not be mistaken for the
    document's own headings and links.
    """
    lines = text.splitlines()

    for number, (line, fenced) in enumerate(zip(lines, fenced_flags(lines)), start=1):
        if not fenced:
            yield number, line


def strip_wrapping_code_fence(text: str) -> str:
    """Unwrap a document the model returned inside a single code fence.

    Only an opening fence on the first non-empty line with its matching close
    on the last non-empty line is unwrapped, so a README that merely starts
    with a code example is left alone.
    """
    lines = text.splitlines()

    first = 0
    while first < len(lines) and not lines[first].strip():
        first += 1

    last = len(lines) - 1
    while last >= 0 and not lines[last].strip():
        last -= 1

    if first >= last:
        return text

    opening = _FENCE_LINE.match(lines[first])
    closing = _FENCE_LINE.match(lines[last])
    if not opening or not closing:
        return text

    # The closing fence must be bare; ```python ... ```python is not a wrapper.
    if closing.group(2):
        return text

    if opening.group(1)[0] != closing.group(1)[0]:
        return text

    inner = lines[first + 1:last]

    # If the inner content still contains an unbalanced fence, this was a code
    # block rather than a wrapper.
    fence_count = sum(1 for line in inner if _FENCE_LINE.match(line))
    if fence_count % 2:
        return text

    return "\n".join(inner)


def normalize_prose_line(line: str, continues: bool = False) -> str:
    """Strip trailing whitespace from one line of prose, keeping a hard break.

    Two or more trailing spaces are Markdown's hard line break. Removing them
    where one is doing its job is not whitespace cleanup - it joins two lines
    into one paragraph - so the break is kept, normalised to the canonical two
    spaces.

    It only does its job in the middle of a paragraph, which is what
    ``continues`` reports: the next line has to exist, be prose and not be
    blank. Trailing spaces on a heading, or on the last line before a blank
    line, a code fence or the end of the document, render as nothing and are
    stripped like any other trailing whitespace.
    """
    stripped = line.rstrip()
    if not stripped:
        return ""

    if not continues or _HEADING.match(stripped):
        return stripped

    trailing = line[len(stripped):]
    if trailing.count(" ") >= 2:
        return stripped + "  "

    return stripped


def normalize_markdown(text: str) -> str:
    """Mechanical cleanup only: nothing here changes the document's meaning.

    Which is why none of it reaches inside a fenced code block. Trailing
    whitespace is content in a ``diff`` block, where a blank context line is
    two spaces; in a fixture or expected-output block, which exists to be
    compared byte for byte; and in any snippet whose language cares. Blank runs
    are the snippet's own formatting. The README's usage and installation
    sections are made of these blocks, and the model was told to produce them
    verbatim.
    """
    if not text or not text.strip():
        return ""

    text = strip_wrapping_code_fence(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    fenced_lines = fenced_flags(lines)

    collapsed: list[str] = []
    blanks = 0
    for index, (line, fenced) in enumerate(zip(lines, fenced_lines)):
        if fenced:
            blanks = 0
            collapsed.append(line)
            continue

        following = index + 1
        continues = (
            following < len(lines)
            and not fenced_lines[following]
            and bool(lines[following].strip())
        )
        normalized = normalize_prose_line(line, continues=continues)
        if normalized:
            blanks = 0
            collapsed.append(normalized)
            continue

        blanks += 1
        if blanks <= BLANK_LINE_LIMIT:
            collapsed.append(normalized)

    while collapsed and not collapsed[0].strip():
        collapsed.pop(0)
    while collapsed and not collapsed[-1].strip():
        collapsed.pop()

    return "\n".join(collapsed) + "\n"


def extract_headings(text: str) -> list[tuple[int, str, str]]:
    """Return ``(level, text, anchor)`` for every heading outside code blocks."""
    headings: list[tuple[int, str, str]] = []
    seen: dict[str, int] = {}

    for _, line in iter_prose_lines(text):
        match = _HEADING.match(line)
        if not match:
            continue

        level = len(match.group(1))
        heading_text = match.group(2).strip()
        anchor = github_anchor(heading_text)

        # GitHub disambiguates repeated anchors with -1, -2, ...
        count = seen.get(anchor, 0)
        seen[anchor] = count + 1
        if count:
            anchor = f"{anchor}-{count}"

        headings.append((level, heading_text, anchor))

    return headings


def extract_internal_links(text: str) -> list[tuple[str, str]]:
    """Return ``(label, target)`` for same-document links outside code blocks."""
    links: list[tuple[str, str]] = []

    for _, line in iter_prose_lines(text):
        for match in _MD_LINK.finditer(line):
            label, target = match.group(1), match.group(2).strip()
            if target.startswith("#"):
                links.append((label, target))

    return links


def find_broken_anchor_links(text: str) -> list[ValidationIssue]:
    """Table-of-contents entries that point at no heading."""
    anchors = {anchor for _, _, anchor in extract_headings(text)}
    issues: list[ValidationIssue] = []

    for label, target in extract_internal_links(text):
        if target.lstrip("#") not in anchors:
            issues.append(
                ValidationIssue(
                    kind="broken-anchor",
                    message=f"link {target!r} (\"{label}\") matches no heading",
                )
            )

    return issues


def _is_placeholder_target(target: str) -> bool:
    lowered = target.strip().lower()
    if not lowered:
        return True
    return any(pattern in lowered for pattern in PLACEHOLDER_PATTERNS)


def find_placeholder_images(text: str) -> list[ValidationIssue]:
    """Images with an empty or obviously made-up target."""
    issues: list[ValidationIssue] = []

    for _, line in iter_prose_lines(text):
        for match in _MD_IMAGE.finditer(line):
            alt, target = match.group(1), match.group(2).strip()
            if _is_placeholder_target(target):
                issues.append(
                    ValidationIssue(
                        kind="placeholder-image",
                        message=f"image \"{alt}\" has a placeholder target {target!r}",
                    )
                )

        for tag in _HTML_IMAGE.finditer(line):
            src_match = _HTML_SRC.search(tag.group(0))
            src = src_match.group(1) if src_match else ""
            if _is_placeholder_target(src):
                issues.append(
                    ValidationIssue(
                        kind="placeholder-image",
                        message=f"<img> tag has a placeholder src {src!r}",
                    )
                )

    return issues


def find_heading_problems(text: str) -> list[ValidationIssue]:
    """Missing or duplicated top-level heading."""
    headings = extract_headings(text)
    top_level = [heading for level, heading, _ in headings if level == 1]

    if not top_level:
        return [ValidationIssue(kind="missing-h1", message="no top-level heading found")]

    if len(top_level) > 1:
        return [
            ValidationIssue(
                kind="duplicate-h1",
                message=f"{len(top_level)} top-level headings: {', '.join(top_level)}",
            )
        ]

    return []


def validate_markdown(text: str) -> list[ValidationIssue]:
    """Report structural problems without changing the document."""
    if not text.strip():
        return [ValidationIssue(kind="empty", message="the generated README is empty")]

    return [
        *find_heading_problems(text),
        *find_broken_anchor_links(text),
        *find_placeholder_images(text),
    ]


def postprocess_readme(text: str) -> tuple[str, list[ValidationIssue]]:
    """Normalize the README and report what could not be fixed mechanically."""
    normalized = normalize_markdown(text)
    return normalized, validate_markdown(normalized)
