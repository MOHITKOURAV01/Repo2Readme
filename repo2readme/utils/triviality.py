"""Decide whether a file has anything worth summarizing in it.

``__init__.py`` used to be in the blanket ignore list::

    IGNORE_FILES = {..., "__init__.py", ...}

which is a name-based approximation of the rule that was actually wanted:
*don't spend an API call on an empty marker file*. The approximation dropped
every package's public API - the re-exports, ``__all__``, ``__version__``, the
package docstring - and, less obviously, it made the dependency graph's package
resolution unreachable, because every candidate it builds ends in
``__init__.py`` and no such path was ever in the file map.

The rule is expressed here as what it means. A file is skipped when nothing is
left of it once whitespace and comments are removed - regardless of its name,
so a zero-byte ``conftest.py`` or a ``.js`` holding one license header costs
nothing either.

Docstrings are content, not comments. A ``mypkg/__init__.py`` whose entire body
is ``\"\"\"Order service.\"\"\"`` is the single most useful line in the package
for a README, and it is kept.
"""

from __future__ import annotations

import re

# Line-comment markers by the language names ``detect_lang`` returns. A
# language that is absent simply has none of its comments stripped, which can
# only make a file look less empty than it is - the safe direction, since the
# consequence of guessing wrong is a wasted request rather than a lost file.
LINE_COMMENT_MARKERS: dict[str, tuple[str, ...]] = {
    "python": ("#",),
    "ruby": ("#",),
    "perl": ("#",),
    "bash": ("#",),
    "powershell": ("#",),
    "r": ("#",),
    "yaml": ("#",),
    "toml": ("#",),
    "makefile": ("#",),
    "dockerfile": ("#",),
    "procfile": ("#",),
    "just": ("#",),
    "cmake": ("#",),
    "ini": ("#", ";"),
    "javascript": ("//",),
    "typescript": ("//",),
    "java": ("//",),
    "go": ("//",),
    "rust": ("//",),
    "c": ("//",),
    "cpp": ("//",),
    "csharp": ("//",),
    "php": ("//", "#"),
    "swift": ("//",),
    "kotlin": ("//",),
    "scala": ("//",),
    "groovy": ("//",),
    "css": (),
    "scss": ("//",),
    "less": ("//",),
    "sql": ("--",),
    "batch": ("rem", "::"),
}

# Languages whose block comments are stripped before the emptiness check.
_C_STYLE_BLOCK = ("javascript", "typescript", "java", "go", "rust", "c", "cpp",
                  "csharp", "php", "swift", "kotlin", "scala", "groovy", "css",
                  "scss", "less")
_HTML_STYLE_BLOCK = ("html", "xml", "markdown")

_C_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_HTML_BLOCK_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

SKIP_REASON = "no readable content"


def strip_comments(content: str, language: str | None = None) -> str:
    """Remove the comments ``language`` uses, leaving everything else.

    Only whole-line comments are removed. A trailing comment after real code
    leaves the code behind, which is all this needs: the question is whether
    *anything* remains, not how much.
    """
    if not content:
        return ""

    language = (language or "").strip().lower()

    if language in _C_STYLE_BLOCK:
        content = _C_BLOCK_COMMENT.sub("", content)
    elif language in _HTML_STYLE_BLOCK:
        content = _HTML_BLOCK_COMMENT.sub("", content)

    markers = LINE_COMMENT_MARKERS.get(language)
    if not markers:
        return content

    kept: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and any(
            stripped.lower().startswith(marker) for marker in markers
        ):
            continue
        kept.append(line)
    return "\n".join(kept)


def is_effectively_empty(content: str | None, language: str | None = None) -> bool:
    """Whether ``content`` has nothing a summary could be written from.

    True for an empty file, a file of whitespace, and a file whose only lines
    are comments. False for anything else - including a file holding only a
    docstring, which is content.
    """
    if content is None:
        return True

    if not content.strip():
        return True

    return not strip_comments(content, language).strip()
