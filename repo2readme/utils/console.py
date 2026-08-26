"""Two ways of putting text on the console, and a rule for choosing between them.

Everything the CLI prints goes through ``rich.print``, which parses ``[...]`` as
a style tag. That is the right thing for a line the CLI wrote itself
(``[green]Saved[/green]``) and the wrong thing for anything that came from
somewhere else, because square brackets are ordinary characters in the values
the CLI prints:

* the generated README - a table of contents is a column of ``[label](#anchor)``
  links, and ``[!NOTE]`` is GitHub's callout syntax;
* file paths - ``src/[id]/page.tsx`` is how a Next.js app spells a dynamic
  route;
* the text of an exception, which usually quotes one of the above.

An unknown tag such as ``[key]`` is deleted. A *known* one such as ``[i]`` is
deleted and switches italics on for everything after it. An unclosed one raises
``rich.errors.MarkupError`` and takes the run down. And when stdout is not a
terminal Rich still renders to a console of a fixed width, so a redirected
document comes out hard-wrapped at 80 columns.

So::

    repo2readme run --local . > README.md

used to produce a file that was not the README the model wrote.

This module draws the line. :func:`echo_document` writes a document to stdout
byte for byte, with no markup pass and no wrapping. :func:`safe` escapes a value
so it can be interpolated into a Rich string without being read as markup.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

import click
from rich.markup import escape


def safe(value: Any) -> str:
    """Escape ``value`` for interpolation into a Rich-markup string.

    ``rprint(f"[red]{safe(exc)}[/red]")`` prints the exception's text, brackets
    and all, instead of letting it close the ``[red]`` tag early or vanish.
    """
    return escape("" if value is None else str(value))


def echo_document(text: str, file: TextIO | None = None) -> None:
    """Write ``text`` to stdout exactly as it is.

    No markup parsing, no styling, no wrapping - the caller is printing a
    document, not a message. A single trailing newline is ensured so the shell
    prompt starts on its own line and a redirected file ends the way a text file
    should; a document that already ends in one does not get a second.
    """
    stream = file if file is not None else sys.stdout

    if text and not text.endswith("\n"):
        text += "\n"

    click.echo(text, file=stream, nl=False)


def echo_documents(*texts: str, file: TextIO | None = None) -> None:
    """Write several documents in order, each through :func:`echo_document`."""
    for text in texts:
        echo_document(text, file=file)
