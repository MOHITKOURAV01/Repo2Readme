"""One place that decides what console output is markup and what is data.

``rich.print`` parses square brackets as style tags. That is what makes
``"[red]failed[/red]"`` render in red, and it is also what makes a file called
``[slug].tsx`` render as ``.tsx``: Rich cannot tell a style tag the caller wrote
from one that arrived inside a filename.

Nothing in a repository, and nothing in a provider's error message, is markup.
Both are data that happens to be printed. Every value from outside this
program - a path, an exception, a provider's error text - therefore has to be
escaped on its way into a Rich string, and the only reliable way to do that is
to have one obvious helper and use it everywhere.

The logging path already made this decision, by constructing its handler with
``markup=False``; these helpers are the same decision for the lines that are
printed rather than logged.

    rprint(f"Saved to {escaped(destination)}")
    rprint(f"[red]{escaped(exc)}[/red]")
    rprint(f"✓ {styled(relative_path, 'green')}")

The style tags a caller writes as literals are left alone, because those really
are markup. Only the interpolated values are escaped.
"""

from __future__ import annotations

from rich.markup import escape

__all__ = ["escaped", "styled"]


def escaped(value: object) -> str:
    """``value`` as a string Rich will render exactly as it reads.

    Accepts any object rather than only ``str`` so that exceptions and
    ``Path`` objects can be interpolated directly, which is how they are almost
    always used::

        rprint(f"[red]Failed to load repository: {escaped(exc)}[/red]")

    An empty string stays empty, so a value that is optional does not have to
    be guarded at the call site.
    """
    return escape(str(value))


def styled(value: object, style: str) -> str:
    """``value`` wrapped in a Rich style tag it cannot break out of.

    The wrapping tags come from ``style``, which is written by this program, so
    they are emitted as markup. ``value`` is escaped first, which is what stops
    a path containing ``[/green]`` from closing the tag early and raising
    ``MarkupError``, and what stops a path containing ``[slug]`` from being
    swallowed as an unknown tag.
    """
    return f"[{style}]{escaped(value)}[/{style}]"
