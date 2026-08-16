"""Writing the generated README to disk.

The CLI used to do this, at the very end of the run::

    with open(output, "w", encoding="utf-8") as f:
        f.write(readme)

Two problems, both of which cost the user the whole run:

* The path was never looked at until that moment. ``--output
  docs/generated/README.md`` with no ``docs/generated`` directory loaded the
  repository, summarized every file, rolled the summaries up, generated the
  README, reviewed it, regenerated it - and then raised ``FileNotFoundError``.
  The README that had just been produced was never printed and could not be
  recovered.

* ``open(..., "w")`` truncates the destination immediately. An interruption
  between the truncate and the write finishing leaves an empty or half-written
  file where the user's previous README was. ``SummaryCache`` has written
  itself through a temporary file and ``os.replace`` since it was written; the
  actual product of the tool did not get the same treatment.

This module does the checking up front and the writing atomically.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Suffix for the copy kept when an existing README is replaced.
BACKUP_SUFFIX = ".bak"


class OutputPathError(ValueError):
    """The output path cannot be written to, with a reason the user can act on."""


@dataclass(frozen=True)
class OutputTarget:
    """A validated destination for the generated README."""

    path: Path
    exists: bool
    created_parent: bool = False

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.path)


def resolve_output_path(output: str | os.PathLike) -> Path:
    """Expand ``~`` and make the path absolute, without touching the disk."""
    return Path(os.path.expanduser(str(output))).absolute()


def _writable(path: Path) -> bool:
    return os.access(path, os.W_OK)


def prepare_output_path(
    output: str | os.PathLike,
    create_parents: bool = True,
) -> OutputTarget:
    """
    Check that ``output`` can be written, before any expensive work happens.

    Creates the parent directory when ``create_parents`` is set, which is the
    behaviour a user asking for ``docs/generated/README.md`` expects.

    Raises
    ------
    OutputPathError
        If the path is empty, names an existing directory, has a parent that is
        not a directory or cannot be created, or points at a file that cannot
        be replaced. The message says which.
    """
    if output is None or not str(output).strip():
        raise OutputPathError("Output path is empty.")

    path = resolve_output_path(output)

    if path.is_dir():
        raise OutputPathError(
            f"{path} is a directory. Give --output a file path, "
            f"for example {path / 'README.md'}."
        )

    parent = path.parent
    created_parent = False

    if parent.exists():
        if not parent.is_dir():
            raise OutputPathError(
                f"{parent} is not a directory, so {path.name} cannot be "
                "written inside it."
            )
        if not _writable(parent):
            raise OutputPathError(
                f"No permission to write in {parent}. Choose another --output "
                "path, or fix the directory's permissions."
            )
    elif create_parents:
        try:
            parent.mkdir(parents=True, exist_ok=True)
            created_parent = True
        except OSError as exc:
            raise OutputPathError(
                f"Could not create the directory {parent}: {exc}"
            ) from exc
    else:
        raise OutputPathError(
            f"The directory {parent} does not exist. Create it, or point "
            "--output somewhere that does."
        )

    exists = path.exists()
    if exists and not _writable(path):
        raise OutputPathError(
            f"{path} exists but is not writable. Change its permissions, or "
            "choose another --output path."
        )

    return OutputTarget(path=path, exists=exists, created_parent=created_parent)


def backup_path_for(path: str | os.PathLike) -> Path:
    """Where the copy of a replaced file goes."""
    path = Path(path)
    return path.with_name(path.name + BACKUP_SUFFIX)


def write_readme(
    output: str | os.PathLike,
    content: str,
    backup: bool = False,
    encoding: str = "utf-8",
) -> Path:
    """
    Write ``content`` to ``output`` atomically.

    The content goes to a temporary file in the destination's own directory -
    so the final step is a rename within one filesystem - and then replaces the
    destination with :func:`os.replace`. Readers of the path see either the old
    file or the new one, never a half-written one, and an interrupted run
    cannot destroy an existing README.

    With ``backup`` set, a copy of the file being replaced is kept alongside it
    with a ``.bak`` suffix.

    Returns the path written.

    Raises
    ------
    OSError
        If the write fails. The destination is untouched when it does.
    """
    path = resolve_output_path(output)
    directory = path.parent

    if backup and path.exists():
        shutil.copy2(path, backup_path_for(path))

    # Mode is taken from the file being replaced, if there is one, so an
    # existing README does not silently become 0600 after a regeneration.
    existing_mode = None
    if path.exists():
        try:
            existing_mode = stat.S_IMODE(path.stat().st_mode)
        except OSError:
            existing_mode = None

    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        if existing_mode is not None:
            os.chmod(tmp_path, existing_mode)
        else:
            # mkstemp creates 0600; a README is an ordinary file.
            os.chmod(tmp_path, 0o644 & ~_umask())

        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return path


def _umask() -> int:
    """Read the process umask without leaving it changed."""
    current = os.umask(0)
    os.umask(current)
    return current
