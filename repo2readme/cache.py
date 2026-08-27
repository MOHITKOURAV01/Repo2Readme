import hashlib
import json
import logging
import os
import tempfile
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 1.1 added the per-entry "repository" field. An older cache is invalidated
# once on upgrade rather than migrated: the entries it holds are unscoped, and
# guessing which repository each one came from is not worth a wrong answer.
CACHE_SCHEMA_VERSION = "1.1"

# How often an autosaving cache writes, in updates. The default of 1 keeps the
# durability guarantee a library caller gets today: every put() is on disk when
# it returns. The CLI knows its own lifecycle and turns autosave off, flushing
# once at the end, which is where the quadratic cost actually mattered.
DEFAULT_AUTOSAVE_EVERY = 1

# Expected fields for each cache entry
EXPECTED_ENTRY_FIELDS = {
    "file_path",
    "content_hash",
    "language",
    "summary",
    "mtime",
}

# Written by a scoped cache, absent from one built without a repository. Not in
# EXPECTED_ENTRY_FIELDS so that an entry from an older schema still validates -
# the schema version check is what rebuilds it, with a message that explains
# itself, rather than a structural warning about a missing field.
REPOSITORY_FIELD = "repository"


def _validate_cache_structure(data: Any) -> bool:
    """
    Validate that the loaded cache data has the expected structure.

    Returns True if valid, False otherwise.
    """
    if not isinstance(data, dict):
        logger.warning("Cache root is not a dictionary, got %s", type(data).__name__)
        return False

    if "schema_version" not in data:
        logger.warning("Cache missing 'schema_version'")
        return False

    if "config_hash" not in data:
        logger.warning("Cache missing 'config_hash'")
        return False

    entries = data.get("entries")
    if not isinstance(entries, list):
        logger.warning(
            "Cache 'entries' is not a list, got %s", type(entries).__name__
        )
        return False

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            logger.warning("Cache entry %d is not a dictionary, got %s", i, type(entry).__name__)
            return False
        missing = EXPECTED_ENTRY_FIELDS - set(entry.keys())
        if missing:
            logger.warning(
                "Cache entry %d missing fields: %s", i, ", ".join(sorted(missing))
            )
            return False

    return True


class SummaryCache:
    """
    File-level summary cache with configuration-aware invalidation.

    Cache entries are keyed by file path and content hash. The cache is
    invalidated when summarization configuration (provider, model, base_url,
    prompt template) changes or when the cache schema version changes.

    Lookups and updates are O(1): the entry list that is written to disk is
    mirrored by an in-memory index keyed on file path, holding the same entry
    objects, so an update mutates the entry in place instead of rebuilding the
    list.

    Writes are decoupled from updates. With ``autosave`` enabled the cache
    writes at most every ``autosave_every`` updates rather than on every one;
    with it disabled the caller decides, by calling :meth:`flush` or by using
    the cache as a context manager. Either way nothing is lost: a dirty flag
    tracks whether the in-memory state differs from disk.

    Thread-safe: all public methods acquire an instance-level lock, and the
    disk write itself is serialised separately so workers are not blocked on
    fsync while another thread is writing. Where both locks are held, ``_lock``
    is always taken before ``_io_lock``.

    A failed write is not fatal - the cache is an optimisation - but it is never
    reported as a success: the dirty flag survives it so a later flush retries,
    :meth:`flush` returns ``False``, and the reason is available from
    :attr:`last_write_error` and counted in :meth:`stats`.

    Entries are scoped to the repository they were produced for. One cache file
    holds the summaries of every repository analyzed from the same working
    directory, and :meth:`get_deleted_files` answers "which of my entries no
    longer exist?" - so without a scope it reported another repository's
    entries as deleted, and the caller dutifully removed them. Pass
    ``repository`` (see ``repo2readme.loaders.source.repository_identity``) and
    every entry is stamped with it; lookups and pruning then only ever see the
    current repository's work. ``repository=None`` keeps the old unscoped
    behaviour for callers that hold one cache per repository themselves.
    """

    def __init__(
        self,
        cache_dir: str,
        config: dict,
        prompt_template_hash: str,
        autosave: bool = True,
        autosave_every: int = DEFAULT_AUTOSAVE_EVERY,
        repository: Optional[str] = None,
    ):
        self.cache_dir = cache_dir
        self.config = config
        self.prompt_template_hash = prompt_template_hash
        self.repository = repository or None
        self.cache_file = os.path.join(cache_dir, "summaries.json")
        self.schema_version = CACHE_SCHEMA_VERSION
        self.autosave = autosave
        self.autosave_every = max(1, int(autosave_every))
        self._data: Optional[dict] = None
        # file_path -> the entry dict that also lives in self._data["entries"]
        self._index: dict[str, dict] = {}
        self._dirty = False
        self._pending_updates = 0
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._last_write_error: Optional[str] = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "updates": 0,
            "removals": 0,
            "invalidations": 0,
            "disk_writes": 0,
            "write_failures": 0,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "SummaryCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()

    def flush(self) -> bool:
        """
        Write pending changes to disk. Returns True if the write succeeded.

        Serialisation happens under the state lock; the disk write itself does
        not, so summarization workers are not blocked while the file is being
        replaced.

        The dirty flag is only cleared once the bytes are on disk. Clearing it
        up front meant a failed write threw the pending work away: the CLI
        flushes exactly once, at the end of the run, so a single unwritable
        cache file silently discarded every summary the run had produced.
        """
        with self._lock:
            if not self._dirty or self._data is None:
                return False
            payload = self._serialize()
            pending = self._pending_updates
            self._dirty = False
            self._pending_updates = 0

        if self._write(payload):
            return True

        with self._lock:
            # Another thread may have made changes of its own while the write
            # was in flight; either way there is something to write again.
            self._dirty = True
            self._pending_updates = max(self._pending_updates, pending)
        return False

    @property
    def last_write_error(self) -> Optional[str]:
        """Why the last write failed, or ``None`` if the last one succeeded."""
        with self._io_lock:
            return self._last_write_error

    def stats(self) -> dict:
        """
        Counters for the current process: cache hits and misses, in-memory
        updates and removals, invalidations, how many times the file was
        actually rewritten, and how many writes failed.
        """
        with self._lock, self._io_lock:
            return dict(self._stats)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_cache_dir(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)

    def _compute_config_hash(self) -> str:
        config_str = json.dumps(
            {
                "provider": self.config.get("provider"),
                "model": self.config.get("model"),
                "base_url": self.config.get("base_url"),
                "prompt_template_hash": self.prompt_template_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(config_str.encode()).hexdigest()

    @staticmethod
    def _compute_content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _rebuild(self) -> None:
        """Reset cache data to a fresh state."""
        self._data = {
            "schema_version": self.schema_version,
            "config_hash": self._compute_config_hash(),
            "entries": [],
        }
        self._index = {}

    def _reindex(self) -> None:
        """Point the index at the entry objects currently in ``_data``."""
        self._index = {
            entry["file_path"]: entry
            for entry in self._data.get("entries", [])
            if isinstance(entry, dict) and "file_path" in entry
        }

    def _load(self) -> None:
        if self._data is not None:
            return

        self._ensure_cache_dir()
        if not os.path.exists(self.cache_file):
            self._rebuild()
            return

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not _validate_cache_structure(data):
                logger.warning("Cache structure validation failed, rebuilding")
                self._rebuild()
                return

            self._data = data
            self._reindex()
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Cache file corrupted or unreadable, rebuilding: {e}")
            self._rebuild()

    def _serialize(self) -> str:
        """Render the current state. Caller must hold ``_lock``."""
        return json.dumps(self._data, indent=2)

    def _write(self, payload: str) -> bool:
        """
        Atomically replace the cache file with ``payload``.

        Returns True if the bytes reached the disk. A failure is logged at error
        level rather than raised - losing the cache costs time on the next run,
        not correctness - but it is reported, because the caller has to know not
        to treat the pending work as saved.

        The counters are updated under ``_io_lock``, which this method already
        holds; incrementing them outside it let concurrent writers lose an
        increment.
        """
        with self._io_lock:
            self._ensure_cache_dir()
            try:
                fd, tmp_path = tempfile.mkstemp(
                    dir=self.cache_dir, prefix="summaries_", suffix=".json.tmp"
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(payload)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, self.cache_file)
                except Exception:
                    # Clean up temp file on failure
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
            except OSError as e:
                logger.error("Failed to write cache file %s: %s", self.cache_file, e)
                self._last_write_error = str(e)
                self._stats["write_failures"] += 1
                return False

            self._last_write_error = None
            self._stats["disk_writes"] += 1
            return True

    def _save(self) -> bool:
        """
        Write immediately. Caller must hold ``_lock``. Returns True on success.

        Kept for callers that need the write to have happened by the time they
        return; the batched path goes through :meth:`_touch` instead. On failure
        the state stays dirty so the next flush tries again.
        """
        payload = self._serialize()
        pending = self._pending_updates
        self._dirty = False
        self._pending_updates = 0

        if self._write(payload):
            return True

        self._dirty = True
        self._pending_updates = pending
        return False

    def _touch(self) -> None:
        """
        Record that in-memory state changed. Caller must hold ``_lock``.

        The whole file has to be rewritten for any change, so writing once per
        entry made a run of N files cost N full serialisations of a file that
        grows to hold N entries. Batching turns that into N / autosave_every.
        """
        self._dirty = True
        self._pending_updates += 1

        if self.autosave and self._pending_updates >= self.autosave_every:
            self._save()

    def _in_scope(self, entry: dict) -> bool:
        """Whether ``entry`` belongs to the repository this cache is scoped to.

        An unscoped cache owns everything, which is what a caller keeping one
        cache file per repository already relied on.
        """
        if self.repository is None:
            return True
        return entry.get(REPOSITORY_FIELD) == self.repository

    def _scoped_entries(self) -> list:
        """Every entry belonging to the current repository. Lock held."""
        return [
            entry
            for entry in self._data.get("entries", [])
            if isinstance(entry, dict) and self._in_scope(entry)
        ]

    def _find_entry(self, file_path: str) -> Optional[dict]:
        entry = self._index.get(file_path)
        if entry is None or not self._in_scope(entry):
            return None
        return entry

    def _clear_entries(self, config_hash: str) -> None:
        """Drop every entry and adopt the current configuration. Lock held."""
        self._data["entries"] = []
        self._index = {}
        self._data["config_hash"] = config_hash
        self._data["schema_version"] = self.schema_version
        self._stats["invalidations"] += 1
        # Invalidation used to be applied in memory and never written, so the
        # stale entries and the stale config hash survived on disk and were
        # re-detected (and re-logged) on every subsequent run.
        self._dirty = True
        self._pending_updates += 1
        if self.autosave:
            self._save()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, file_path: str, content: str, language: str) -> Optional[dict]:
        """
        Return cached summary if valid, otherwise None.
        """
        with self._lock:
            self._load()

            # Invalidate if configuration changed
            current_config_hash = self._compute_config_hash()
            if self._data.get("config_hash") != current_config_hash:
                logger.info("Configuration changed, invalidating cache")
                self._clear_entries(current_config_hash)
                self._stats["misses"] += 1
                return None

            # Invalidate if schema version changed
            if self._data.get("schema_version") != self.schema_version:
                logger.info(
                    "Cache schema version changed from %s to %s, invalidating cache",
                    self._data.get("schema_version"),
                    self.schema_version,
                )
                self._clear_entries(current_config_hash)
                self._stats["misses"] += 1
                return None

            entry = self._find_entry(file_path)
            if entry is None:
                self._stats["misses"] += 1
                return None

            content_hash = self._compute_content_hash(content)
            if entry.get("content_hash") != content_hash:
                self._stats["misses"] += 1
                return None

            # Language mismatch could indicate detection logic changed
            if entry.get("language") != language:
                logger.debug(
                    "Language mismatch for %s: cached=%s, current=%s",
                    file_path,
                    entry.get("language"),
                    language,
                )
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return entry.get("summary")

    def put(
        self, file_path: str, content: str, language: str, summary: dict, mtime: float
    ) -> None:
        """
        Store summary in cache.
        """
        with self._lock:
            self._load()

            payload = {
                "file_path": file_path,
                "content_hash": self._compute_content_hash(content),
                "language": language,
                "summary": summary,
                "mtime": mtime,
            }
            if self.repository is not None:
                payload[REPOSITORY_FIELD] = self.repository

            existing = self._index.get(file_path)
            if existing is None:
                self._data["entries"].append(payload)
                self._index[file_path] = payload
            else:
                # The index holds the same object that is in the entry list, so
                # updating in place keeps both in sync without an O(N) rebuild.
                # An entry another repository left at this path is overwritten
                # rather than merged, so one path is never claimed twice.
                existing.clear()
                existing.update(payload)

            self._stats["updates"] += 1
            self._touch()

    def get_deleted_files(self, current_files: set) -> list:
        """
        Return cache entries for files that no longer exist in current_files.

        Only entries belonging to this cache's repository are considered.
        ``current_files`` is the file list of one repository, so without that
        restriction every entry from every *other* repository analyzed from the
        same working directory was reported as deleted - and the CLI removes
        what this returns.
        """
        with self._lock:
            self._load()
            return [
                entry
                for entry in self._scoped_entries()
                if entry.get("file_path") not in current_files
            ]

    def entries_for_repository(self) -> list:
        """Copies of the entries belonging to this cache's repository."""
        with self._lock:
            self._load()
            return [dict(entry) for entry in self._scoped_entries()]

    def remove_entries(self, file_paths: list) -> None:
        """
        Remove specific entries from cache.
        """
        with self._lock:
            self._load()
            paths_to_remove = set(file_paths)
            if not paths_to_remove:
                return

            remaining = [
                entry
                for entry in self._data.get("entries", [])
                if not (
                    entry.get("file_path") in paths_to_remove
                    and self._in_scope(entry)
                )
            ]
            removed = len(self._data.get("entries", [])) - len(remaining)
            if not removed:
                return

            self._data["entries"] = remaining
            self._reindex()
            self._stats["removals"] += removed
            self._dirty = True
            self._pending_updates += 1
            if self.autosave:
                self._save()
