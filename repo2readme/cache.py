import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Sentinel for "the caller did not pass this", so that None can mean "no bound".
_UNSET: Any = object()

CACHE_SCHEMA_VERSION = "1.1"

# Versions this one can read without discarding what is already on disk.
MIGRATABLE_SCHEMA_VERSIONS = ("1.0",)

# How often an autosaving cache writes, in updates. The default of 1 keeps the
# durability guarantee a library caller gets today: every put() is on disk when
# it returns. The CLI knows its own lifecycle and turns autosave off, flushing
# once at the end, which is where the quadratic cost actually mattered.
DEFAULT_AUTOSAVE_EVERY = 1

# The cache had no bound of any kind: entries were only ever removed for files
# that no longer exist in the repository being processed right now. Used across
# a handful of repositories it grew to tens of megabytes, and the whole file is
# parsed on the first lookup and rewritten on every flush.
DEFAULT_MAX_ENTRIES = 5000
DEFAULT_MAX_AGE_DAYS = 90

SECONDS_PER_DAY = 86400

# Expected fields for each cache entry
EXPECTED_ENTRY_FIELDS = {
    "file_path",
    "content_hash",
    "language",
    "summary",
    "mtime",
}

# Added in schema 1.1. Missing values are backfilled rather than rejected, so a
# 1.0 cache is migrated instead of thrown away.
OPTIONAL_ENTRY_FIELDS = {
    "created_at",
    "last_used_at",
    "namespace",
}


@dataclass(frozen=True)
class PruneReport:
    """What a :meth:`SummaryCache.prune` call removed."""

    expired: int
    evicted: int
    entries_before: int
    entries_after: int

    @property
    def removed(self) -> int:
        return self.entries_before - self.entries_after


@dataclass(frozen=True)
class CacheSummary:
    """A description of the cache on disk, for ``repo2readme cache info``."""

    cache_file: str
    exists: bool
    entries: int
    size_bytes: int
    schema_version: str
    namespaces: dict = field(default_factory=dict)
    oldest_created_at: float | None = None
    newest_created_at: float | None = None

    @property
    def repositories(self) -> int:
        return len(self.namespaces)


def _created_at(entry: dict, default: float) -> float:
    value = entry.get("created_at")
    return value if isinstance(value, (int, float)) else default


def _last_used_at(entry: dict, default: float) -> float:
    value = entry.get("last_used_at")
    if isinstance(value, (int, float)):
        return value
    return _created_at(entry, default)


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


def _migrate_entries(data: dict, now: float) -> bool:
    """Backfill the fields schema 1.1 added. Returns True if anything changed.

    Schema 1.0 entries carry no timestamps at all, which is why nothing could
    expire. Rather than discard a cache that is otherwise perfectly usable,
    give its entries the current time: they then age out from now on, which is
    the conservative reading of "we do not know how old this is".
    """
    changed = False

    for entry in data.get("entries", []):
        if not isinstance(entry, dict):
            continue
        if "created_at" not in entry:
            entry["created_at"] = now
            changed = True
        if "last_used_at" not in entry:
            entry["last_used_at"] = entry.get("created_at", now)
            changed = True

    if data.get("schema_version") != CACHE_SCHEMA_VERSION:
        data["schema_version"] = CACHE_SCHEMA_VERSION
        changed = True

    return changed


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
    fsync while another thread is writing.
    """

    def __init__(
        self,
        cache_dir: str,
        config: dict,
        prompt_template_hash: str,
        autosave: bool = True,
        autosave_every: int = DEFAULT_AUTOSAVE_EVERY,
        namespace: str | None = None,
        max_entries: int | None = DEFAULT_MAX_ENTRIES,
        max_age_days: float | None = DEFAULT_MAX_AGE_DAYS,
    ):
        self.cache_dir = cache_dir
        self.config = config
        self.prompt_template_hash = prompt_template_hash
        self.cache_file = os.path.join(cache_dir, "summaries.json")
        self.schema_version = CACHE_SCHEMA_VERSION
        self.autosave = autosave
        self.autosave_every = max(1, int(autosave_every))
        # Which repository this instance is working on. Entries record it, so
        # a second repository analysed from the same directory is no longer
        # reported as "these files were deleted" and evicted.
        self.namespace = namespace
        self.max_entries = max_entries
        self.max_age_days = max_age_days
        self._data: Optional[dict] = None
        # file_path -> the entry dict that also lives in self._data["entries"]
        self._index: dict[str, dict] = {}
        self._dirty = False
        self._pending_updates = 0
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "updates": 0,
            "removals": 0,
            "invalidations": 0,
            "disk_writes": 0,
            "evictions": 0,
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
        Write pending changes to disk. Returns True if a write happened.

        Serialisation happens under the state lock; the disk write itself does
        not, so summarization workers are not blocked while the file is being
        replaced.
        """
        with self._lock:
            if not self._dirty or self._data is None:
                return False
            payload = self._serialize()
            self._dirty = False
            self._pending_updates = 0

        self._write(payload)
        return True

    def stats(self) -> dict:
        """
        Counters for the current process: cache hits and misses, in-memory
        updates and removals, invalidations, and how many times the file was
        actually rewritten.
        """
        with self._lock:
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

            # A 1.0 cache is upgraded in place rather than discarded: the only
            # difference is bookkeeping the entries did not carry.
            if data.get("schema_version") in MIGRATABLE_SCHEMA_VERSIONS:
                logger.info(
                    "Migrating cache from schema %s to %s",
                    data.get("schema_version"),
                    self.schema_version,
                )
                if _migrate_entries(data, time.time()):
                    self._dirty = True
                    self._pending_updates += 1

            self._reindex()
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Cache file corrupted or unreadable, rebuilding: {e}")
            self._rebuild()

    def _serialize(self) -> str:
        """Render the current state. Caller must hold ``_lock``."""
        return json.dumps(self._data, indent=2)

    def _write(self, payload: str) -> None:
        """Atomically replace the cache file with ``payload``."""
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
                logger.warning(f"Failed to write cache file: {e}")
                return

        self._stats["disk_writes"] += 1

    def _save(self) -> None:
        """
        Write immediately. Caller must hold ``_lock``.

        Kept for callers that need the write to have happened by the time they
        return; the batched path goes through :meth:`_touch` instead.
        """
        payload = self._serialize()
        self._dirty = False
        self._pending_updates = 0
        self._write(payload)

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

    def _find_entry(self, file_path: str) -> Optional[dict]:
        return self._index.get(file_path)

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

            # Recording the read is what makes least-recently-used eviction
            # possible; without it there was no way to tell a summary that is
            # used on every run from one nobody has touched in months.
            entry["last_used_at"] = time.time()
            self._dirty = True

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

            now = time.time()
            existing = self._index.get(file_path)

            payload = {
                "file_path": file_path,
                "content_hash": self._compute_content_hash(content),
                "language": language,
                "summary": summary,
                "mtime": mtime,
                "created_at": (existing or {}).get("created_at", now),
                "last_used_at": now,
                "namespace": self.namespace,
            }

            if existing is None:
                self._data["entries"].append(payload)
                self._index[file_path] = payload
            else:
                # The index holds the same object that is in the entry list, so
                # updating in place keeps both in sync without an O(N) rebuild.
                existing.update(payload)

            self._stats["updates"] += 1
            self._touch()

    def _belongs_here(self, entry: dict) -> bool:
        """Whether ``entry`` was written for the repository being processed.

        An entry with no namespace predates the field and is treated as ours,
        which keeps the old behaviour for a cache written by an older version.
        """
        if self.namespace is None:
            return True
        entry_namespace = entry.get("namespace")
        return entry_namespace is None or entry_namespace == self.namespace

    def get_deleted_files(self, current_files: set) -> list:
        """
        Return cache entries for files that no longer exist in current_files.

        Only entries belonging to this cache's namespace are considered. The
        cache directory is the *current working directory*, not the repository,
        so running the tool from one place against two repositories put both in
        one file - and every entry from the other repository looked like a
        deleted file and was evicted. Two repositories analysed from the same
        directory used to erase each other's work on every run.
        """
        with self._lock:
            self._load()
            return [
                entry
                for entry in self._data.get("entries", [])
                if entry.get("file_path") not in current_files
                and self._belongs_here(entry)
            ]

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
                if entry.get("file_path") not in paths_to_remove
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

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def prune(
        self,
        max_entries: int | None = _UNSET,
        max_age_days: float | None = _UNSET,
        now: float | None = None,
    ) -> PruneReport:
        """
        Drop expired and surplus entries.

        Entries older than ``max_age_days`` (measured from when the summary was
        generated) go first, then the least recently used are dropped until at
        most ``max_entries`` remain. Passing ``None`` for either disables that
        bound; omitting them uses the values this cache was built with.

        The cache had no eviction at all: entries were removed only for files
        missing from the repository being processed, or wiped wholesale when
        the configuration changed. Nothing recorded when a summary was written,
        so age-based expiry was not even expressible.
        """
        if max_entries is _UNSET:
            max_entries = self.max_entries
        if max_age_days is _UNSET:
            max_age_days = self.max_age_days

        now = time.time() if now is None else now

        with self._lock:
            self._load()
            entries = self._data.get("entries", [])
            before = len(entries)

            kept = entries
            expired = 0

            if max_age_days is not None and max_age_days >= 0:
                cutoff = now - (max_age_days * SECONDS_PER_DAY)
                fresh = [e for e in kept if _created_at(e, now) >= cutoff]
                expired = len(kept) - len(fresh)
                kept = fresh

            evicted = 0
            if max_entries is not None and max_entries >= 0 and len(kept) > max_entries:
                # Least recently used first, so what a daily run touches stays.
                kept = sorted(kept, key=lambda e: _last_used_at(e, now), reverse=True)
                evicted = len(kept) - max_entries
                kept = kept[:max_entries]

            removed = before - len(kept)
            if not removed:
                return PruneReport(0, 0, before, before)

            self._data["entries"] = kept
            self._reindex()
            self._stats["evictions"] += removed
            self._dirty = True
            self._pending_updates += 1
            if self.autosave:
                self._save()

            return PruneReport(
                expired=expired,
                evicted=evicted,
                entries_before=before,
                entries_after=len(kept),
            )

    def describe(self) -> CacheSummary:
        """Counts, size and age of what is currently cached."""
        with self._lock:
            self._load()
            entries = self._data.get("entries", [])

            try:
                size_bytes = os.path.getsize(self.cache_file)
            except OSError:
                size_bytes = 0

            namespaces = Counter(
                str(entry.get("namespace") or "(unknown)") for entry in entries
            )
            created = [
                entry["created_at"]
                for entry in entries
                if isinstance(entry.get("created_at"), (int, float))
            ]

            return CacheSummary(
                cache_file=self.cache_file,
                exists=os.path.exists(self.cache_file),
                entries=len(entries),
                size_bytes=size_bytes,
                schema_version=str(self._data.get("schema_version", "")),
                namespaces=dict(namespaces),
                oldest_created_at=min(created) if created else None,
                newest_created_at=max(created) if created else None,
            )

    def clear(self) -> int:
        """Drop every entry. Returns how many were removed."""
        with self._lock:
            self._load()
            removed = len(self._data.get("entries", []))
            self._data["entries"] = []
            self._index = {}
            self._stats["removals"] += removed
            self._dirty = True
            self._pending_updates += 1
            if self.autosave:
                self._save()
            return removed
