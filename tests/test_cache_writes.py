"""Tests for how and when the summary cache reaches disk.

The cache used to rewrite the whole file once per stored summary, which made a
run of N files cost N full serializations of a file that grows to hold N
entries, and it applied configuration invalidation in memory without ever
persisting it.
"""

import json
import os
import threading

import pytest

from repo2readme.cache import DEFAULT_AUTOSAVE_EVERY, SummaryCache


@pytest.fixture
def config():
    return {"provider": "groq", "model": "openai/gpt-oss-120b", "base_url": None}


def _make_cache(tmp_path, config, **kwargs):
    return SummaryCache(
        cache_dir=str(tmp_path / "cache"),
        config=config,
        prompt_template_hash="prompt-hash",
        **kwargs,
    )


def _summary(path):
    return {"file_path": path, "description": "does a thing"}


def _read_cache_file(cache):
    with open(cache.cache_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Batched writes
# ---------------------------------------------------------------------------


def test_autosave_is_on_by_default(tmp_path, config):
    """A library caller keeps the durability it has today."""
    assert DEFAULT_AUTOSAVE_EVERY == 1

    cache = _make_cache(tmp_path, config)
    cache.put("a.py", "content", "python", _summary("a.py"), 1.0)

    assert os.path.exists(cache.cache_file)
    assert cache.stats()["disk_writes"] == 1


def test_disabled_autosave_writes_once_for_many_puts(tmp_path, config):
    cache = _make_cache(tmp_path, config, autosave=False)

    for i in range(50):
        cache.put(f"f{i}.py", f"content {i}", "python", _summary(f"f{i}.py"), 1.0)

    assert cache.stats()["disk_writes"] == 0
    assert not os.path.exists(cache.cache_file)

    assert cache.flush() is True
    assert cache.stats()["disk_writes"] == 1
    assert len(_read_cache_file(cache)["entries"]) == 50


def test_autosave_every_batches(tmp_path, config):
    cache = _make_cache(tmp_path, config, autosave_every=10)

    for i in range(30):
        cache.put(f"f{i}.py", "content", "python", _summary(f"f{i}.py"), 1.0)

    assert cache.stats()["disk_writes"] == 3


def test_flush_is_a_no_op_when_nothing_changed(tmp_path, config):
    cache = _make_cache(tmp_path, config, autosave=False)
    cache.put("a.py", "content", "python", _summary("a.py"), 1.0)

    assert cache.flush() is True
    assert cache.flush() is False
    assert cache.stats()["disk_writes"] == 1


def test_context_manager_flushes_on_exit(tmp_path, config):
    cache = _make_cache(tmp_path, config, autosave=False)

    with cache:
        cache.put("a.py", "content", "python", _summary("a.py"), 1.0)
        assert not os.path.exists(cache.cache_file)

    assert os.path.exists(cache.cache_file)
    assert len(_read_cache_file(cache)["entries"]) == 1


def test_context_manager_flushes_even_when_the_body_raises(tmp_path, config):
    cache = _make_cache(tmp_path, config, autosave=False)

    with pytest.raises(RuntimeError), cache:
        cache.put("a.py", "content", "python", _summary("a.py"), 1.0)
        raise RuntimeError("boom")

    assert len(_read_cache_file(cache)["entries"]) == 1


def _without_timestamps(data):
    """The cache file with its wall-clock fields removed.

    Entries record when they were written and last used, so two caches built a
    fraction of a second apart never match byte for byte. Everything batching
    could plausibly affect is still compared.
    """
    stripped = dict(data)
    stripped["entries"] = [
        {k: v for k, v in entry.items() if k not in ("created_at", "last_used_at")}
        for entry in data.get("entries", [])
    ]
    return stripped


def test_batched_and_autosaved_caches_produce_identical_files(tmp_path, config):
    eager = _make_cache(tmp_path / "eager", config)
    batched = _make_cache(tmp_path / "batched", config, autosave=False)

    for i in range(20):
        for cache in (eager, batched):
            cache.put(f"f{i}.py", f"content {i}", "python", _summary(f"f{i}.py"), 1.0)
    batched.flush()

    assert _without_timestamps(_read_cache_file(eager)) == _without_timestamps(
        _read_cache_file(batched)
    )


def test_batched_writes_survive_a_reload(tmp_path, config):
    cache = _make_cache(tmp_path, config, autosave=False)
    cache.put("a.py", "content", "python", _summary("a.py"), 1.0)
    cache.flush()

    reloaded = _make_cache(tmp_path, config)
    assert reloaded.get("a.py", "content", "python") == _summary("a.py")


# ---------------------------------------------------------------------------
# O(1) lookup and update
# ---------------------------------------------------------------------------


def test_updating_a_file_replaces_its_entry_in_place(tmp_path, config):
    cache = _make_cache(tmp_path, config, autosave=False)

    cache.put("a.py", "v1", "python", _summary("a.py"), 1.0)
    entry = cache._data["entries"][0]

    cache.put("a.py", "v2", "python", {"file_path": "a.py", "description": "v2"}, 2.0)

    assert len(cache._data["entries"]) == 1
    # Same object, mutated - no O(N) rebuild of the list.
    assert cache._data["entries"][0] is entry
    assert entry["content_hash"] == SummaryCache._compute_content_hash("v2")
    assert entry["mtime"] == 2.0
    assert cache.get("a.py", "v2", "python")["description"] == "v2"


def test_the_index_mirrors_the_entry_list(tmp_path, config):
    cache = _make_cache(tmp_path, config, autosave=False)

    for i in range(5):
        cache.put(f"f{i}.py", "content", "python", _summary(f"f{i}.py"), 1.0)

    assert set(cache._index) == {e["file_path"] for e in cache._data["entries"]}
    for path, entry in cache._index.items():
        assert any(e is entry for e in cache._data["entries"])
        assert entry["file_path"] == path


def test_the_index_is_rebuilt_after_removal(tmp_path, config):
    cache = _make_cache(tmp_path, config, autosave=False)

    for i in range(5):
        cache.put(f"f{i}.py", "content", "python", _summary(f"f{i}.py"), 1.0)
    cache.remove_entries(["f1.py", "f3.py"])

    assert set(cache._index) == {"f0.py", "f2.py", "f4.py"}
    assert cache.get("f1.py", "content", "python") is None
    assert cache.get("f0.py", "content", "python") is not None


def test_the_index_is_built_when_loading_from_disk(tmp_path, config):
    first = _make_cache(tmp_path, config, autosave=False)
    for i in range(3):
        first.put(f"f{i}.py", "content", "python", _summary(f"f{i}.py"), 1.0)
    first.flush()

    second = _make_cache(tmp_path, config)
    assert second.get("f2.py", "content", "python") is not None
    assert set(second._index) == {"f0.py", "f1.py", "f2.py"}


def test_removing_nothing_does_not_write(tmp_path, config):
    cache = _make_cache(tmp_path, config)
    cache.put("a.py", "content", "python", _summary("a.py"), 1.0)
    writes_before = cache.stats()["disk_writes"]

    cache.remove_entries([])
    cache.remove_entries(["not-in-cache.py"])

    assert cache.stats()["disk_writes"] == writes_before


# ---------------------------------------------------------------------------
# Invalidation now reaches disk
# ---------------------------------------------------------------------------


def test_configuration_change_is_persisted_without_a_later_put(tmp_path, config):
    """The reset used to be applied in memory and dropped, so the stale entries
    and stale config hash survived on disk and were re-detected every run."""
    first = _make_cache(tmp_path, config)
    first.put("a.py", "content", "python", _summary("a.py"), 1.0)

    changed = dict(config, model="a-different-model")
    second = _make_cache(tmp_path, changed)
    assert second.get("a.py", "content", "python") is None

    on_disk = _read_cache_file(second)
    assert on_disk["entries"] == []
    assert on_disk["config_hash"] == second._compute_config_hash()

    # A third run sees a consistent file and does not invalidate again.
    third = _make_cache(tmp_path, changed)
    third.get("a.py", "content", "python")
    assert third.stats()["invalidations"] == 0


def test_schema_version_change_is_persisted(tmp_path, config):
    cache = _make_cache(tmp_path, config)
    cache.put("a.py", "content", "python", _summary("a.py"), 1.0)

    stale = _make_cache(tmp_path, config)
    stale.schema_version = "99.0"
    assert stale.get("a.py", "content", "python") is None

    on_disk = _read_cache_file(stale)
    assert on_disk["schema_version"] == "99.0"
    assert on_disk["entries"] == []


def test_invalidation_is_flushed_when_autosave_is_off(tmp_path, config):
    first = _make_cache(tmp_path, config)
    first.put("a.py", "content", "python", _summary("a.py"), 1.0)

    second = _make_cache(tmp_path, dict(config, provider="openai"), autosave=False)
    assert second.get("a.py", "content", "python") is None
    assert _read_cache_file(second)["entries"] != []  # not written yet

    second.flush()
    assert _read_cache_file(second)["entries"] == []


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_stats_counts_hits_misses_and_writes(tmp_path, config):
    cache = _make_cache(tmp_path, config, autosave=False)

    cache.get("a.py", "content", "python")  # miss
    cache.put("a.py", "content", "python", _summary("a.py"), 1.0)
    cache.get("a.py", "content", "python")  # hit
    cache.get("a.py", "changed", "python")  # miss: content hash
    cache.get("a.py", "content", "ruby")  # miss: language
    cache.flush()

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 3
    assert stats["updates"] == 1
    assert stats["disk_writes"] == 1


def test_stats_returns_a_copy(tmp_path, config):
    cache = _make_cache(tmp_path, config)
    stats = cache.stats()
    stats["hits"] = 999
    assert cache.stats()["hits"] == 0


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_puts_with_batching_lose_nothing(tmp_path, config):
    cache = _make_cache(tmp_path, config, autosave=False)

    def worker(start):
        for i in range(start, start + 25):
            cache.put(f"f{i}.py", f"content {i}", "python", _summary(f"f{i}.py"), 1.0)

    threads = [threading.Thread(target=worker, args=(n * 25,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    cache.flush()

    entries = _read_cache_file(cache)["entries"]
    assert len(entries) == 100
    assert {e["file_path"] for e in entries} == {f"f{i}.py" for i in range(100)}


def test_concurrent_flushes_leave_a_valid_file(tmp_path, config):
    cache = _make_cache(tmp_path, config, autosave=False)
    for i in range(40):
        cache.put(f"f{i}.py", "content", "python", _summary(f"f{i}.py"), 1.0)

    def worker():
        for _ in range(5):
            cache.put("hot.py", "content", "python", _summary("hot.py"), 1.0)
            cache.flush()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    data = _read_cache_file(cache)
    assert len(data["entries"]) == 41


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def test_write_count_does_not_grow_with_the_number_of_files(tmp_path, config):
    """The point of the change: writes are O(1) in the file count, not O(N)."""
    small = _make_cache(tmp_path / "small", config, autosave=False)
    large = _make_cache(tmp_path / "large", config, autosave=False)

    for i in range(10):
        small.put(f"f{i}.py", "content", "python", _summary(f"f{i}.py"), 1.0)
    for i in range(400):
        large.put(f"f{i}.py", "content", "python", _summary(f"f{i}.py"), 1.0)

    small.flush()
    large.flush()

    assert small.stats()["disk_writes"] == large.stats()["disk_writes"] == 1
