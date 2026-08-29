"""A damaged cache entry costs one file, not the repository.

``_validate_cache_structure`` walked every entry and returned ``False`` for the
whole file at the first one it disliked; ``_load`` then called ``_rebuild()``,
which replaced the entry list with an empty one. One entry missing one field
discarded every other summary in the file - each of which had been paid for
with an API call - and reported it as "rebuilding".

Entries are independent. Each is separately keyed and separately re-checked at
lookup time, so a well-formed entry sitting next to a broken one is not itself
suspect. Only the file's own structure - a root object, a schema version, a
configuration hash, a list to hold the entries - is genuinely all-or-nothing.
"""

import json
import logging
import os

import pytest

from repo2readme.cache import (
    CACHE_SCHEMA_VERSION,
    MAX_LOGGED_DISCARDS,
    SummaryCache,
    _validate_cache_shell,
    _validate_cache_structure,
    entry_problem,
    partition_entries,
)

CONFIG = {"provider": "groq", "model": "openai/gpt-oss-120b", "base_url": None}
PROMPT_HASH = "prompt-hash"


def _entry(path, *, content_hash="h", language="python", mtime=1.0):
    return {
        "file_path": path,
        "content_hash": content_hash,
        "language": language,
        "summary": {"description": f"summary of {path}"},
        "mtime": mtime,
    }


def _cache(tmp_path, **kwargs):
    return SummaryCache(str(tmp_path), CONFIG, PROMPT_HASH, **kwargs)


def _seed(tmp_path, count):
    """Write ``count`` real entries through the cache and return their paths."""
    cache = _cache(tmp_path)
    paths = []
    for i in range(count):
        path = f"src/f{i}.py"
        paths.append(path)
        cache.put(path, f"content {i}", "python", {"description": f"s{i}"}, float(i))
    cache.flush()
    return paths


def _read(tmp_path):
    return json.loads((tmp_path / "summaries.json").read_text())


def _write(tmp_path, data):
    os.makedirs(tmp_path, exist_ok=True)
    (tmp_path / "summaries.json").write_text(json.dumps(data, indent=2))


class TestEntryProblem:
    def test_a_complete_entry_has_no_problem(self):
        assert entry_problem(_entry("a.py")) is None

    @pytest.mark.parametrize(
        "field", ["file_path", "content_hash", "language", "summary", "mtime"]
    )
    def test_a_missing_field_is_named(self, field):
        entry = _entry("a.py")
        del entry[field]
        problem = entry_problem(entry)
        assert problem is not None
        assert field in problem

    def test_several_missing_fields_are_all_named(self):
        entry = {"file_path": "a.py"}
        problem = entry_problem(entry)
        for field in ("content_hash", "language", "summary", "mtime"):
            assert field in problem

    @pytest.mark.parametrize("value", ["not a dict", 42, None, ["a"]])
    def test_a_non_dict_entry_reports_its_type(self, value):
        problem = entry_problem(value)
        assert problem is not None
        assert type(value).__name__ in problem

    @pytest.mark.parametrize("path", ["", None, 3, ["a.py"]])
    def test_an_unusable_file_path_is_rejected(self, path):
        # The path is the index key. An entry that cannot be keyed can never be
        # looked up, so keeping it would only corrupt the index.
        entry = _entry("a.py")
        entry["file_path"] = path
        assert entry_problem(entry) is not None

    def test_extra_fields_are_allowed(self):
        # A field added by a future version must not make the entry unusable,
        # which is the shape of upgrade this whole change is about.
        entry = _entry("a.py")
        entry["token_count"] = 91
        assert entry_problem(entry) is None


class TestPartitionEntries:
    def test_all_good_entries_are_kept(self):
        entries = [_entry(f"f{i}.py") for i in range(5)]
        usable, discarded = partition_entries(entries)
        assert usable == entries
        assert discarded == []

    def test_one_bad_entry_costs_only_itself(self):
        entries = [_entry(f"f{i}.py") for i in range(5)]
        del entries[2]["mtime"]
        usable, discarded = partition_entries(entries)
        assert len(usable) == 4
        assert [e["file_path"] for e in usable] == [
            "f0.py", "f1.py", "f3.py", "f4.py"
        ]
        assert len(discarded) == 1

    def test_the_discarded_index_is_the_position_in_the_file(self):
        entries = [_entry("a.py"), "junk", _entry("c.py"), {"file_path": "d.py"}]
        _, discarded = partition_entries(entries)
        assert [index for index, _ in discarded] == [1, 3]

    def test_the_discarded_reason_is_carried_out(self):
        _, discarded = partition_entries(["junk"])
        assert "not a dictionary" in discarded[0][1]

    def test_an_empty_list_partitions_cleanly(self):
        assert partition_entries([]) == ([], [])

    def test_every_entry_bad_yields_nothing_usable(self):
        usable, discarded = partition_entries(["a", "b", "c"])
        assert usable == []
        assert len(discarded) == 3

    def test_the_usable_entries_are_the_same_objects(self):
        # _reindex points the index at the entry objects in the list; a copy
        # here would silently break in-place updates in put().
        entries = [_entry("a.py")]
        usable, _ = partition_entries(entries)
        assert usable[0] is entries[0]


class TestValidateShell:
    def test_a_well_formed_shell_passes_whatever_the_entries_are(self):
        data = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "config_hash": "abc",
            "entries": ["junk", {"file_path": "a.py"}],
        }
        assert _validate_cache_shell(data) is True

    @pytest.mark.parametrize("missing", ["schema_version", "config_hash"])
    def test_a_missing_top_level_field_fails(self, missing):
        data = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "config_hash": "abc",
            "entries": [],
        }
        del data[missing]
        assert _validate_cache_shell(data) is False

    @pytest.mark.parametrize("data", ["a string", 42, None, ["a"]])
    def test_a_non_dict_root_fails(self, data):
        assert _validate_cache_shell(data) is False

    @pytest.mark.parametrize("entries", ["not a list", 42, {"a": 1}, None])
    def test_entries_that_are_not_a_list_fail(self, entries):
        data = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "config_hash": "abc",
            "entries": entries,
        }
        assert _validate_cache_shell(data) is False

    def test_the_config_hash_is_why_the_shell_is_all_or_nothing(self):
        # Without it there is no way to ask whether any entry was produced by
        # the settings in use now, so no entry can be trusted individually.
        data = {"schema_version": CACHE_SCHEMA_VERSION, "entries": [_entry("a.py")]}
        assert _validate_cache_shell(data) is False


class TestValidateStructureStillAnswersIsThisClean:
    """The old predicate keeps its old meaning; it just no longer decides."""

    def test_a_clean_file_is_still_valid(self):
        data = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "config_hash": "abc",
            "entries": [_entry("a.py")],
        }
        assert _validate_cache_structure(data) is True

    def test_a_file_with_one_bad_entry_is_still_not_clean(self):
        data = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "config_hash": "abc",
            "entries": [_entry("a.py"), "junk"],
        }
        assert _validate_cache_structure(data) is False


class TestLoadKeepsWhatItCan:
    def test_one_damaged_entry_does_not_discard_the_rest(self, tmp_path):
        paths = _seed(tmp_path, 200)
        raw = _read(tmp_path)
        del raw["entries"][57]["mtime"]
        _write(tmp_path, raw)

        cache = _cache(tmp_path)
        hits = sum(
            1
            for i, path in enumerate(paths)
            if cache.get(path, f"content {i}", "python") is not None
        )

        assert hits == 199
        assert cache.stats()["discarded_entries"] == 1

    def test_several_kinds_of_damage_at_once(self, tmp_path):
        paths = _seed(tmp_path, 20)
        raw = _read(tmp_path)
        raw["entries"][3] = "not a dict"
        del raw["entries"][7]["summary"]
        raw["entries"][11]["file_path"] = ""
        _write(tmp_path, raw)

        cache = _cache(tmp_path)
        hits = sum(
            1
            for i, path in enumerate(paths)
            if cache.get(path, f"content {i}", "python") is not None
        )

        assert hits == 17
        assert cache.stats()["discarded_entries"] == 3

    def test_the_damaged_entry_itself_is_not_served(self, tmp_path):
        _seed(tmp_path, 5)
        raw = _read(tmp_path)
        del raw["entries"][2]["content_hash"]
        damaged_path = raw["entries"][2]["file_path"]
        _write(tmp_path, raw)

        cache = _cache(tmp_path)
        assert cache.get(damaged_path, "content 2", "python") is None

    def test_a_broken_shell_still_rebuilds(self, tmp_path):
        _seed(tmp_path, 5)
        raw = _read(tmp_path)
        del raw["config_hash"]
        _write(tmp_path, raw)

        cache = _cache(tmp_path)
        assert cache.get("src/f0.py", "content 0", "python") is None
        assert cache.stats()["discarded_entries"] == 0

    def test_every_entry_damaged_behaves_like_an_empty_cache(self, tmp_path):
        _seed(tmp_path, 4)
        raw = _read(tmp_path)
        raw["entries"] = ["junk"] * 4
        _write(tmp_path, raw)

        cache = _cache(tmp_path)
        assert cache.get("src/f0.py", "content 0", "python") is None
        cache.put("src/new.py", "c", "python", {"description": "d"}, 1.0)
        assert cache.get("src/new.py", "c", "python") is not None

    def test_an_undamaged_cache_discards_nothing(self, tmp_path):
        paths = _seed(tmp_path, 10)
        cache = _cache(tmp_path)
        for i, path in enumerate(paths):
            assert cache.get(path, f"content {i}", "python") is not None
        assert cache.stats()["discarded_entries"] == 0


class TestTheFileHealsItself:
    def test_the_cleaned_entries_are_written_back(self, tmp_path):
        _seed(tmp_path, 10)
        raw = _read(tmp_path)
        del raw["entries"][4]["mtime"]
        _write(tmp_path, raw)

        cache = _cache(tmp_path)
        cache.get("src/f0.py", "content 0", "python")
        cache.flush()

        assert len(_read(tmp_path)["entries"]) == 9

    def test_the_same_damage_is_not_reported_twice(self, tmp_path):
        _seed(tmp_path, 10)
        raw = _read(tmp_path)
        del raw["entries"][4]["mtime"]
        _write(tmp_path, raw)

        first = _cache(tmp_path)
        first.get("src/f0.py", "content 0", "python")
        first.flush()

        second = _cache(tmp_path)
        second.get("src/f0.py", "content 0", "python")
        assert second.stats()["discarded_entries"] == 0

    def test_the_surviving_summaries_are_still_usable_after_the_rewrite(
        self, tmp_path
    ):
        paths = _seed(tmp_path, 10)
        raw = _read(tmp_path)
        del raw["entries"][4]["mtime"]
        _write(tmp_path, raw)

        first = _cache(tmp_path)
        first.get("src/f0.py", "content 0", "python")
        first.flush()

        second = _cache(tmp_path)
        hits = sum(
            1
            for i, path in enumerate(paths)
            if second.get(path, f"content {i}", "python") is not None
        )
        assert hits == 9

    def test_loading_a_clean_cache_leaves_it_alone(self, tmp_path):
        _seed(tmp_path, 5)
        before = (tmp_path / "summaries.json").read_text()

        cache = _cache(tmp_path)
        cache.get("src/f0.py", "content 0", "python")
        assert cache.flush() is False  # nothing was dirty

        assert (tmp_path / "summaries.json").read_text() == before


class TestReporting:
    def test_the_discarded_entries_are_named(self, tmp_path, caplog):
        _seed(tmp_path, 5)
        raw = _read(tmp_path)
        del raw["entries"][2]["mtime"]
        _write(tmp_path, raw)

        with caplog.at_level(logging.WARNING, logger="repo2readme.cache"):
            _cache(tmp_path).get("src/f0.py", "content 0", "python")

        assert "Discarding cache entry 2" in caplog.text
        assert "mtime" in caplog.text

    def test_the_summary_says_how_many_were_kept(self, tmp_path, caplog):
        _seed(tmp_path, 5)
        raw = _read(tmp_path)
        del raw["entries"][2]["mtime"]
        _write(tmp_path, raw)

        with caplog.at_level(logging.WARNING, logger="repo2readme.cache"):
            _cache(tmp_path).get("src/f0.py", "content 0", "python")

        assert "Discarded 1 unusable cache entry, kept 4" in caplog.text

    def test_wholesale_damage_does_not_log_one_line_per_entry(
        self, tmp_path, caplog
    ):
        _seed(tmp_path, 40)
        raw = _read(tmp_path)
        raw["entries"] = ["junk"] * 40
        _write(tmp_path, raw)

        with caplog.at_level(logging.WARNING, logger="repo2readme.cache"):
            _cache(tmp_path).get("src/f0.py", "content 0", "python")

        named = [r for r in caplog.records if "Discarding cache entry" in r.message]
        assert len(named) == MAX_LOGGED_DISCARDS
        assert "and 35 further unusable cache entries" in caplog.text

    def test_the_plural_reads_correctly(self, tmp_path, caplog):
        _seed(tmp_path, 5)
        raw = _read(tmp_path)
        del raw["entries"][1]["mtime"]
        del raw["entries"][2]["mtime"]
        _write(tmp_path, raw)

        with caplog.at_level(logging.WARNING, logger="repo2readme.cache"):
            _cache(tmp_path).get("src/f0.py", "content 0", "python")

        assert "Discarded 2 unusable cache entries, kept 3" in caplog.text


class TestConfigurationIsReconciledAtLoad:
    """A ``put`` before the first ``get`` must not be thrown away.

    The configuration hash used to be checked only in ``get``. A ``put`` that
    ran first appended to a list still carrying another configuration's hash,
    and the first ``get`` afterwards invalidated the lot - including the entry
    just written. ``_rebuild`` hid this whenever the file happened to be
    rejected outright, which is how it survived.
    """

    def _write_foreign(self, tmp_path):
        _write(
            tmp_path,
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "config_hash": "produced-by-another-provider",
                "entries": [_entry("src/old.py")],
            },
        )

    def test_a_put_before_any_get_survives(self, tmp_path):
        self._write_foreign(tmp_path)

        cache = _cache(tmp_path)
        cache.put("src/new.py", "c", "python", {"description": "d"}, 1.0)

        assert cache.get("src/new.py", "c", "python") == {"description": "d"}

    def test_the_foreign_entries_are_still_dropped(self, tmp_path):
        self._write_foreign(tmp_path)

        cache = _cache(tmp_path)
        assert cache.get("src/old.py", "content", "python") is None

    def test_the_current_hash_is_adopted_on_disk(self, tmp_path):
        self._write_foreign(tmp_path)

        cache = _cache(tmp_path)
        cache.put("src/new.py", "c", "python", {"description": "d"}, 1.0)
        cache.flush()

        assert _read(tmp_path)["config_hash"] != "produced-by-another-provider"

    def test_a_stale_schema_version_is_reconciled_the_same_way(self, tmp_path):
        _write(
            tmp_path,
            {
                "schema_version": "0.1",
                "config_hash": SummaryCache(
                    str(tmp_path), CONFIG, PROMPT_HASH
                )._compute_config_hash(),
                "entries": [_entry("src/old.py")],
            },
        )

        cache = _cache(tmp_path)
        cache.put("src/new.py", "c", "python", {"description": "d"}, 1.0)

        assert cache.get("src/new.py", "c", "python") is not None
        assert cache.get("src/old.py", "content", "python") is None

    def test_a_matching_configuration_is_left_alone(self, tmp_path):
        paths = _seed(tmp_path, 3)
        cache = _cache(tmp_path)
        cache.put("src/new.py", "c", "python", {"description": "d"}, 1.0)

        for i, path in enumerate(paths):
            assert cache.get(path, f"content {i}", "python") is not None
        assert cache.get("src/new.py", "c", "python") is not None


class TestStats:
    def test_the_counter_starts_at_zero(self, tmp_path):
        assert _cache(tmp_path).stats()["discarded_entries"] == 0

    def test_the_counter_records_what_was_dropped(self, tmp_path):
        _seed(tmp_path, 6)
        raw = _read(tmp_path)
        raw["entries"][0] = "junk"
        raw["entries"][5] = "junk"
        _write(tmp_path, raw)

        cache = _cache(tmp_path)
        cache.get("src/f1.py", "content 1", "python")
        assert cache.stats()["discarded_entries"] == 2

    def test_hits_and_misses_are_unaffected(self, tmp_path):
        _seed(tmp_path, 4)
        raw = _read(tmp_path)
        raw["entries"][0] = "junk"
        _write(tmp_path, raw)

        cache = _cache(tmp_path)
        assert cache.get("src/f1.py", "content 1", "python") is not None
        assert cache.get("src/f0.py", "content 0", "python") is None

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
