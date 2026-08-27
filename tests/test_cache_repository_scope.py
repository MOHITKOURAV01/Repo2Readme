"""One cache file, several repositories.

The cache lives under the working directory, not the repository, so analyzing
two repositories from the same shell shares a single ``summaries.json``.
``get_deleted_files`` answers "which of my entries no longer exist?" and the CLI
removes what it returns - so an unscoped cache handed the second run every entry
the first run had produced.
"""

import json
import os

import pytest

from repo2readme.cache import CACHE_SCHEMA_VERSION, REPOSITORY_FIELD, SummaryCache
from repo2readme.loaders.source import repository_identity

CONFIG = {"provider": None, "model": None, "base_url": None}
PROMPT_HASH = "prompt-hash"


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "cache")


def build(cache_dir, repository=None, autosave=True):
    return SummaryCache(
        cache_dir=cache_dir,
        config=CONFIG,
        prompt_template_hash=PROMPT_HASH,
        autosave=autosave,
        repository=repository,
    )


def summarize(cache, path, content="x = 1"):
    cache.put(path, content, "python", {"file_path": path, "description": path}, 0.0)


class TestPruningStaysWithinOneRepository:
    def test_a_second_repository_does_not_evict_the_first(self, cache_dir):
        first = build(cache_dir, repository="/repos/alpha")
        summarize(first, "/repos/alpha/main.py")
        first.flush()

        second = build(cache_dir, repository="/repos/beta")
        summarize(second, "/repos/beta/app.py")
        stale = second.get_deleted_files({"/repos/beta/app.py"})

        assert stale == []

    def test_the_first_repository_is_still_warm_afterwards(self, cache_dir):
        first = build(cache_dir, repository="/repos/alpha")
        summarize(first, "/repos/alpha/main.py")
        first.flush()

        second = build(cache_dir, repository="/repos/beta")
        summarize(second, "/repos/beta/app.py")
        for entry in second.get_deleted_files({"/repos/beta/app.py"}):
            second.remove_entries([entry["file_path"]])
        second.flush()

        again = build(cache_dir, repository="/repos/alpha")
        assert again.get("/repos/alpha/main.py", "x = 1", "python") is not None

    def test_a_deleted_file_in_this_repository_is_still_reported(self, cache_dir):
        cache = build(cache_dir, repository="/repos/alpha")
        summarize(cache, "/repos/alpha/main.py")
        summarize(cache, "/repos/alpha/gone.py")

        stale = cache.get_deleted_files({"/repos/alpha/main.py"})

        assert [entry["file_path"] for entry in stale] == ["/repos/alpha/gone.py"]

    def test_remove_entries_leaves_another_repository_alone(self, cache_dir):
        first = build(cache_dir, repository="/repos/alpha")
        summarize(first, "/repos/shared/main.py")
        first.flush()

        second = build(cache_dir, repository="/repos/beta")
        second.remove_entries(["/repos/shared/main.py"])
        second.flush()

        again = build(cache_dir, repository="/repos/alpha")
        assert again.get("/repos/shared/main.py", "x = 1", "python") is not None

    def test_an_unscoped_cache_keeps_the_previous_behaviour(self, cache_dir):
        # Callers that keep one cache file per repository never had the bug and
        # must not have their sweep silently disabled.
        cache = build(cache_dir)
        summarize(cache, "/repos/alpha/main.py")

        stale = cache.get_deleted_files({"/repos/alpha/other.py"})

        assert [entry["file_path"] for entry in stale] == ["/repos/alpha/main.py"]


class TestLookupsAreScoped:
    def test_an_entry_from_another_repository_is_a_miss(self, cache_dir):
        first = build(cache_dir, repository="/repos/alpha")
        summarize(first, "/tmp/clone/main.py")
        first.flush()

        second = build(cache_dir, repository="https://example.com/beta")
        assert second.get("/tmp/clone/main.py", "x = 1", "python") is None

    def test_a_reused_path_is_claimed_by_one_repository_at_a_time(self, cache_dir):
        first = build(cache_dir, repository="/repos/alpha")
        summarize(first, "/tmp/clone/main.py", content="alpha")
        first.flush()

        second = build(cache_dir, repository="/repos/beta")
        summarize(second, "/tmp/clone/main.py", content="beta")
        second.flush()

        with open(os.path.join(cache_dir, "summaries.json")) as handle:
            stored = json.load(handle)

        entries = stored["entries"]
        assert len(entries) == 1
        assert entries[0][REPOSITORY_FIELD] == "/repos/beta"

    def test_entries_are_stamped_with_the_repository(self, cache_dir):
        cache = build(cache_dir, repository="https://example.com/acme/app")
        summarize(cache, "/tmp/clone/main.py")
        cache.flush()

        with open(os.path.join(cache_dir, "summaries.json")) as handle:
            stored = json.load(handle)

        assert stored["entries"][0][REPOSITORY_FIELD] == "https://example.com/acme/app"

    def test_an_unscoped_cache_writes_no_repository_field(self, cache_dir):
        cache = build(cache_dir)
        summarize(cache, "/repos/alpha/main.py")
        cache.flush()

        with open(os.path.join(cache_dir, "summaries.json")) as handle:
            stored = json.load(handle)

        assert REPOSITORY_FIELD not in stored["entries"][0]

    def test_entries_for_repository_returns_only_this_ones(self, cache_dir):
        first = build(cache_dir, repository="/repos/alpha")
        summarize(first, "/repos/alpha/main.py")
        first.flush()

        second = build(cache_dir, repository="/repos/beta")
        summarize(second, "/repos/beta/app.py")

        assert [e["file_path"] for e in second.entries_for_repository()] == [
            "/repos/beta/app.py"
        ]


class TestUpgradeFromAnUnscopedCacheFile:
    def test_an_old_cache_is_invalidated_once(self, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        legacy = {
            "schema_version": "1.0",
            "config_hash": build(cache_dir)._compute_config_hash(),
            "entries": [
                {
                    "file_path": "/repos/alpha/main.py",
                    "content_hash": "whatever",
                    "language": "python",
                    "summary": {"file_path": "/repos/alpha/main.py"},
                    "mtime": 0.0,
                }
            ],
        }
        with open(os.path.join(cache_dir, "summaries.json"), "w") as handle:
            json.dump(legacy, handle)

        cache = build(cache_dir, repository="/repos/alpha")

        assert cache.get("/repos/alpha/main.py", "x = 1", "python") is None
        assert cache.stats()["invalidations"] == 1

    def test_the_old_file_still_validates_structurally(self, cache_dir):
        # The rebuild must come from the schema version, which explains itself
        # in the log, not from a structural complaint about a missing field.
        from repo2readme.cache import _validate_cache_structure

        assert _validate_cache_structure(
            {
                "schema_version": "1.0",
                "config_hash": "x",
                "entries": [
                    {
                        "file_path": "a.py",
                        "content_hash": "h",
                        "language": "python",
                        "summary": {},
                        "mtime": 0.0,
                    }
                ],
            }
        )

    def test_the_schema_version_was_bumped(self):
        assert CACHE_SCHEMA_VERSION == "1.1"


class TestRepositoryIdentity:
    @pytest.mark.parametrize(
        "first, second",
        [
            ("https://github.com/acme/app.git", "https://github.com/acme/app"),
            ("https://github.com/acme/app/", "https://github.com/acme/app"),
            ("https://github.com/acme/app?tab=readme", "https://github.com/acme/app"),
            ("git@github.com:acme/app.git", "git@github.com:acme/app"),
        ],
    )
    def test_spellings_of_one_remote_agree(self, first, second):
        assert repository_identity(first) == repository_identity(second)

    def test_a_local_path_identifies_absolutely(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        assert repository_identity(".") == repository_identity(str(tmp_path))

    def test_a_file_url_and_its_path_agree(self, tmp_path):
        assert repository_identity(f"file://{tmp_path}") == repository_identity(
            str(tmp_path)
        )

    def test_different_repositories_do_not_collide(self):
        assert repository_identity(
            "https://github.com/acme/app"
        ) != repository_identity("https://github.com/acme/other")

    def test_an_empty_source_is_empty(self):
        assert repository_identity("") == ""

    def test_an_unclassifiable_source_is_returned_as_typed(self):
        # The loader is about to fail on this with a better message than a
        # cache key could give; identity must not raise first.
        assert repository_identity("wat://nope") == "wat://nope"
