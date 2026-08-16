"""Tests for the cache's bounds, its per-repository scoping, and the CLI.

From the issue: the cache had no eviction of any kind, entries carried no
timestamp so age-based expiry was not even expressible, the cache directory is
the *current working directory* rather than the repository (so two repositories
analysed from one place shared a file and evicted each other on every run), and
there was no command to look at the cache or clear it.
"""

import importlib
import json
import time

import pytest
from click.testing import CliRunner

from repo2readme.cache import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_MAX_ENTRIES,
    SECONDS_PER_DAY,
    SummaryCache,
)
from repo2readme.services.cache_admin import (
    build_info_lines,
    cache_info,
    clear_cache,
    default_cache_dir,
    format_size,
    format_timestamp,
    prune_cache,
)

cli_main = importlib.import_module("repo2readme.cli.main")

CONFIG = {"provider": "groq", "model": "m", "base_url": None}


def make_cache(tmp_path, namespace=None, **kwargs):
    return SummaryCache(
        cache_dir=str(tmp_path),
        config=CONFIG,
        prompt_template_hash="hash",
        autosave=False,
        namespace=namespace,
        **kwargs,
    )


def summary(path):
    return {"file_path": path, "description": "a summary"}


def fill(cache, count, prefix="f"):
    for i in range(count):
        cache.put(f"{prefix}{i}.py", f"content {i}", "python", summary(f"{prefix}{i}.py"), 1.0)


def read_file(cache):
    with open(cache.cache_file, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def test_entries_record_when_they_were_written(tmp_path):
    cache = make_cache(tmp_path)
    before = time.time()

    cache.put("a.py", "content", "python", summary("a.py"), 1.0)
    cache.flush()

    entry = read_file(cache)["entries"][0]
    assert before <= entry["created_at"] <= time.time()
    assert entry["last_used_at"] >= entry["created_at"]


def test_a_hit_records_the_use(tmp_path):
    cache = make_cache(tmp_path)
    cache.put("a.py", "content", "python", summary("a.py"), 1.0)
    assert cache.describe().entries == 1

    time.sleep(0.01)
    assert cache.get("a.py", "content", "python") is not None
    cache.flush()

    entry = read_file(cache)["entries"][0]
    assert entry["last_used_at"] > entry["created_at"]


def test_rewriting_an_entry_keeps_its_creation_time(tmp_path):
    cache = make_cache(tmp_path)
    cache.put("a.py", "one", "python", summary("a.py"), 1.0)
    cache.flush()
    created = read_file(cache)["entries"][0]["created_at"]

    time.sleep(0.01)
    cache.put("a.py", "two", "python", summary("a.py"), 2.0)
    cache.flush()

    entry = read_file(cache)["entries"][0]
    assert entry["created_at"] == created
    assert entry["last_used_at"] > created


# ---------------------------------------------------------------------------
# Migration from schema 1.0
# ---------------------------------------------------------------------------


def test_a_schema_1_0_cache_is_migrated_not_discarded(tmp_path):
    cache_file = tmp_path / "summaries.json"
    legacy = make_cache(tmp_path)
    cache_file.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "config_hash": legacy._compute_config_hash(),
                "entries": [
                    {
                        "file_path": "a.py",
                        "content_hash": SummaryCache._compute_content_hash("content"),
                        "language": "python",
                        "summary": summary("a.py"),
                        "mtime": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cache = make_cache(tmp_path)

    # The entry survives: it is perfectly usable, it just lacked bookkeeping.
    assert cache.get("a.py", "content", "python") == summary("a.py")

    cache.flush()
    data = read_file(cache)
    assert data["schema_version"] == CACHE_SCHEMA_VERSION
    assert "created_at" in data["entries"][0]


def test_a_migrated_entry_ages_from_now(tmp_path):
    cache_file = tmp_path / "summaries.json"
    template = make_cache(tmp_path)
    cache_file.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "config_hash": template._compute_config_hash(),
                "entries": [
                    {
                        "file_path": "a.py",
                        "content_hash": "x",
                        "language": "python",
                        "summary": summary("a.py"),
                        "mtime": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cache = make_cache(tmp_path)
    report = cache.prune(max_age_days=1)

    assert report.expired == 0


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def test_nothing_is_pruned_from_a_small_fresh_cache(tmp_path):
    cache = make_cache(tmp_path)
    fill(cache, 10)

    report = cache.prune()

    assert report.removed == 0
    assert report.entries_after == 10


def test_the_entry_count_is_capped(tmp_path):
    cache = make_cache(tmp_path)
    fill(cache, 50)

    report = cache.prune(max_entries=10, max_age_days=None)

    assert report.evicted == 40
    assert report.entries_after == 10
    assert cache.describe().entries == 10


def test_the_least_recently_used_are_dropped_first(tmp_path):
    cache = make_cache(tmp_path)
    fill(cache, 5)

    # Touch two of them, so they are the most recently used.
    time.sleep(0.01)
    cache.get("f0.py", "content 0", "python")
    cache.get("f4.py", "content 4", "python")

    cache.prune(max_entries=2, max_age_days=None)
    cache.flush()

    kept = {entry["file_path"] for entry in read_file(cache)["entries"]}
    assert kept == {"f0.py", "f4.py"}


def test_old_entries_expire(tmp_path):
    cache = make_cache(tmp_path)
    fill(cache, 3)
    cache.flush()

    # Age one of them by a year.
    data = read_file(cache)
    data["entries"][0]["created_at"] -= 365 * SECONDS_PER_DAY
    (tmp_path / "summaries.json").write_text(json.dumps(data), encoding="utf-8")

    fresh = make_cache(tmp_path)
    report = fresh.prune(max_age_days=90, max_entries=None)

    assert report.expired == 1
    assert report.entries_after == 2


def test_bounds_can_be_disabled(tmp_path):
    cache = make_cache(tmp_path)
    fill(cache, 30)

    report = cache.prune(max_entries=None, max_age_days=None)

    assert report.removed == 0
    assert report.entries_after == 30


def test_the_configured_bounds_are_used_by_default(tmp_path):
    cache = make_cache(tmp_path, max_entries=5, max_age_days=None)
    fill(cache, 20)

    report = cache.prune()

    assert report.entries_after == 5


def test_pruning_is_counted(tmp_path):
    cache = make_cache(tmp_path)
    fill(cache, 20)
    cache.prune(max_entries=5, max_age_days=None)

    assert cache.stats()["evictions"] == 15


def test_the_defaults_are_sane():
    assert DEFAULT_MAX_ENTRIES > 0
    assert DEFAULT_MAX_AGE_DAYS > 0


def test_clear_drops_everything(tmp_path):
    cache = make_cache(tmp_path)
    fill(cache, 7)

    assert cache.clear() == 7
    assert cache.describe().entries == 0


# ---------------------------------------------------------------------------
# Namespaces - two repositories sharing one cache file
# ---------------------------------------------------------------------------


def test_another_repository_is_not_reported_as_deleted(tmp_path):
    project_a = make_cache(tmp_path, namespace="/work/project-a")
    project_a.put("/work/project-a/app.py", "a", "python", summary("app.py"), 1.0)
    project_a.flush()

    project_b = make_cache(tmp_path, namespace="/work/project-b")
    project_b.put("/work/project-b/main.py", "b", "python", summary("main.py"), 1.0)

    stale = project_b.get_deleted_files({"/work/project-b/main.py"})

    # project-a's entry belongs to another repository; it used to be listed
    # here and then evicted, so the two repositories erased each other.
    assert stale == []


def test_a_repository_still_prunes_its_own_deleted_files(tmp_path):
    cache = make_cache(tmp_path, namespace="/work/project-a")
    cache.put("/work/project-a/gone.py", "x", "python", summary("gone.py"), 1.0)
    cache.put("/work/project-a/kept.py", "y", "python", summary("kept.py"), 1.0)

    stale = cache.get_deleted_files({"/work/project-a/kept.py"})

    assert [entry["file_path"] for entry in stale] == ["/work/project-a/gone.py"]


def test_two_repositories_keep_their_hits(tmp_path):
    a = make_cache(tmp_path, namespace="/work/a")
    a.put("/work/a/app.py", "a", "python", summary("app.py"), 1.0)
    a.flush()

    b = make_cache(tmp_path, namespace="/work/b")
    b.put("/work/b/app.py", "b", "python", summary("app.py"), 1.0)
    b.remove_entries([e["file_path"] for e in b.get_deleted_files({"/work/b/app.py"})])
    b.flush()

    again = make_cache(tmp_path, namespace="/work/a")
    assert again.get("/work/a/app.py", "a", "python") is not None


def test_entries_without_a_namespace_are_still_treated_as_ours(tmp_path):
    legacy = make_cache(tmp_path)  # namespace=None, as an older version wrote
    legacy.put("app.py", "x", "python", summary("app.py"), 1.0)
    legacy.flush()

    cache = make_cache(tmp_path, namespace="/work/a")

    assert [e["file_path"] for e in cache.get_deleted_files(set())] == ["app.py"]


def test_the_namespace_is_recorded(tmp_path):
    cache = make_cache(tmp_path, namespace="/work/a")
    cache.put("app.py", "x", "python", summary("app.py"), 1.0)
    cache.flush()

    assert read_file(cache)["entries"][0]["namespace"] == "/work/a"


def test_the_cli_namespace_is_stable_for_a_local_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert cli_main.cache_namespace(None, ".") == cli_main.cache_namespace(
        None, str(tmp_path)
    )


def test_the_cli_namespace_of_a_url_is_the_url():
    url = "https://github.com/acme/app.git"

    assert cli_main.cache_namespace(url, None) == url


# ---------------------------------------------------------------------------
# describe / cache_admin
# ---------------------------------------------------------------------------


def test_describe_reports_the_contents(tmp_path):
    cache = make_cache(tmp_path, namespace="/work/a")
    fill(cache, 4)
    cache.flush()

    described = cache.describe()

    assert described.entries == 4
    assert described.exists is True
    assert described.size_bytes > 0
    assert described.schema_version == CACHE_SCHEMA_VERSION
    assert described.namespaces == {"/work/a": 4}
    assert described.repositories == 1
    assert described.oldest_created_at <= described.newest_created_at


def test_describe_of_a_missing_cache(tmp_path):
    described = cache_info(str(tmp_path / "nothing-here"))

    assert described.exists is False
    assert described.entries == 0


def test_describe_counts_repositories_separately(tmp_path):
    a = make_cache(tmp_path, namespace="/work/a")
    fill(a, 2, prefix="a")
    a.flush()
    b = make_cache(tmp_path, namespace="/work/b")
    fill(b, 3, prefix="b")
    b.flush()

    described = cache_info(str(tmp_path))

    assert described.repositories == 2
    assert described.namespaces == {"/work/a": 2, "/work/b": 3}


def test_prune_cache_writes_the_result(tmp_path):
    cache = make_cache(tmp_path)
    fill(cache, 20)
    cache.flush()

    report = prune_cache(str(tmp_path), max_entries=5, max_age_days=None)

    assert report.entries_after == 5
    assert cache_info(str(tmp_path)).entries == 5


def test_clear_cache_empties_the_file(tmp_path):
    cache = make_cache(tmp_path)
    fill(cache, 6)
    cache.flush()

    assert clear_cache(str(tmp_path)) == 6
    assert cache_info(str(tmp_path)).entries == 0


def test_clear_cache_can_remove_the_directory(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = make_cache(cache_dir)
    fill(cache, 3)
    cache.flush()

    assert clear_cache(str(cache_dir), remove_directory=True) == 3
    assert not cache_dir.exists()


def test_clear_cache_of_a_missing_directory_is_harmless(tmp_path):
    assert clear_cache(str(tmp_path / "nope")) == 0


def test_inspecting_the_cache_does_not_invalidate_it(tmp_path):
    cache = make_cache(tmp_path)
    fill(cache, 3)
    cache.flush()

    cache_info(str(tmp_path))

    assert cache_info(str(tmp_path)).entries == 3


def test_the_default_cache_dir_is_under_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert default_cache_dir().startswith(str(tmp_path))
    assert default_cache_dir().endswith("summaries.json") is False


@pytest.mark.parametrize(
    "size, expected",
    [(0, "0 B"), (512, "512 B"), (2048, "2.0 KB"), (5 * 1024 * 1024, "5.0 MB")],
)
def test_format_size(size, expected):
    assert format_size(size) == expected


def test_format_timestamp_handles_nothing():
    assert format_timestamp(None) == "-"
    assert format_timestamp(0) == "-"
    assert len(format_timestamp(time.time())) == len("2024-01-01 00:00")


def test_info_lines_describe_an_empty_cache(tmp_path):
    lines = build_info_lines(cache_info(str(tmp_path / "nope")))

    assert any("No cache at" in line for line in lines)


def test_info_lines_list_repositories(tmp_path):
    cache = make_cache(tmp_path, namespace="/work/a")
    fill(cache, 2)
    cache.flush()

    text = "\n".join(build_info_lines(cache_info(str(tmp_path))))

    assert "/work/a" in text
    assert "Entries" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCacheCommands:
    def test_info_on_an_empty_directory(self, tmp_path):
        result = CliRunner().invoke(
            cli_main.main, ["cache", "info", "--cache-dir", str(tmp_path / "none")]
        )

        assert result.exit_code == 0
        assert "No cache" in result.output

    def test_info_reports_entries(self, tmp_path):
        cache = make_cache(tmp_path, namespace="/work/a")
        fill(cache, 3)
        cache.flush()

        result = CliRunner().invoke(
            cli_main.main, ["cache", "info", "--cache-dir", str(tmp_path)]
        )

        assert result.exit_code == 0
        assert "Summary cache" in result.output
        assert "/work/a" in result.output

    def test_prune_removes_surplus_entries(self, tmp_path):
        cache = make_cache(tmp_path)
        fill(cache, 30)
        cache.flush()

        result = CliRunner().invoke(
            cli_main.main,
            [
                "cache",
                "prune",
                "--cache-dir",
                str(tmp_path),
                "--max-entries",
                "5",
                "--max-age-days",
                "-1",
            ],
        )

        assert result.exit_code == 0
        assert "Removed" in result.output
        assert cache_info(str(tmp_path)).entries == 5

    def test_prune_says_when_there_is_nothing_to_do(self, tmp_path):
        cache = make_cache(tmp_path)
        fill(cache, 2)
        cache.flush()

        result = CliRunner().invoke(
            cli_main.main, ["cache", "prune", "--cache-dir", str(tmp_path)]
        )

        assert result.exit_code == 0
        assert "Nothing to prune" in result.output

    def test_clear_asks_first(self, tmp_path):
        cache = make_cache(tmp_path)
        fill(cache, 4)
        cache.flush()

        result = CliRunner().invoke(
            cli_main.main, ["cache", "clear", "--cache-dir", str(tmp_path)], input="n\n"
        )

        assert result.exit_code == 0
        assert cache_info(str(tmp_path)).entries == 4

    def test_clear_with_force_does_not_ask(self, tmp_path):
        cache = make_cache(tmp_path)
        fill(cache, 4)
        cache.flush()

        result = CliRunner().invoke(
            cli_main.main, ["cache", "clear", "--cache-dir", str(tmp_path), "--force"]
        )

        assert result.exit_code == 0
        assert "Removed 4" in result.output
        assert cache_info(str(tmp_path)).entries == 0

    def test_clear_on_a_missing_cache(self, tmp_path):
        result = CliRunner().invoke(
            cli_main.main,
            ["cache", "clear", "--cache-dir", str(tmp_path / "none"), "--force"],
        )

        assert result.exit_code == 0
        assert "No cache" in result.output

    def test_the_group_is_discoverable(self):
        result = CliRunner().invoke(cli_main.main, ["cache", "--help"])

        assert result.exit_code == 0
        for command in ("info", "prune", "clear"):
            assert command in result.output
