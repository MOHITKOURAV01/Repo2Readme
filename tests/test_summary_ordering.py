"""The same repository has to produce the same README.

From the issue: summaries were appended as each worker finished, so six
identical runs over twelve files produced six different orderings. The list is
interpolated straight into the generation prompt, so every run asked a different
question - and a fully warm cache still produced a different README, which is
not what a cache is for.
"""

import random
import threading
import time

import pytest

from repo2readme.services import summarization
from repo2readme.services.summarization import (
    build_directory_tree,
    generate_all_summaries,
    generate_hierarchical_summaries,
)


class NullCache:
    """A cache that never hits and never stores."""

    def __init__(self):
        self.puts = []

    def get(self, file_path, content, language):
        return None

    def put(self, file_path, content, language, summary, mtime):
        self.puts.append(file_path)


class WarmCache:
    """A cache that answers everything, the way a re-run does."""

    def get(self, file_path, content, language):
        return {"file_path": file_path, "description": "cached"}

    def put(self, file_path, content, language, summary, mtime):  # pragma: no cover
        raise AssertionError("a warm cache should not be written to")


def documents(count, prefix="/repo"):
    return [
        {
            "content": f"content of file {index}",
            "metadata": {
                "file_path": f"{prefix}/f{index:02d}.py",
                "relative_path": f"f{index:02d}.py",
                "language": "python",
                "mtime": 0,
            },
        }
        for index in range(count)
    ]


def paths_of(summaries):
    return [summary["file_path"] for summary in summaries]


@pytest.fixture
def uneven_summarizer(monkeypatch):
    """A summarizer whose files take wildly different amounts of time."""

    def summarize(file_path, language, content, **kwargs):
        time.sleep(random.random() / 200)
        return {"file_path": file_path, "description": "d"}

    monkeypatch.setattr(summarization, "summarize_file", summarize)
    return summarize


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_summaries_come_back_in_document_order(uneven_summarizer):
    docs = documents(24)
    summaries, errors = generate_all_summaries(
        documents=docs, summary_cache=NullCache(), max_workers=8
    )

    assert not errors
    assert paths_of(summaries) == [d["metadata"]["file_path"] for d in docs]


def test_repeated_runs_agree(uneven_summarizer):
    docs = documents(20)
    orderings = {
        tuple(
            paths_of(
                generate_all_summaries(
                    documents=docs, summary_cache=NullCache(), max_workers=8
                )[0]
            )
        )
        for _ in range(8)
    }

    assert len(orderings) == 1


def test_a_warm_cache_gives_the_same_order_as_a_cold_one(uneven_summarizer):
    docs = documents(16)

    cold, _ = generate_all_summaries(
        documents=docs, summary_cache=NullCache(), max_workers=6
    )
    warm, _ = generate_all_summaries(
        documents=docs, summary_cache=WarmCache(), max_workers=6
    )

    assert paths_of(cold) == paths_of(warm)


def test_worker_count_does_not_change_the_order(uneven_summarizer):
    docs = documents(16)
    orderings = {
        tuple(
            paths_of(
                generate_all_summaries(
                    documents=docs, summary_cache=NullCache(), max_workers=workers
                )[0]
            )
        )
        for workers in (1, 2, 4, 8)
    }

    assert len(orderings) == 1


def test_failures_are_reported_in_document_order(monkeypatch):
    def summarize(file_path, language, content, **kwargs):
        time.sleep(random.random() / 200)
        if file_path.endswith(("01.py", "05.py", "09.py")):
            raise RuntimeError("rate limited")
        return {"file_path": file_path, "description": "d"}

    monkeypatch.setattr(summarization, "summarize_file", summarize)

    summaries, errors = generate_all_summaries(
        documents=documents(12), summary_cache=NullCache(), max_workers=6
    )

    assert [failure.file_path for failure in errors] == [
        "/repo/f01.py",
        "/repo/f05.py",
        "/repo/f09.py",
    ]
    # And the successful ones close over the gaps, still in order.
    assert paths_of(summaries) == [
        f"/repo/f{index:02d}.py"
        for index in range(12)
        if index not in (1, 5, 9)
    ]


# ---------------------------------------------------------------------------
# Behaviour that must not have changed
# ---------------------------------------------------------------------------


def test_every_document_is_processed_exactly_once(monkeypatch):
    seen = []
    lock = threading.Lock()

    def summarize(file_path, language, content, **kwargs):
        with lock:
            seen.append(file_path)
        return {"file_path": file_path, "description": "d"}

    monkeypatch.setattr(summarization, "summarize_file", summarize)

    generate_all_summaries(
        documents=documents(30), summary_cache=NullCache(), max_workers=8
    )

    assert sorted(seen) == sorted(f"/repo/f{i:02d}.py" for i in range(30))


def test_no_documents_returns_two_empty_lists():
    assert generate_all_summaries(documents=[], summary_cache=NullCache()) == ([], [])


def test_error_placeholders_still_reach_the_first_list(monkeypatch):
    def summarize(file_path, language, content, **kwargs):
        if file_path.endswith("02.py"):
            return {"file_path": file_path, "error": "boom"}
        return {"file_path": file_path, "description": "d"}

    monkeypatch.setattr(summarization, "summarize_file", summarize)

    cache = NullCache()
    summaries, errors = generate_all_summaries(
        documents=documents(4), summary_cache=cache, max_workers=4
    )

    assert not errors
    assert summaries[2] == {"file_path": "/repo/f02.py", "error": "boom"}
    # A failure is still never cached.
    assert "/repo/f02.py" not in cache.puts
    assert len(cache.puts) == 3


def test_a_summary_that_is_falsy_is_still_kept(monkeypatch):
    """The slot is a sentinel, so a provider returning None does not vanish."""
    monkeypatch.setattr(
        summarization, "summarize_file", lambda file_path, language, content, **k: None
    )

    summaries, errors = generate_all_summaries(
        documents=documents(3), summary_cache=NullCache(), max_workers=3
    )

    assert summaries == [None, None, None]
    assert not errors


def test_progress_advances_once_per_document(uneven_summarizer):
    class Recorder:
        def __init__(self):
            self.advances = 0

        def update(self, task_id, advance=0, **kwargs):
            self.advances += advance

    recorder = Recorder()
    generate_all_summaries(
        documents=documents(9),
        summary_cache=NullCache(),
        max_workers=4,
        progress=recorder,
        task_id=1,
    )

    assert recorder.advances == 9


# ---------------------------------------------------------------------------
# The roll-up
# ---------------------------------------------------------------------------


def test_directory_tree_keeps_files_in_the_order_given():
    summaries = [
        {"file_path": "src/b.py", "description": "b"},
        {"file_path": "src/a.py", "description": "a"},
    ]
    tree = build_directory_tree(summaries)

    assert [f["file_path"] for f in tree["children"]["src"]["files"]] == [
        "src/b.py",
        "src/a.py",
    ]


def test_rollup_visits_directories_in_sorted_order(monkeypatch):
    visited = []

    def fake_summarize_directory(dir_path, contents_summaries, **kwargs):
        visited.append(dir_path)
        return {"file_path": dir_path, "description": "dir"}

    import repo2readme.summarize.directory_summary as directory_summary

    monkeypatch.setattr(
        directory_summary, "summarize_directory", fake_summarize_directory
    )

    # Deliberately shuffled input; the roll-up must not depend on it.
    summaries = []
    for directory in ("zeta", "alpha", "mid"):
        for index in range(6):
            summaries.append(
                {"file_path": f"{directory}/f{index}.py", "description": "d"}
            )

    generate_hierarchical_summaries(file_summaries=summaries)

    assert visited == ["alpha", "mid", "zeta"]


def test_rollup_is_skipped_for_small_repositories():
    summaries = [{"file_path": f"f{i}.py", "description": "d"} for i in range(10)]
    assert generate_hierarchical_summaries(file_summaries=summaries) == summaries
