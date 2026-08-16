"""Tests for the directory roll-up.

The behaviour under test comes from the issue: the roll-up walked the tree
recursively and made one blocking LLM call per directory, nothing was cached,
failed roll-ups were fed to the README prompt as ``{"error": ...}``
placeholders, and the 15-file threshold was a literal in the middle of the
function.
"""

import threading
import time

import pytest

from repo2readme.services import rollup
from repo2readme.services.rollup import (
    DEFAULT_ROLLUP_THRESHOLD,
    RollupResult,
    build_directory_tree,
    contents_fingerprint,
    count_directories,
    directory_cache_key,
    generate_hierarchical_summaries,
    is_directory_key,
    nodes_by_depth,
)


def summaries(*paths):
    return [{"file_path": p, "description": f"summary of {p}"} for p in paths]


def wide_repo(directories=4, files_per_directory=8):
    """A repository with enough files to trigger the roll-up."""
    paths = []
    for d in range(directories):
        for f in range(files_per_directory):
            paths.append(f"pkg{d}/module{f}.py")
    return summaries(*paths)


class RecordingSummarizer:
    """Stands in for ``summarize_directory``, recording what it was asked."""

    def __init__(self, delay=0.0, fail_for=()):
        self.delay = delay
        self.fail_for = set(fail_for)
        self.calls = []
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def __call__(self, dir_path, contents_summaries, provider=None,
                 model_name=None, base_url=None):
        with self._lock:
            self.calls.append(dir_path)
            self.active += 1
            self.peak = max(self.peak, self.active)

        try:
            if self.delay:
                time.sleep(self.delay)
            if dir_path in self.fail_for:
                return {"file_path": dir_path, "error": "rate limited"}
            return {"file_path": dir_path, "description": f"rolled up {dir_path}"}
        finally:
            with self._lock:
                self.active -= 1


class FakeCache:
    """The slice of SummaryCache the roll-up uses."""

    def __init__(self):
        self.entries = {}
        self.hits = 0
        self.misses = 0

    def get(self, file_path, content, language):
        entry = self.entries.get(file_path)
        if entry and entry[0] == content and entry[1] == language:
            self.hits += 1
            return entry[2]
        self.misses += 1
        return None

    def put(self, file_path, content, language, summary, mtime):
        self.entries[file_path] = (content, language, summary, mtime)


@pytest.fixture
def summarizer(monkeypatch):
    recorder = RecordingSummarizer()
    monkeypatch.setattr(rollup, "summarize_directory", recorder)
    return recorder


# ---------------------------------------------------------------------------
# Tree shape
# ---------------------------------------------------------------------------


def test_build_directory_tree_places_files_under_their_directories():
    tree = build_directory_tree(summaries("a/b/c.py", "a/d.py", "top.py"))

    assert [f["file_path"] for f in tree["files"]] == ["top.py"]
    assert set(tree["children"]) == {"a"}
    a = tree["children"]["a"]
    assert [f["file_path"] for f in a["files"]] == ["a/d.py"]
    assert [f["file_path"] for f in a["children"]["b"]["files"]] == ["a/b/c.py"]


def test_build_directory_tree_ignores_junk():
    tree = build_directory_tree(["a string", {"description": "no path"}, {}])

    assert tree["files"] == []
    assert tree["children"] == {}


def test_nodes_by_depth_groups_siblings():
    tree = build_directory_tree(summaries("a/x.py", "b/y.py", "a/c/z.py"))
    levels = nodes_by_depth(tree)

    assert [len(level) for level in levels] == [1, 2, 1]
    assert {node["path"] for node in levels[1]} == {"a", "b"}
    assert [node["path"] for node in levels[2]] == ["a/c"]


def test_count_directories_excludes_the_root():
    tree = build_directory_tree(summaries("a/x.py", "b/y.py", "a/c/z.py"))

    assert count_directories(tree) == 3


def test_directory_keys_are_distinguishable():
    key = directory_cache_key("src/utils")

    assert is_directory_key(key)
    assert not is_directory_key("src/utils/helpers.py")
    assert "src/utils" in key


def test_the_fingerprint_is_stable_and_content_sensitive():
    a = contents_fingerprint(summaries("x.py", "y.py"))
    b = contents_fingerprint(summaries("x.py", "y.py"))
    c = contents_fingerprint(summaries("x.py", "z.py"))

    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# The threshold
# ---------------------------------------------------------------------------


def test_a_small_repository_skips_the_rollup(summarizer):
    files = summaries("a/x.py", "b/y.py")

    result = generate_hierarchical_summaries(files)

    assert result.summaries == files
    assert result.failures == []
    assert summarizer.calls == []


def test_the_threshold_is_configurable(summarizer):
    files = summaries("a/x.py", "a/y.py", "b/z.py")

    skipped = generate_hierarchical_summaries(files, threshold=10)
    assert skipped.summaries == files
    assert summarizer.calls == []

    rolled = generate_hierarchical_summaries(files, threshold=2)
    assert rolled.summaries != files
    assert summarizer.calls == ["a"]  # the low threshold forced the roll-up


def test_the_default_threshold_is_unchanged():
    assert DEFAULT_ROLLUP_THRESHOLD == 15


def test_a_repository_at_the_threshold_is_left_alone(summarizer):
    files = summaries(*[f"pkg/m{i}.py" for i in range(DEFAULT_ROLLUP_THRESHOLD)])

    result = generate_hierarchical_summaries(files)

    assert result.summaries == files
    assert summarizer.calls == []


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_sibling_directories_are_summarized_concurrently(monkeypatch):
    recorder = RecordingSummarizer(delay=0.05)
    monkeypatch.setattr(rollup, "summarize_directory", recorder)

    generate_hierarchical_summaries(wide_repo(directories=4), max_workers=4)

    assert len(recorder.calls) == 4
    # Serially this would be four calls one after another; the point of the
    # change is that they overlap.
    assert recorder.peak > 1


def test_max_workers_bounds_the_concurrency(monkeypatch):
    recorder = RecordingSummarizer(delay=0.05)
    monkeypatch.setattr(rollup, "summarize_directory", recorder)

    generate_hierarchical_summaries(wide_repo(directories=6), max_workers=2)

    assert recorder.peak <= 2


def test_one_worker_still_produces_every_summary(monkeypatch):
    recorder = RecordingSummarizer()
    monkeypatch.setattr(rollup, "summarize_directory", recorder)

    result = generate_hierarchical_summaries(wide_repo(directories=3), max_workers=1)

    assert recorder.peak == 1
    assert len(result.summaries) == 3


def test_a_child_is_finished_before_its_parent(monkeypatch):
    """Depth ordering is what makes the concurrency safe."""
    order = []

    def summarize(dir_path, contents_summaries, **kwargs):
        order.append(dir_path)
        return {"file_path": dir_path, "description": "x"}

    monkeypatch.setattr(rollup, "summarize_directory", summarize)

    paths = [f"src/deep/m{i}.py" for i in range(10)]
    paths += [f"src/other{i}.py" for i in range(10)]
    generate_hierarchical_summaries(summaries(*paths))

    assert order.index("src/deep") < order.index("src")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_an_unchanged_directory_is_not_summarized_twice(summarizer):
    files = wide_repo(directories=3)
    cache = FakeCache()

    generate_hierarchical_summaries(files, summary_cache=cache)
    first = list(summarizer.calls)
    summarizer.calls.clear()

    second = generate_hierarchical_summaries(files, summary_cache=cache)

    assert first  # the first run did the work
    assert summarizer.calls == []  # the second run paid for nothing
    assert cache.hits == len(first)
    assert len(second.summaries) == 3


def test_a_changed_file_summary_invalidates_its_directory(summarizer):
    files = wide_repo(directories=2)
    cache = FakeCache()

    generate_hierarchical_summaries(files, summary_cache=cache)
    summarizer.calls.clear()

    files[0] = dict(files[0], description="something else entirely")
    generate_hierarchical_summaries(files, summary_cache=cache)

    assert summarizer.calls == ["pkg0"]  # only the directory that changed


def test_the_cached_summary_is_the_one_returned(summarizer):
    files = wide_repo(directories=1, files_per_directory=20)
    cache = FakeCache()

    first = generate_hierarchical_summaries(files, summary_cache=cache)
    second = generate_hierarchical_summaries(files, summary_cache=cache)

    assert first.summaries == second.summaries


def test_the_cache_keys_are_reported(summarizer):
    result = generate_hierarchical_summaries(wide_repo(directories=3))

    assert result.cache_keys == {
        directory_cache_key("pkg0"),
        directory_cache_key("pkg1"),
        directory_cache_key("pkg2"),
    }
    assert all(is_directory_key(key) for key in result.cache_keys)


def test_no_cache_is_a_valid_choice(summarizer):
    result = generate_hierarchical_summaries(wide_repo(directories=2), summary_cache=None)

    assert len(result.summaries) == 2


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_a_failed_rollup_never_reaches_the_readme_prompt(monkeypatch):
    recorder = RecordingSummarizer(fail_for={"pkg1"})
    monkeypatch.setattr(rollup, "summarize_directory", recorder)

    result = generate_hierarchical_summaries(wide_repo(directories=3))

    for summary in result.summaries:
        assert "error" not in summary


def test_a_failed_rollup_is_reported(monkeypatch):
    recorder = RecordingSummarizer(fail_for={"pkg1"})
    monkeypatch.setattr(rollup, "summarize_directory", recorder)

    result = generate_hierarchical_summaries(wide_repo(directories=3))

    assert [f.file_path for f in result.failures] == ["pkg1"]
    assert "rate limited" in result.failures[0].reason


def test_a_failed_rollup_falls_back_to_its_contents(monkeypatch):
    recorder = RecordingSummarizer(fail_for={"pkg1"})
    monkeypatch.setattr(rollup, "summarize_directory", recorder)

    result = generate_hierarchical_summaries(wide_repo(directories=2))

    paths = {s["file_path"] for s in result.summaries}
    # pkg0 rolled up; pkg1 failed, so its file summaries are used instead of
    # losing the directory entirely.
    assert "pkg0" in paths
    assert "pkg1" not in paths
    assert "pkg1/module0.py" in paths


def test_a_raising_summarizer_is_treated_as_a_failure(monkeypatch):
    def explode(dir_path, contents_summaries, **kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(rollup, "summarize_directory", explode)

    result = generate_hierarchical_summaries(wide_repo(directories=2))

    assert {f.file_path for f in result.failures} == {"pkg0", "pkg1"}
    assert all("connection reset" in f.reason for f in result.failures)


def test_a_failed_rollup_is_not_cached(monkeypatch):
    recorder = RecordingSummarizer(fail_for={"pkg0", "pkg1"})
    monkeypatch.setattr(rollup, "summarize_directory", recorder)
    cache = FakeCache()

    generate_hierarchical_summaries(wide_repo(directories=2), summary_cache=cache)
    recorder.calls.clear()
    recorder.fail_for = set()

    generate_hierarchical_summaries(wide_repo(directories=2), summary_cache=cache)

    # A failure must be retried on the next run, not served from the cache.
    assert sorted(recorder.calls) == ["pkg0", "pkg1"]


# ---------------------------------------------------------------------------
# Shape of the result
# ---------------------------------------------------------------------------


def test_a_single_child_directory_is_not_restated(summarizer):
    paths = [f"src/only/m{i}.py" for i in range(20)]

    generate_hierarchical_summaries(summaries(*paths))

    # src has exactly one child (src/only) and no files of its own, so there is
    # nothing to synthesize at that level and no call to make.
    assert summarizer.calls == ["src/only"]


def test_root_level_files_survive_the_rollup(summarizer):
    paths = [f"pkg/m{i}.py" for i in range(20)] + ["README.md", "setup.py"]

    result = generate_hierarchical_summaries(summaries(*paths))

    paths_out = {s["file_path"] for s in result.summaries}
    assert {"README.md", "setup.py", "pkg"} == paths_out


def test_progress_is_advanced_once_per_directory(summarizer):
    class Progress:
        def __init__(self):
            self.advanced = 0
            self.total = None

        def update(self, task_id, advance=None, total=None, completed=None):
            if advance:
                self.advanced += advance
            if total is not None:
                self.total = total

    progress = Progress()
    files = wide_repo(directories=3)

    generate_hierarchical_summaries(files, progress=progress, task_id=1)

    assert progress.total == count_directories(build_directory_tree(files))
    assert progress.advanced == progress.total


def test_an_empty_input_produces_nothing(summarizer):
    result = generate_hierarchical_summaries([], threshold=-1)

    assert result.summaries == []
    assert result.failures == []


def test_the_result_defaults_are_empty():
    result = RollupResult()

    assert result.summaries == []
    assert result.failures == []
    assert result.cache_keys == set()


def test_summarization_still_re_exports_the_entry_point():
    from repo2readme.services import summarization

    assert summarization.generate_hierarchical_summaries is (
        generate_hierarchical_summaries
    )
    assert summarization.build_directory_tree is build_directory_tree
