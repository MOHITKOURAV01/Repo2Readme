"""Tests for the per-stage run estimate."""

import importlib
import random

import pytest
from click.testing import CliRunner

from repo2readme.cache import SummaryCache
from repo2readme.services.estimation import (
    CHARS_PER_TOKEN,
    STAGE_DIRECTORY_ROLLUPS,
    STAGE_FILE_SUMMARIES,
    STAGE_README,
    STAGE_REVIEW,
    build_estimate_lines,
    estimate_analysis_cost,
    estimate_run,
    estimate_tokens,
    format_size,
    readme_prompt_overhead,
    review_prompt_overhead,
    rollup_prompt_overhead,
    round_tokens,
    summary_prompt_overhead,
)
from repo2readme.services.summarization import (
    ROLLUP_THRESHOLD,
    count_directory_rollups,
    generate_hierarchical_summaries,
)

cli_main = importlib.import_module("repo2readme.cli.main")


def docs(count, size=300, prefix="src"):
    return [
        {
            "content": "x" * size,
            "metadata": {
                "file_path": f"/repo/{prefix}/mod{i % 4}/f{i}.py",
                "relative_path": f"{prefix}/mod{i % 4}/f{i}.py",
                "file_type": ".py",
                "language": "python",
            },
        }
        for i in range(count)
    ]


def stage(estimate, name):
    return next(s for s in estimate.stages if s.name == name)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("x" * (CHARS_PER_TOKEN * 100)) == 100


def test_estimate_tokens_never_returns_zero_for_real_text():
    assert estimate_tokens("x") == 1


def test_estimate_tokens_of_nothing_is_zero():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


@pytest.mark.parametrize(
    "value, expected",
    [(0, 0), (999, 999), (1_234, 1_000), (1_600, 2_000), (219_448, 220_000)],
)
def test_rounding_drops_precision_the_estimate_cannot_justify(value, expected):
    assert round_tokens(value) == expected


def test_format_size_is_unchanged():
    assert format_size(512) == "512 B"
    assert format_size(2048) == "2.0 KB"
    assert format_size(5 * 1024 * 1024) == "5.0 MB"


def test_the_old_helper_still_describes_the_source():
    tokens, size, count = estimate_analysis_cost(docs(3, size=300))
    assert count == 3
    assert size == 900
    assert tokens == 3 * (300 // CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Prompt overheads are measured, not guessed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "measure",
    [
        summary_prompt_overhead,
        rollup_prompt_overhead,
        readme_prompt_overhead,
        review_prompt_overhead,
    ],
)
def test_every_prompt_overhead_is_positive(measure):
    assert measure() > 0


def test_summary_overhead_tracks_the_real_template():
    from repo2readme.summarize.summary import PROMPT_TEMPLATE

    assert summary_prompt_overhead() >= estimate_tokens(PROMPT_TEMPLATE)


def test_readme_overhead_tracks_the_real_template():
    from repo2readme.readme.readme_generator import README_PROMPT_TEMPLATE

    assert readme_prompt_overhead() == estimate_tokens(README_PROMPT_TEMPLATE)


# ---------------------------------------------------------------------------
# count_directory_rollups agrees with the run
# ---------------------------------------------------------------------------


def _observed_rollups(paths, monkeypatch):
    calls = []

    def fake_summarize_directory(dir_path, contents_summaries, **kwargs):
        calls.append(len(contents_summaries))
        return {"file_path": dir_path, "description": "d"}

    monkeypatch.setattr(
        "repo2readme.summarize.directory_summary.summarize_directory",
        fake_summarize_directory,
    )
    generate_hierarchical_summaries(
        [{"file_path": path, "description": "x"} for path in paths]
    )
    return len(calls), sum(calls)


@pytest.mark.parametrize(
    "paths",
    [
        [f"src/a/f{i}.py" for i in range(20)],
        [f"f{i}.py" for i in range(20)],
        ["a/b/c/d/deep.py"] + [f"pkg/m{i}.py" for i in range(19)],
        [f"x/y{i}/z{j}.py" for i in range(5) for j in range(4)],
        ["only.py"] + [f"a/b{i}.py" for i in range(30)],
    ],
)
def test_rollup_count_matches_the_run_for_known_layouts(paths, monkeypatch):
    assert count_directory_rollups(paths) == _observed_rollups(paths, monkeypatch)


def test_rollup_count_matches_the_run_for_random_layouts(monkeypatch):
    """The estimate and the run must not be able to drift apart."""
    rng = random.Random(20260825)
    for _ in range(15):
        size = rng.randint(ROLLUP_THRESHOLD + 1, 60)
        paths = [
            f"d{rng.randint(0, 4)}/s{rng.randint(0, 3)}/f{i}.py" for i in range(size)
        ]
        assert count_directory_rollups(paths) == _observed_rollups(paths, monkeypatch)


def test_no_rollups_below_the_threshold():
    assert count_directory_rollups([f"f{i}.py" for i in range(ROLLUP_THRESHOLD)]) == (0, 0)


def test_a_flat_repository_above_the_threshold_needs_no_rollup():
    """Nothing to merge when every file is at the root."""
    paths = [f"f{i}.py" for i in range(ROLLUP_THRESHOLD + 5)]
    assert count_directory_rollups(paths) == (0, 0)


# ---------------------------------------------------------------------------
# estimate_run
# ---------------------------------------------------------------------------


def test_the_readme_loop_is_counted():
    estimate = estimate_run(docs(3))
    assert stage(estimate, STAGE_README).requests == 3
    assert stage(estimate, STAGE_REVIEW).requests == 3
    assert stage(estimate, STAGE_README).tokens > 0


def test_the_readme_loop_is_reported_as_an_upper_bound():
    estimate = estimate_run(docs(3))
    assert stage(estimate, STAGE_README).is_bounded
    assert stage(estimate, STAGE_REVIEW).is_bounded
    assert not stage(estimate, STAGE_FILE_SUMMARIES).is_bounded


def test_the_iteration_cap_comes_from_the_orchestrator():
    from repo2readme.services.orchestrator import MAX_README_ITERATIONS

    assert stage(estimate_run(docs(2)), STAGE_README).requests == MAX_README_ITERATIONS


def test_the_iteration_cap_can_be_overridden():
    estimate = estimate_run(docs(2), max_readme_iterations=1)
    assert stage(estimate, STAGE_README).requests == 1


def test_directory_rollups_are_counted_above_the_threshold():
    estimate = estimate_run(docs(ROLLUP_THRESHOLD + 10))
    assert stage(estimate, STAGE_DIRECTORY_ROLLUPS).requests > 0


def test_no_rollup_stage_for_a_small_repository():
    estimate = estimate_run(docs(3))
    assert stage(estimate, STAGE_DIRECTORY_ROLLUPS).requests == 0
    assert STAGE_DIRECTORY_ROLLUPS not in [s.name for s in estimate.billable_stages]


def test_summary_requests_match_the_uncached_file_count():
    estimate = estimate_run(docs(7))
    assert stage(estimate, STAGE_FILE_SUMMARIES).requests == 7
    assert estimate.files_selected == 7
    assert estimate.files_cached == 0


def test_the_total_exceeds_the_file_count():
    """81 files was reported as 81 requests; the README loop makes it more."""
    estimate = estimate_run(docs(81))
    assert estimate.total_requests > 81


def test_source_bytes_are_measured_in_utf8():
    estimate = estimate_run([{"content": "é", "metadata": {"file_path": "/a.py"}}])
    assert estimate.total_bytes == 2


def test_an_empty_run_estimates_nothing():
    estimate = estimate_run([])
    assert estimate.total_requests == 0
    assert estimate.total_tokens == 0
    assert estimate.billable_stages == ()


# ---------------------------------------------------------------------------
# Cache awareness
# ---------------------------------------------------------------------------


class _AllCached:
    def contains(self, file_path, content, language):
        return True


class _NoneCached:
    def contains(self, file_path, content, language):
        return False


class _Exploding:
    def contains(self, file_path, content, language):
        raise RuntimeError("cache on fire")


def test_cached_files_are_not_charged_for():
    estimate = estimate_run(docs(20), summary_cache=_AllCached())
    assert estimate.files_cached == 20
    assert estimate.files_to_summarize == 0
    assert stage(estimate, STAGE_FILE_SUMMARIES).requests == 0
    assert stage(estimate, STAGE_FILE_SUMMARIES).tokens == 0


def test_a_fully_cached_run_still_costs_the_readme_loop():
    estimate = estimate_run(docs(20), summary_cache=_AllCached())
    assert estimate.total_requests > 0


def test_caching_lowers_the_estimate():
    cold = estimate_run(docs(20), summary_cache=_NoneCached())
    warm = estimate_run(docs(20), summary_cache=_AllCached())
    assert warm.total_tokens < cold.total_tokens
    assert warm.total_requests < cold.total_requests


def test_the_rollup_still_covers_cached_files():
    """Every selected file has a summary by roll-up time, cached or not."""
    cold = estimate_run(docs(30), summary_cache=_NoneCached())
    warm = estimate_run(docs(30), summary_cache=_AllCached())
    assert (
        stage(warm, STAGE_DIRECTORY_ROLLUPS).requests
        == stage(cold, STAGE_DIRECTORY_ROLLUPS).requests
    )


def test_a_broken_cache_does_not_break_the_estimate():
    estimate = estimate_run(docs(4), summary_cache=_Exploding())
    assert estimate.files_cached == 0
    assert stage(estimate, STAGE_FILE_SUMMARIES).requests == 4


def test_a_cache_without_contains_is_treated_as_cold():
    estimate = estimate_run(docs(4), summary_cache=object())
    assert estimate.files_cached == 0


def test_no_cache_at_all_is_treated_as_cold():
    assert estimate_run(docs(4)).files_cached == 0


# ---------------------------------------------------------------------------
# SummaryCache.contains
# ---------------------------------------------------------------------------


def _cache(tmp_path, **config):
    return SummaryCache(
        cache_dir=str(tmp_path / "cache"),
        config={"provider": "groq", "model": "m", "base_url": None, **config},
        prompt_template_hash="hash",
    )


def test_contains_reports_a_stored_entry(tmp_path):
    cache = _cache(tmp_path)
    cache.put("/a.py", "code", "python", {"description": "d"}, 0)
    assert cache.contains("/a.py", "code", "python") is True


def test_contains_is_false_for_changed_content(tmp_path):
    cache = _cache(tmp_path)
    cache.put("/a.py", "code", "python", {"description": "d"}, 0)
    assert cache.contains("/a.py", "different", "python") is False


def test_contains_is_false_for_a_changed_language(tmp_path):
    cache = _cache(tmp_path)
    cache.put("/a.py", "code", "python", {"description": "d"}, 0)
    assert cache.contains("/a.py", "code", "ruby") is False


def test_contains_is_false_for_an_unknown_file(tmp_path):
    assert _cache(tmp_path).contains("/nope.py", "code", "python") is False


def test_contains_does_not_move_the_counters(tmp_path):
    cache = _cache(tmp_path)
    cache.put("/a.py", "code", "python", {"description": "d"}, 0)
    before = cache.stats()
    cache.contains("/a.py", "code", "python")
    cache.contains("/nope.py", "code", "python")
    after = cache.stats()
    assert before["hits"] == after["hits"]
    assert before["misses"] == after["misses"]


def test_contains_does_not_invalidate_on_a_config_change(tmp_path):
    """Asking about a run must not rewrite the cache on the strength of it."""
    cache = _cache(tmp_path)
    cache.put("/a.py", "code", "python", {"description": "d"}, 0)
    cache.flush()

    other = _cache(tmp_path, model="a-different-model")
    assert other.contains("/a.py", "code", "python") is False
    assert other.stats()["invalidations"] == 0

    # The original entry is still on disk, so the first cache still finds it.
    assert _cache(tmp_path).contains("/a.py", "code", "python") is True


def test_get_still_invalidates_on_a_config_change(tmp_path):
    cache = _cache(tmp_path)
    cache.put("/a.py", "code", "python", {"description": "d"}, 0)
    cache.flush()

    other = _cache(tmp_path, model="a-different-model")
    assert other.get("/a.py", "code", "python") is None
    assert other.stats()["invalidations"] == 1


def test_get_and_contains_agree(tmp_path):
    cache = _cache(tmp_path)
    cache.put("/a.py", "code", "python", {"description": "d"}, 0)
    for args in [
        ("/a.py", "code", "python"),
        ("/a.py", "other", "python"),
        ("/a.py", "code", "ruby"),
        ("/b.py", "code", "python"),
    ]:
        assert cache.contains(*args) == (cache.get(*args) is not None)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_rendered_breakdown_names_every_billable_stage():
    body = "\n".join(build_estimate_lines(estimate_run(docs(30))))
    assert STAGE_FILE_SUMMARIES in body
    assert STAGE_DIRECTORY_ROLLUPS in body
    assert STAGE_README in body
    assert STAGE_REVIEW in body
    assert "Total (upper bound)" in body


def test_rendered_breakdown_shows_the_cached_split():
    body = "\n".join(build_estimate_lines(estimate_run(docs(9), summary_cache=_AllCached())))
    assert "0 to summarize, 9 cached" in body


def test_rendered_breakdown_hides_the_split_when_nothing_is_cached():
    body = "\n".join(build_estimate_lines(estimate_run(docs(9))))
    assert "cached" not in body


def test_rendered_breakdown_says_so_when_there_is_nothing_to_send():
    estimate = estimate_run(docs(4), summary_cache=_AllCached(), max_readme_iterations=0)
    body = "\n".join(build_estimate_lines(estimate))
    assert "already cached" in body


def test_rendered_columns_line_up():
    """Including for a repository whose totals run to seven figures."""
    lines = build_estimate_lines(estimate_run(docs(4000, size=9000)))
    table = lines[lines.index(next(x for x in lines if x.startswith("Stage"))) :]
    assert len({len(line) for line in table}) == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_dry_run_prints_the_breakdown(tmp_path):
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--dry-run"]
    )

    assert result.exit_code == 0
    assert "File summaries" in result.output
    assert "Total (upper bound)" in result.output


def test_the_confirmation_prompt_follows_the_breakdown(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")

    monkeypatch.setattr(
        cli_main, "setup_api_keys", lambda provider: pytest.fail("should not be called")
    )
    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path)], input="n\n"
    )

    assert result.exit_code == 0
    assert result.output.index("Total (upper bound)") < result.output.index("Proceed?")
