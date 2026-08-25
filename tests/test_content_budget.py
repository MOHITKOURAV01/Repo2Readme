"""Tests for the per-request content budget."""

import importlib
import inspect

import pytest
from click.testing import CliRunner

from repo2readme.cache import SummaryCache
from repo2readme.utils.content_budget import (
    DEFAULT_MAX_CONTENT_CHARS,
    HEAD_SHARE,
    MARKER_TEMPLATE,
    MIN_MAX_CONTENT_CHARS,
    InvalidContentBudgetError,
    annotate_truncation,
    apply_content_budget,
    validate_max_content_chars,
)

cli_main = importlib.import_module("repo2readme.cli.main")


def source(lines, prefix="line"):
    return "".join(f"{prefix} {i}: def f{i}():\n    return {i}\n" for i in range(lines))


def marker_for(budgeted):
    return MARKER_TEMPLATE.format(omitted=budgeted.omitted_lines)


# ---------------------------------------------------------------------------
# validate_max_content_chars
# ---------------------------------------------------------------------------


def test_none_means_use_the_default():
    assert validate_max_content_chars(None) is None


def test_zero_means_send_files_whole():
    assert validate_max_content_chars(0) == 0


def test_a_usable_budget_is_accepted():
    assert validate_max_content_chars(10_000) == 10_000


def test_a_negative_budget_is_rejected():
    with pytest.raises(InvalidContentBudgetError) as excinfo:
        validate_max_content_chars(-1)
    assert "0 to send files whole" in str(excinfo.value)


def test_a_budget_too_small_for_an_excerpt_is_rejected():
    with pytest.raises(InvalidContentBudgetError) as excinfo:
        validate_max_content_chars(MIN_MAX_CONTENT_CHARS - 1)
    assert str(MIN_MAX_CONTENT_CHARS) in str(excinfo.value)


def test_a_non_numeric_budget_is_rejected():
    with pytest.raises(InvalidContentBudgetError):
        validate_max_content_chars("lots")


# ---------------------------------------------------------------------------
# apply_content_budget
# ---------------------------------------------------------------------------


def test_a_small_file_is_untouched():
    text = "print('hello')\n"
    budgeted = apply_content_budget(text, 10_000)
    assert budgeted.text == text
    assert budgeted.truncated is False
    assert budgeted.omitted_lines == 0


def test_a_file_exactly_at_the_budget_is_untouched():
    text = "x" * 1_000
    assert apply_content_budget(text, 1_000).truncated is False


def test_a_large_file_is_cut_to_the_budget():
    budgeted = apply_content_budget(source(4_000), 2_000)
    assert budgeted.truncated is True
    assert budgeted.kept_chars <= 2_000


def test_the_gap_is_marked():
    budgeted = apply_content_budget(source(4_000), 2_000)
    assert marker_for(budgeted) in budgeted.text
    assert "omitted" in budgeted.text


def test_the_head_of_the_file_is_kept():
    """Imports and the module docstring live at the top."""
    budgeted = apply_content_budget(source(4_000), 2_000)
    assert budgeted.text.startswith("line 0: def f0():")


def test_the_tail_of_the_file_is_kept():
    """The public entry points and the main guard live at the bottom."""
    budgeted = apply_content_budget(source(4_000), 2_000)
    assert budgeted.text.rstrip().endswith("return 3999")


def test_every_kept_line_is_a_whole_source_line():
    """The excerpt must never be spliced through the middle of a statement."""
    text = source(4_000)
    budgeted = apply_content_budget(text, 2_000)
    head, _, tail = budgeted.text.partition(marker_for(budgeted))
    originals = set(text.splitlines())
    for line in head.splitlines() + tail.splitlines():
        assert line in originals


def test_the_head_gets_the_larger_share():
    budgeted = apply_content_budget(source(4_000), 4_000)
    head, _, tail = budgeted.text.partition(marker_for(budgeted))
    assert len(head) > len(tail)
    assert len(head) >= len(budgeted.text) * (HEAD_SHARE - 0.15)


def test_the_original_size_is_recorded():
    text = source(4_000)
    budgeted = apply_content_budget(text, 2_000)
    assert budgeted.original_chars == len(text)
    assert budgeted.original_lines == len(text.splitlines())
    assert budgeted.omitted_lines > 0
    kept = budgeted.original_lines - budgeted.omitted_lines
    assert 0 < kept < budgeted.original_lines


def test_zero_sends_the_file_whole():
    text = source(4_000)
    budgeted = apply_content_budget(text, 0)
    assert budgeted.text == text
    assert budgeted.truncated is False


def test_the_default_budget_is_used_when_none_is_given():
    text = "x" * (DEFAULT_MAX_CONTENT_CHARS + 1_000)
    assert apply_content_budget(text).truncated is True
    assert apply_content_budget("x" * 100).truncated is False


def test_empty_content_is_handled():
    budgeted = apply_content_budget("", 1_000)
    assert budgeted.text == ""
    assert budgeted.truncated is False
    assert apply_content_budget(None, 1_000).text == ""


def test_a_single_enormous_line_still_fits_the_budget():
    """A minified bundle has no line boundaries to cut on."""
    budgeted = apply_content_budget("x" * 100_000, 2_000)
    assert budgeted.kept_chars <= 2_000


@pytest.mark.parametrize("lines", [20, 200, 2_000, 20_000])
@pytest.mark.parametrize("budget", [600, 2_000, 10_000])
def test_the_budget_is_never_exceeded(lines, budget):
    assert apply_content_budget(source(lines), budget).kept_chars <= budget


# ---------------------------------------------------------------------------
# annotate_truncation
# ---------------------------------------------------------------------------


def test_a_truncated_summary_says_so():
    budgeted = apply_content_budget(source(4_000), 2_000)
    annotated = annotate_truncation({"file_path": "a.py", "description": "d"}, budgeted)
    assert annotated["truncated"] is True
    assert annotated["omitted_lines"] == budgeted.omitted_lines
    assert annotated["original_lines"] == budgeted.original_lines


def test_an_untruncated_summary_is_left_alone():
    budgeted = apply_content_budget("x\n", 1_000)
    summary = {"file_path": "a.py", "description": "d"}
    assert annotate_truncation(summary, budgeted) == summary
    assert "truncated" not in annotate_truncation(summary, budgeted)


def test_a_failure_placeholder_is_left_alone():
    """Annotating it would make partition_summaries look at a different dict."""
    budgeted = apply_content_budget(source(4_000), 2_000)
    failure = {"file_path": "a.py", "error": "boom"}
    assert annotate_truncation(failure, budgeted) == failure


def test_a_non_dict_summary_is_left_alone():
    budgeted = apply_content_budget(source(4_000), 2_000)
    assert annotate_truncation("a plain string", budgeted) == "a plain string"


def test_the_original_summary_is_not_mutated():
    budgeted = apply_content_budget(source(4_000), 2_000)
    summary = {"file_path": "a.py", "description": "d"}
    annotate_truncation(summary, budgeted)
    assert "truncated" not in summary


# ---------------------------------------------------------------------------
# summarize_file
# ---------------------------------------------------------------------------


def _capture_chain(monkeypatch, result=None):
    sent = {}

    class FakeChain:
        def invoke(self, payload):
            sent["content"] = payload["content"]
            return result or {"file_path": payload["file_path"], "description": "d"}

    monkeypatch.setattr(
        "repo2readme.summarize.summary.create_summarizer",
        lambda *args, **kwargs: FakeChain(),
    )
    return sent


def test_a_large_file_no_longer_goes_whole(monkeypatch):
    """198,000 characters in one request is what the provider rejected."""
    sent = _capture_chain(monkeypatch)
    from repo2readme.summarize.summary import summarize_file

    summarize_file("big.py", "python", source(9_000), max_content_chars=5_000)

    assert len(sent["content"]) <= 5_000


def test_the_summary_records_that_it_is_an_excerpt(monkeypatch):
    _capture_chain(monkeypatch)
    from repo2readme.summarize.summary import summarize_file

    summary = summarize_file("big.py", "python", source(9_000), max_content_chars=5_000)

    assert summary["truncated"] is True
    assert summary["omitted_lines"] > 0


def test_a_small_file_is_sent_unchanged_and_unannotated(monkeypatch):
    sent = _capture_chain(monkeypatch)
    from repo2readme.summarize.summary import summarize_file

    summary = summarize_file("a.py", "python", "x = 1\n", max_content_chars=5_000)

    assert sent["content"] == "x = 1\n"
    assert "truncated" not in summary


def test_zero_restores_the_old_behaviour(monkeypatch):
    sent = _capture_chain(monkeypatch)
    from repo2readme.summarize.summary import summarize_file
    text = source(9_000)

    summarize_file("big.py", "python", text, max_content_chars=0)

    assert sent["content"] == text


def test_a_provider_failure_is_still_a_failure_placeholder(monkeypatch):
    class ExplodingChain:
        def invoke(self, payload):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "repo2readme.summarize.summary.create_summarizer",
        lambda *args, **kwargs: ExplodingChain(),
    )
    from repo2readme.summarize.summary import summarize_file

    summary = summarize_file("big.py", "python", source(9_000), max_content_chars=5_000)

    assert summary["error"]
    assert "truncated" not in summary


def test_the_roll_up_is_bounded_too(monkeypatch):
    sent = {}

    class FakeChain:
        def invoke(self, payload):
            sent["contents"] = payload["contents"]
            return {"file_path": payload["dir_path"], "description": "d"}

    monkeypatch.setattr(
        "repo2readme.summarize.directory_summary.create_llm", lambda **k: object()
    )
    monkeypatch.setattr(
        "repo2readme.summarize.directory_summary.PromptTemplate",
        lambda **k: _PipeInto(FakeChain()),
    )
    from repo2readme.summarize.directory_summary import summarize_directory

    summaries = [{"file_path": f"f{i}.py", "description": "x" * 200} for i in range(200)]
    summarize_directory("src", summaries, max_content_chars=3_000)

    assert len(sent["contents"]) <= 3_000


class _PipeInto:
    """Stands in for a PromptTemplate so ``prompt | model | parser`` works."""

    def __init__(self, chain):
        self._chain = chain

    def __or__(self, other):
        return self

    def invoke(self, payload):
        return self._chain.invoke(payload)


# ---------------------------------------------------------------------------
# The budget is part of the cache key
# ---------------------------------------------------------------------------


def _cache(tmp_path, **config):
    return SummaryCache(
        cache_dir=str(tmp_path / "cache"),
        config={"provider": "groq", "model": "m", "base_url": None, **config},
        prompt_template_hash="hash",
    )


def test_changing_the_budget_invalidates_the_cache(tmp_path):
    """A summary written from an excerpt is not valid for a whole file."""
    first = _cache(tmp_path, max_content_chars=1_000)
    first.put("/a.py", "code", "python", {"description": "d"}, 0)
    first.flush()

    second = _cache(tmp_path, max_content_chars=50_000)
    assert second.get("/a.py", "code", "python") is None


def test_the_same_budget_keeps_the_cache(tmp_path):
    first = _cache(tmp_path, max_content_chars=1_000)
    first.put("/a.py", "code", "python", {"description": "d"}, 0)
    first.flush()

    second = _cache(tmp_path, max_content_chars=1_000)
    assert second.get("/a.py", "code", "python") == {"description": "d"}


def test_the_existing_config_keys_still_invalidate(tmp_path):
    first = _cache(tmp_path)
    first.put("/a.py", "code", "python", {"description": "d"}, 0)
    first.flush()

    assert _cache(tmp_path, model="other").get("/a.py", "code", "python") is None
    assert _cache(tmp_path, provider="openai").get("/a.py", "code", "python") is None


def test_the_prompt_template_still_invalidates(tmp_path):
    first = _cache(tmp_path)
    first.put("/a.py", "code", "python", {"description": "d"}, 0)
    first.flush()

    other = SummaryCache(
        cache_dir=str(tmp_path / "cache"),
        config={"provider": "groq", "model": "m", "base_url": None},
        prompt_template_hash="a-different-hash",
    )
    assert other.get("/a.py", "code", "python") is None


def test_an_unserialisable_config_value_does_not_crash_the_hash(tmp_path):
    cache = _cache(tmp_path, extra=object())
    assert isinstance(cache._compute_config_hash(), str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_every_stage_accepts_a_budget():
    from repo2readme.services.summarization import (
        generate_all_summaries,
        generate_hierarchical_summaries,
    )
    from repo2readme.summarize.directory_summary import summarize_directory
    from repo2readme.summarize.summary import summarize_file

    for target in (
        summarize_file,
        summarize_directory,
        generate_all_summaries,
        generate_hierarchical_summaries,
    ):
        assert "max_content_chars" in inspect.signature(target).parameters


def test_the_flag_is_documented():
    result = CliRunner().invoke(cli_main.main, ["run", "--help"])
    # Click rewraps help text, so compare with the line breaks collapsed.
    flattened = " ".join(result.output.split())
    assert "--max-content-chars" in flattened
    assert "0 to send whole files" in flattened


def test_a_negative_budget_is_rejected_before_the_repository_is_loaded(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        cli_main, "RepoLoader", lambda *a, **k: pytest.fail("repository was loaded")
    )
    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--max-content-chars", "-5"]
    )
    assert result.exit_code == 2


def test_the_flag_reaches_the_summarization_stage(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
    seen = {}

    monkeypatch.setattr(cli_main, "setup_api_keys", lambda provider: None)
    monkeypatch.setattr(
        cli_main,
        "generate_all_summaries",
        lambda documents, summary_cache, **kwargs: (
            seen.update(kwargs),
            ([{"file_path": "main.py", "description": "d"}], []),
        )[1],
    )
    monkeypatch.setattr(
        cli_main,
        "generate_hierarchical_summaries",
        lambda file_summaries, **kwargs: file_summaries,
    )
    monkeypatch.setattr(cli_main, "run_pipeline", lambda *a, **k: "# r\n\nbody\n")

    result = CliRunner().invoke(
        cli_main.main,
        ["run", "--local", str(tmp_path), "--max-content-chars", "9000", "--force"],
    )

    assert result.exit_code == 0
    assert seen["max_content_chars"] == 9_000
