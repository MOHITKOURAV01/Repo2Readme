"""Tests for the empty-analysis guard and the shared skip-reason grouping."""

import importlib

import pytest
from click.testing import CliRunner

from repo2readme.services.preflight import (
    CATEGORY_ORDER,
    EmptyAnalysis,
    build_empty_analysis_lines,
    build_skip_summary_lines,
    categorize_skip_reason,
    check_analysis_not_empty,
    group_skip_reasons,
    render_empty_analysis,
)

cli_main = importlib.import_module("repo2readme.cli.main")


# ---------------------------------------------------------------------------
# categorize_skip_reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason, expected",
    [
        ("excluded by pattern", "excluded by --exclude"),
        ("ignored by default rules", "ignored by default rules"),
        ("protected large file", "protected large file"),
        ("ignored by gitignore", "ignored by .gitignore"),
        ("binary_file", "binary content"),
        ("broken symbolic link", "broken symbolic link"),
        ("symbolic link outside repository", "symlink outside the repository"),
        ("circular or duplicate symbolic link", "circular symbolic link"),
        ("filtered", "ignored by default rules"),
    ],
)
def test_known_reasons_map_to_a_stable_category(reason, expected):
    assert categorize_skip_reason(reason) == expected


def test_size_reasons_collapse_regardless_of_the_measurement():
    """Every distinct size used to produce its own row in the breakdown."""
    reasons = [
        "exceeds maximum file size (317440 B > 204800 B limit)",
        "exceeds maximum file size (999999 B > 204800 B limit)",
        "exceeds maximum file size (204801 B > 204800 B limit)",
    ]
    categories = {categorize_skip_reason(reason) for reason in reasons}
    assert categories == {"over --max-file-size-kb"}


def test_unexpected_error_reasons_collapse_regardless_of_the_exception():
    a = categorize_skip_reason("unexpected_error: KeyError('language')")
    b = categorize_skip_reason("unexpected_error: ZeroDivisionError()")
    assert a == b == "unexpected error"


def test_unreadable_size_reason_is_categorised():
    assert (
        categorize_skip_reason("cannot determine file size: [Errno 13] denied")
        == "unreadable"
    )


def test_case_is_not_significant():
    assert categorize_skip_reason("Excluded By Pattern") == "excluded by --exclude"


def test_empty_reason_becomes_other():
    assert categorize_skip_reason("") == "other"
    assert categorize_skip_reason(None) == "other"
    assert categorize_skip_reason("   ") == "other"


def test_unknown_reason_keeps_its_own_text():
    """A reason added elsewhere in the pipeline must stay readable."""
    assert categorize_skip_reason("quarantined by policy") == "quarantined by policy"


def test_unknown_reason_with_a_payload_is_cut_at_the_delimiter():
    assert categorize_skip_reason("quarantined: /etc/shadow") == "quarantined"
    assert categorize_skip_reason("odd rule (42 B)") == "odd rule"


def test_every_category_in_the_order_is_reachable():
    produced = {categorize_skip_reason(prefix) for prefix, _ in _prefix_samples()}
    assert set(CATEGORY_ORDER) <= produced


def _prefix_samples():
    return [
        ("excluded by pattern", None),
        ("ignored by default rules", None),
        ("protected large file", None),
        ("exceeds maximum file size (1 B > 0 B limit)", None),
        ("cannot determine file size: x", None),
        ("ignored by gitignore", None),
        ("binary_file", None),
        ("broken symbolic link", None),
        ("symbolic link outside repository", None),
        ("circular or duplicate symbolic link", None),
        ("encoding_error", None),
        ("unexpected_error: x", None),
        ("filtered", None),
    ]


# ---------------------------------------------------------------------------
# group_skip_reasons
# ---------------------------------------------------------------------------


def test_grouping_counts_by_category():
    skipped = [
        ("a.png", "binary_file"),
        ("b.png", "binary_file"),
        ("c.json", "ignored by default rules"),
    ]
    assert group_skip_reasons(skipped) == [
        ("binary content", 2),
        ("ignored by default rules", 1),
    ]


def test_grouping_orders_by_count_then_by_category_order():
    skipped = [
        ("a", "binary_file"),
        ("b", "excluded by pattern"),
        ("c", "ignored by default rules"),
    ]
    # All tied at one, so CATEGORY_ORDER decides: --exclude, defaults, binary.
    assert [category for category, _ in group_skip_reasons(skipped)] == [
        "excluded by --exclude",
        "ignored by default rules",
        "binary content",
    ]


def test_grouping_is_deterministic_across_input_order():
    skipped = [("a", "binary_file"), ("b", "excluded by pattern")]
    assert group_skip_reasons(skipped) == group_skip_reasons(list(reversed(skipped)))


def test_grouping_tolerates_missing_reasons():
    assert group_skip_reasons([("a",)]) == [("other", 1)]


def test_grouping_of_nothing_is_empty():
    assert group_skip_reasons([]) == []
    assert group_skip_reasons(None) == []


def test_skip_summary_lines_are_empty_when_nothing_was_skipped():
    assert build_skip_summary_lines([]) == []


def test_skip_summary_lines_render_each_category_once():
    lines = build_skip_summary_lines(
        [("a", "binary_file"), ("b", "binary_file"), ("c", "excluded by pattern")]
    )
    body = "\n".join(lines)
    assert "Skipped Files Summary" in body
    assert body.count("binary content") == 1
    counts = {
        line.rsplit(":", 1)[0].strip(): line.rsplit(":", 1)[1].strip()
        for line in lines
        if ":" in line and not line.startswith("[")
    }
    assert counts["binary content"] == "2"
    assert counts["excluded by --exclude"] == "1"


# ---------------------------------------------------------------------------
# check_analysis_not_empty
# ---------------------------------------------------------------------------


def test_a_run_with_documents_is_not_empty():
    assert check_analysis_not_empty("/repo", 3, []) is None


def test_a_run_with_documents_is_not_empty_even_when_files_were_skipped():
    assert check_analysis_not_empty("/repo", 3, [("a", "binary_file")]) is None


def test_an_empty_repository_is_reported_as_having_no_files():
    result = check_analysis_not_empty("/repo", 0, [])
    assert result is not None
    assert not result.everything_filtered
    assert "No files were found under /repo" in result.headline


def test_a_fully_filtered_repository_names_the_count():
    skipped = [(f"f{i}.json", "ignored by default rules") for i in range(12)]
    result = check_analysis_not_empty("/repo", 0, skipped)
    assert result.everything_filtered
    assert result.skipped_count == 12
    assert "All 12 file(s)" in result.headline
    assert result.reasons == (("ignored by default rules", 12),)


def test_hints_are_drawn_from_the_categories_that_occurred():
    result = check_analysis_not_empty(
        "/repo",
        0,
        [("a.py", "excluded by pattern"), ("b.py", "exceeds maximum file size (1 B > 0 B limit)")],
    )
    hints = result.hints()
    assert any("--exclude" in hint for hint in hints)
    assert any("--max-file-size-kb" in hint for hint in hints)


def test_no_hint_is_offered_for_a_cause_the_user_cannot_change():
    result = check_analysis_not_empty("/repo", 0, [("a.png", "binary_file")])
    assert result.hints() == []


def test_root_path_is_stringified():
    from pathlib import Path

    result = check_analysis_not_empty(Path("/repo"), 0, [])
    assert isinstance(result.root_path, str)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_rendered_lines_mention_that_nothing_was_written():
    result = EmptyAnalysis(root_path="/repo", skipped_count=0)
    body = "\n".join(build_empty_analysis_lines(result))
    assert "Nothing was written" in body
    assert "--dry-run" in body


def test_dry_run_rendering_does_not_suggest_a_dry_run():
    result = EmptyAnalysis(root_path="/repo", skipped_count=0)
    body = "\n".join(build_empty_analysis_lines(result, suggest_dry_run=False))
    assert "Nothing was written" in body
    assert "--dry-run" not in body


def test_render_uses_the_supplied_printer():
    printed = []
    render_empty_analysis(EmptyAnalysis("/repo", 0), printed.append)
    assert printed and any("No files were found" in line for line in printed)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def _install_fakes(monkeypatch, tmp_path, documents, skipped, calls):
    class FakeRepoLoader:
        def __init__(self, source, *args, **kwargs):
            self.source = source

        def load(self, return_skip_info=False):
            if return_skip_info:
                return documents, str(tmp_path), self, skipped
            return documents, str(tmp_path), self

        def cleanup(self):
            calls["cleanup"] = True

    monkeypatch.setattr(cli_main, "RepoLoader", FakeRepoLoader)
    monkeypatch.setattr(
        cli_main, "setup_api_keys", lambda provider: calls.setdefault("api_keys", True)
    )
    monkeypatch.setattr(
        cli_main,
        "generate_all_summaries",
        lambda documents, summary_cache, **kwargs: (
            calls.setdefault("summarized", True),
            ([{"file_path": "main.py", "description": "d"}], []),
        )[1],
    )
    monkeypatch.setattr(
        cli_main,
        "generate_hierarchical_summaries",
        lambda file_summaries, **kwargs: file_summaries,
    )

    def fake_run_pipeline(summaries, tree, dependency_overview, **kwargs):
        calls["pipeline"] = summaries
        return "# Invented Project\n\nSomething.\n"

    monkeypatch.setattr(cli_main, "run_pipeline", fake_run_pipeline)


def test_empty_repository_stops_before_the_confirmation_prompt(monkeypatch, tmp_path):
    calls = {}
    _install_fakes(monkeypatch, tmp_path, [], [], calls)

    result = CliRunner().invoke(cli_main.main, ["run", "--local", str(tmp_path)])

    assert result.exit_code == 1
    assert "No files were found" in result.output
    assert "Proceed?" not in result.output
    assert "pipeline" not in calls
    assert "api_keys" not in calls


def test_empty_repository_does_not_touch_the_output_file(monkeypatch, tmp_path):
    calls = {}
    output = tmp_path / "README.md"
    output.write_text("MY HAND WRITTEN README", encoding="utf-8")
    _install_fakes(monkeypatch, tmp_path, [], [], calls)

    result = CliRunner().invoke(
        cli_main.main,
        ["run", "--local", str(tmp_path), "--output", str(output), "--force"],
        input="y\ny\n",
    )

    assert result.exit_code == 1
    assert output.read_text(encoding="utf-8") == "MY HAND WRITTEN README"
    assert "pipeline" not in calls


def test_fully_filtered_repository_names_the_rule_responsible(monkeypatch, tmp_path):
    calls = {}
    skipped = [(f"data/f{i}.json", "ignored by default rules") for i in range(7)]
    _install_fakes(monkeypatch, tmp_path, [], skipped, calls)

    result = CliRunner().invoke(cli_main.main, ["run", "--local", str(tmp_path)])

    assert result.exit_code == 1
    assert "All 7 file(s)" in result.output
    assert "ignored by default rules" in result.output
    assert "--include" in result.output


def test_the_temporary_clone_is_cleaned_up_when_the_analysis_is_empty(
    monkeypatch, tmp_path
):
    calls = {}
    _install_fakes(monkeypatch, tmp_path, [], [], calls)

    CliRunner().invoke(cli_main.main, ["run", "--local", str(tmp_path)])

    assert calls.get("cleanup") is True


def test_dry_run_explains_an_empty_analysis_without_failing(monkeypatch, tmp_path):
    calls = {}
    skipped = [("big.bin", "exceeds maximum file size (999 B > 10 B limit)")]
    _install_fakes(monkeypatch, tmp_path, [], skipped, calls)

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--dry-run"]
    )

    assert result.exit_code == 0
    assert "over --max-file-size-kb" in result.output
    assert "--max-file-size-kb" in result.output
    assert "Dry run complete" in result.output


def test_a_run_with_documents_still_reaches_the_pipeline(monkeypatch, tmp_path):
    calls = {}
    documents = [
        type(
            "Doc",
            (),
            {
                "page_content": "print('hi')\n",
                "metadata": {
                    "file_path": str(tmp_path / "main.py"),
                    "relative_path": "main.py",
                    "file_type": ".py",
                    "language": "python",
                },
            },
        )()
    ]
    _install_fakes(monkeypatch, tmp_path, documents, [], calls)

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path)], input="y\n"
    )

    assert result.exit_code == 0
    assert calls.get("pipeline") is not None


def test_dry_run_skip_summary_groups_sizes_into_one_row(monkeypatch, tmp_path):
    """The old breakdown printed one row per distinct byte count."""
    calls = {}
    skipped = [
        ("a.bin", "exceeds maximum file size (317440 B > 204800 B limit)"),
        ("b.bin", "exceeds maximum file size (999999 B > 204800 B limit)"),
    ]
    documents = [
        type(
            "Doc",
            (),
            {
                "page_content": "x\n",
                "metadata": {
                    "file_path": str(tmp_path / "main.py"),
                    "relative_path": "main.py",
                    "file_type": ".py",
                    "language": "python",
                },
            },
        )()
    ]
    _install_fakes(monkeypatch, tmp_path, documents, skipped, calls)

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--dry-run"]
    )

    assert result.exit_code == 0
    assert result.output.count("over --max-file-size-kb") == 1
    assert "317440" not in result.output
