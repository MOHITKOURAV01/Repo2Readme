"""A failed directory roll-up must not take its contents with it.

``summarize_directory`` does not raise. Like ``summarize_file`` it logs and
returns a ``{"file_path": ..., "error": ...}`` placeholder - but that placeholder
*is* the directory's summary, so returning it discards every summary underneath
the directory, and everything underneath those, all the way down to the leaves.
"""

from unittest.mock import patch

import pytest

from repo2readme.services.reporting import (
    SummaryFailure,
    build_rollup_report_lines,
    render_rollup_report,
)
from repo2readme.services.summarization import (
    ROLLUP_THRESHOLD,
    generate_hierarchical_summaries,
)

TARGET = "repo2readme.summarize.directory_summary.summarize_directory"


def files(count, directory="pkg/sub"):
    """Enough file summaries to push the run past the roll-up threshold."""
    return [
        {"file_path": f"{directory}/f{index}.py", "description": f"file {index}"}
        for index in range(count)
    ]


def failing(error="Error code: 429 - rate limit reached"):
    def summarize(dir_path, contents_summaries, **kwargs):
        return {"file_path": dir_path, "error": error}

    return summarize


def succeeding():
    def summarize(dir_path, contents_summaries, **kwargs):
        return {
            "file_path": dir_path,
            "description": f"{len(contents_summaries)} items in {dir_path}",
        }

    return summarize


def only_fails_at(path):
    def summarize(dir_path, contents_summaries, **kwargs):
        if dir_path == path:
            return {"file_path": dir_path, "error": "rate limit"}
        return {
            "file_path": dir_path,
            "description": f"{len(contents_summaries)} items in {dir_path}",
        }

    return summarize


class TestFailedRollupKeepsItsContents:
    def test_every_file_summary_survives(self):
        source = files(20)

        with patch(TARGET, failing()):
            summaries, _ = generate_hierarchical_summaries(source)

        assert summaries == source

    def test_the_error_placeholder_never_reaches_the_prompt(self):
        with patch(TARGET, failing()):
            summaries, _ = generate_hierarchical_summaries(files(20))

        assert not any("error" in summary for summary in summaries)

    def test_the_failure_is_reported(self):
        with patch(TARGET, failing()):
            _, failures = generate_hierarchical_summaries(files(20))

        assert [failure.file_path for failure in failures] == ["pkg/sub", "pkg"]
        assert all("429" in failure.reason for failure in failures)

    def test_a_failure_is_a_summary_failure(self):
        with patch(TARGET, failing()):
            _, failures = generate_hierarchical_summaries(files(20))

        assert all(isinstance(failure, SummaryFailure) for failure in failures)

    def test_one_failed_directory_does_not_stop_the_others(self):
        source = files(10, "alpha") + files(10, "beta")

        with patch(TARGET, only_fails_at("alpha")):
            summaries, failures = generate_hierarchical_summaries(source)

        described = {summary.get("file_path") for summary in summaries}
        assert [failure.file_path for failure in failures] == ["alpha"]
        # beta was condensed; alpha's ten files came through as themselves.
        assert "beta" in described
        assert "alpha/f0.py" in described

    def test_a_parent_can_still_condense_what_a_failed_child_handed_up(self):
        source = files(20, "pkg/sub")

        with patch(TARGET, only_fails_at("pkg/sub")):
            summaries, failures = generate_hierarchical_summaries(source)

        assert [failure.file_path for failure in failures] == ["pkg/sub"]
        assert [summary["file_path"] for summary in summaries] == ["pkg"]


class TestSuccessfulRollupIsUnchanged:
    def test_a_working_rollup_still_condenses(self):
        with patch(TARGET, succeeding()):
            summaries, failures = generate_hierarchical_summaries(files(20))

        assert failures == []
        # pkg holds one thing - the summary of pkg/sub - so it is passed up
        # rather than summarized again.
        assert [summary["file_path"] for summary in summaries] == ["pkg/sub"]
        assert summaries[0]["description"] == "20 items in pkg/sub"

    def test_a_small_repository_skips_the_rollup(self):
        source = files(ROLLUP_THRESHOLD)

        def never(*args, **kwargs):
            raise AssertionError("no roll-up below the threshold")

        with patch(TARGET, never):
            summaries, failures = generate_hierarchical_summaries(source)

        assert summaries == source
        assert failures == []

    def test_summaries_without_a_path_do_not_become_a_bare_none(self):
        # build_directory_tree skips entries with no file_path, so the tree can
        # come back empty. Returning [None] puts a null in the README prompt.
        source = [{"description": f"nameless {index}"} for index in range(20)]

        with patch(TARGET, succeeding()):
            summaries, failures = generate_hierarchical_summaries(source)

        assert summaries == []
        assert failures == []


class TestProgressReporting:
    def test_progress_advances_for_a_failed_directory_too(self):
        class Progress:
            def __init__(self):
                self.advanced = 0

            def update(self, task_id, **kwargs):
                self.advanced += kwargs.get("advance", 0)

        progress = Progress()

        with patch(TARGET, failing()):
            generate_hierarchical_summaries(
                files(20), progress=progress, task_id=1
            )

        # pkg/sub and pkg: a bar that stops moving on the first failure looks
        # like a hang.
        assert progress.advanced == 2


class TestRollupReport:
    def test_nothing_is_printed_when_nothing_failed(self):
        assert build_rollup_report_lines([]) == []

    def test_the_report_names_the_directories(self):
        lines = build_rollup_report_lines(
            [
                SummaryFailure(file_path="pkg/sub", reason="rate limit"),
                SummaryFailure(file_path="pkg", reason="rate limit"),
            ]
        )
        text = "\n".join(lines)

        assert "Directories not condensed: 2" in text
        assert "pkg/sub" in text
        assert "nothing was lost" in text

    def test_identical_reasons_are_grouped(self):
        lines = build_rollup_report_lines(
            [
                SummaryFailure(file_path=f"dir{index}", reason="rate limit")
                for index in range(3)
            ]
        )

        assert sum(1 for line in lines if "rate limit" in line) == 1

    def test_long_lists_are_truncated(self):
        lines = build_rollup_report_lines(
            [
                SummaryFailure(file_path=f"dir{index}", reason="rate limit")
                for index in range(9)
            ],
            max_paths_per_group=5,
        )

        assert "    ... and 4 more" in lines

    def test_render_uses_the_printer(self):
        printed = []
        render_rollup_report(
            [SummaryFailure(file_path="pkg", reason="rate limit")], printed.append
        )

        assert any("Directory summary report" in line for line in printed)

    def test_render_prints_nothing_on_a_clean_run(self):
        printed = []
        render_rollup_report([], printed.append)

        assert printed == []


class TestCliSurfacesRollupFailures:
    @pytest.fixture
    def cli(self):
        import importlib

        return importlib.import_module("repo2readme.cli.main")

    def test_the_run_reports_a_failed_rollup(self, cli, monkeypatch, tmp_path):
        from click.testing import CliRunner

        source = tmp_path / "repo"
        source.mkdir()
        (source / "main.py").write_text("print('hello')\n")

        monkeypatch.setattr(cli, "setup_api_keys", lambda provider: None)
        monkeypatch.setattr(
            cli,
            "generate_all_summaries",
            lambda **kwargs: ([{"file_path": "main.py", "description": "x"}], []),
        )
        monkeypatch.setattr(
            cli,
            "generate_hierarchical_summaries",
            lambda **kwargs: (
                [{"file_path": "main.py", "description": "x"}],
                [SummaryFailure(file_path="pkg", reason="rate limit")],
            ),
        )
        monkeypatch.setattr(cli, "run_pipeline", lambda **kwargs: "# Readme\n")

        result = CliRunner().invoke(
            cli.main, ["run", "--local", str(source), "--force"]
        )

        assert result.exit_code == 0
        assert "Directory summary report" in result.output
        assert "pkg" in result.output
