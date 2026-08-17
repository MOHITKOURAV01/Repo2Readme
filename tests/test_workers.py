"""What --max-workers means, in one place.

The flag used to be read differently by the two stages that consume it. The
traversal pipeline clamped it to at least one and ignored how much work there
was; the summarization stage treated ``0`` as unset (``max_workers or 4``) and
passed a negative value straight to ``ThreadPoolExecutor``, which rejects it -
part way through the progress bar, after the repository had been loaded and the
token estimate confirmed.
"""

import importlib
from concurrent.futures import ThreadPoolExecutor

import pytest
from click.testing import CliRunner

from repo2readme.loaders.traversal.pipeline import TraversalPipeline
from repo2readme.services.summarization import generate_all_summaries
from repo2readme.utils.workers import (
    DEFAULT_MAX_WORKERS,
    resolve_worker_count,
    validate_max_workers,
)

cli_main = importlib.import_module("repo2readme.cli.main")


class TestValidateMaxWorkers:
    def test_none_is_passed_through(self):
        assert validate_max_workers(None) is None

    def test_a_positive_value_is_accepted(self):
        assert validate_max_workers(8) == 8

    def test_one_is_accepted(self):
        assert validate_max_workers(1) == 1

    def test_zero_is_rejected(self):
        # It used to mean four, because `0 or 4` is 4.
        with pytest.raises(ValueError, match="1 or greater"):
            validate_max_workers(0)

    def test_a_negative_value_is_rejected(self):
        with pytest.raises(ValueError, match="got -1"):
            validate_max_workers(-1)


class TestResolveWorkerCount:
    def test_defaults_when_nothing_was_requested(self):
        assert resolve_worker_count(None, 100) == DEFAULT_MAX_WORKERS

    def test_never_more_workers_than_items(self):
        assert resolve_worker_count(None, 2) == 2
        assert resolve_worker_count(64, 2) == 2

    def test_an_explicit_request_is_honoured(self):
        assert resolve_worker_count(16, 100) == 16

    def test_no_work_still_yields_a_usable_pool(self):
        # ThreadPoolExecutor(max_workers=0) raises, so a pool is never empty.
        assert resolve_worker_count(None, 0) == 1
        assert resolve_worker_count(8, 0) == 1

    def test_a_non_positive_request_is_clamped_rather_than_crashing(self):
        # Defence in depth: the CLI rejects these, but the functions are public.
        assert resolve_worker_count(0, 10) == 1
        assert resolve_worker_count(-5, 10) == 1

    def test_the_default_is_overridable(self):
        assert resolve_worker_count(None, 100, default=2) == 2

    @pytest.mark.parametrize(
        ("requested", "items"),
        [(None, 0), (None, 1), (None, 50), (0, 3), (-2, 3), (1, 3), (99, 3)],
    )
    def test_the_result_is_always_a_valid_pool_size(self, requested, items):
        count = resolve_worker_count(requested, items)

        assert count >= 1
        with ThreadPoolExecutor(max_workers=count) as executor:
            assert executor is not None


class TestBothStagesAgree:
    @pytest.mark.parametrize(
        ("requested", "items"),
        [(None, 0), (None, 3), (None, 10), (1, 10), (4, 2), (64, 2), (8, 8)],
    )
    def test_traversal_and_summarization_resolve_the_same_count(
        self, tmp_path, requested, items
    ):
        pipeline = TraversalPipeline(folder_path=str(tmp_path), max_workers=requested)

        assert pipeline._resolve_worker_count(items) == resolve_worker_count(
            requested, items
        )

    def test_an_explicit_request_is_capped_by_the_file_count_in_traversal(
        self, tmp_path
    ):
        # This used to start 64 threads for a two-file repository.
        pipeline = TraversalPipeline(folder_path=str(tmp_path), max_workers=64)

        assert pipeline._resolve_worker_count(2) == 2


class TestSummarizationStage:
    def _documents(self, count):
        return [
            {
                "content": f"x = {i}",
                "metadata": {"file_path": f"f{i}.py", "file_type": ".py", "mtime": 1.0},
            }
            for i in range(count)
        ]

    class _Cache:
        def get(self, *_args):
            return None

        def put(self, *_args):
            return None

    @pytest.mark.parametrize("max_workers", [None, 1, 2, 8, 0, -3])
    def test_summaries_are_produced_whatever_the_worker_count(
        self, monkeypatch, max_workers
    ):
        from repo2readme.services import summarization

        monkeypatch.setattr(
            summarization,
            "summarize_file",
            lambda file_path, language, content, **_kwargs: {
                "file_path": file_path,
                "description": "ok",
            },
        )

        summaries, errors = generate_all_summaries(
            documents=self._documents(3),
            summary_cache=self._Cache(),
            max_workers=max_workers,
        )

        assert errors == []
        assert len(summaries) == 3

    def test_an_empty_document_list_short_circuits(self):
        summaries, errors = generate_all_summaries(
            documents=[], summary_cache=self._Cache(), max_workers=-1
        )

        assert (summaries, errors) == ([], [])


class TestCliValidation:
    def _invoke(self, tmp_path, value):
        source = tmp_path / "repo"
        source.mkdir()
        (source / "main.py").write_text("print('hi')\n", encoding="utf-8")

        return CliRunner().invoke(
            cli_main.main,
            ["run", "--local", str(source), "--dry-run", "--max-workers", value],
        )

    def test_zero_is_rejected_with_a_message_naming_the_option(self, tmp_path):
        result = self._invoke(tmp_path, "0")

        assert result.exit_code != 0
        assert "--max-workers" in result.output
        assert "1 or greater" in result.output

    def test_a_negative_value_is_rejected(self, tmp_path):
        result = self._invoke(tmp_path, "-1")

        assert result.exit_code != 0
        assert "--max-workers" in result.output

    def test_the_repository_is_not_even_loaded(self, tmp_path, monkeypatch):
        def fail(*_args, **_kwargs):
            raise AssertionError("the repository must not be loaded")

        monkeypatch.setattr(cli_main, "RepoLoader", fail)

        assert self._invoke(tmp_path, "0").exit_code != 0

    def test_a_valid_value_is_accepted(self, tmp_path):
        result = self._invoke(tmp_path, "2")

        assert result.exit_code == 0
        assert "Dry run complete" in result.output

    def test_the_flag_can_still_be_omitted(self, tmp_path):
        source = tmp_path / "repo"
        source.mkdir()
        (source / "main.py").write_text("print('hi')\n", encoding="utf-8")

        result = CliRunner().invoke(
            cli_main.main, ["run", "--local", str(source), "--dry-run"]
        )

        assert result.exit_code == 0
