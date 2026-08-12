import importlib
import logging

import pytest
from click.testing import CliRunner

from repo2readme import __version__
from repo2readme.utils.logging_config import (
    FILE_LOG_FORMAT,
    NOISY_LOGGERS,
    configure_logging,
    reset_logging,
    resolve_level,
)

cli_main = importlib.import_module("repo2readme.cli.main")


@pytest.fixture(autouse=True)
def clean_logging():
    """Keep the root logger as we found it, handlers and level included."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    original_noisy = {name: logging.getLogger(name).level for name in NOISY_LOGGERS}

    yield

    reset_logging()
    root.handlers[:] = original_handlers
    root.setLevel(original_level)
    for name, level in original_noisy.items():
        logging.getLogger(name).setLevel(level)


def _our_handlers():
    return [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, "_repo2readme_handler", False)
    ]


class TestLevelResolution:
    def test_default_is_warning(self):
        assert resolve_level() == logging.WARNING

    def test_single_v_is_info(self):
        assert resolve_level(verbosity=1) == logging.INFO

    def test_double_v_is_debug(self):
        assert resolve_level(verbosity=2) == logging.DEBUG

    def test_more_than_two_v_stays_debug(self):
        assert resolve_level(verbosity=5) == logging.DEBUG

    def test_quiet_is_error(self):
        assert resolve_level(quiet=True) == logging.ERROR

    def test_quiet_wins_over_verbosity(self):
        assert resolve_level(verbosity=2, quiet=True) == logging.ERROR


class TestConfigureLogging:
    def test_installs_a_console_handler(self):
        level = configure_logging()

        assert level == logging.WARNING
        assert len(_our_handlers()) == 1

    def test_repeated_configuration_does_not_stack_handlers(self):
        for _ in range(5):
            configure_logging(verbosity=1)

        assert len(_our_handlers()) == 1

    def test_foreign_handlers_are_left_alone(self):
        root = logging.getLogger()
        foreign = logging.NullHandler()
        root.addHandler(foreign)
        try:
            configure_logging()
            reset_logging()
            assert foreign in root.handlers
        finally:
            root.removeHandler(foreign)

    def test_verbosity_sets_the_handler_level(self):
        configure_logging(verbosity=2)
        assert _our_handlers()[0].level == logging.DEBUG

    def test_noisy_loggers_are_damped_by_default(self):
        configure_logging(verbosity=1)
        for name in NOISY_LOGGERS:
            assert logging.getLogger(name).level == logging.WARNING

    def test_double_v_unmutes_third_party_loggers(self):
        configure_logging(verbosity=2)
        for name in NOISY_LOGGERS:
            assert logging.getLogger(name).level == logging.DEBUG

    def test_log_file_captures_debug_even_when_console_is_quiet(self, tmp_path):
        log_file = tmp_path / "run.log"

        configure_logging(quiet=True, log_file=str(log_file))
        logging.getLogger("repo2readme.test").debug("a debug detail")

        for handler in _our_handlers():
            handler.flush()

        assert "a debug detail" in log_file.read_text(encoding="utf-8")

    def test_log_file_records_context(self, tmp_path):
        log_file = tmp_path / "run.log"

        configure_logging(log_file=str(log_file))
        logging.getLogger("repo2readme.cache").warning("cache is corrupt")

        for handler in _our_handlers():
            handler.flush()

        contents = log_file.read_text(encoding="utf-8")
        assert "WARNING" in contents
        assert "repo2readme.cache" in contents
        assert "cache is corrupt" in contents

    def test_file_format_includes_time_level_and_logger(self):
        assert "%(asctime)s" in FILE_LOG_FORMAT
        assert "%(levelname)" in FILE_LOG_FORMAT
        assert "%(name)s" in FILE_LOG_FORMAT

    def test_unwritable_log_file_does_not_stop_the_run(self, tmp_path, capsys):
        unwritable = tmp_path / "no-such-dir" / "run.log"

        level = configure_logging(log_file=str(unwritable))

        assert level == logging.WARNING
        assert len(_our_handlers()) == 1  # console handler still installed
        assert "Could not open log file" in capsys.readouterr().err

    def test_warnings_reach_the_console_handler(self, tmp_path):
        log_file = tmp_path / "run.log"
        configure_logging(log_file=str(log_file))

        # The three modules that already log go through the root logger.
        logging.getLogger("repo2readme.summarize.summary").warning(
            "Summary error for a.py: boom"
        )

        for handler in _our_handlers():
            handler.flush()

        assert "Summary error for a.py" in log_file.read_text(encoding="utf-8")


class TestCliIntegration:
    def test_version_flag(self):
        result = CliRunner().invoke(cli_main.main, ["--version"])

        assert result.exit_code == 0
        assert __version__ in result.output

    def test_verbose_flag_is_accepted_and_not_passed_to_the_command(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")

        result = CliRunner().invoke(
            cli_main.main, ["run", "--local", str(tmp_path), "--dry-run", "-vv"]
        )

        assert result.exit_code == 0
        assert "Dry run complete." in result.output

    def test_quiet_flag_is_accepted(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")

        result = CliRunner().invoke(
            cli_main.main, ["run", "--local", str(tmp_path), "--dry-run", "--quiet"]
        )

        assert result.exit_code == 0

    def test_quiet_and_verbose_together_is_a_usage_error(self, tmp_path):
        result = CliRunner().invoke(
            cli_main.main, ["run", "--local", str(tmp_path), "--dry-run", "-v", "-q"]
        )

        assert result.exit_code != 0
        assert "cannot be used together" in result.output

    def test_log_file_option_writes_a_file(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("x = 1", encoding="utf-8")
        log_file = tmp_path / "run.log"

        result = CliRunner().invoke(
            cli_main.main,
            [
                "run",
                "--local",
                str(repo),
                "--dry-run",
                "-vv",
                "--log-file",
                str(log_file),
            ],
        )

        assert result.exit_code == 0
        assert log_file.exists()

    def test_help_documents_the_verbosity_options(self):
        result = CliRunner().invoke(cli_main.main, ["run", "--help"], terminal_width=200)

        assert "--verbose" in result.output
        assert "--quiet" in result.output
        assert "--log-file" in result.output

    def test_repeated_cli_invocations_do_not_stack_handlers(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        runner = CliRunner()

        for _ in range(3):
            runner.invoke(
                cli_main.main, ["run", "--local", str(tmp_path), "--dry-run", "-v"]
            )

        assert len(_our_handlers()) == 1
