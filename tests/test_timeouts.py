"""Tests for the per-request deadline."""

import importlib
import inspect

import pytest
from click.testing import CliRunner

from repo2readme.llm import factory
from repo2readme.llm.timeouts import (
    DEFAULT_TIMEOUT_SECONDS,
    NO_TIMEOUT,
    README_TIMEOUT_MULTIPLIER,
    InvalidTimeoutError,
    readme_timeout,
    resolve_timeout,
    validate_timeout,
)
from repo2readme.providers import PROVIDERS, get_provider
from repo2readme.utils.retry import is_retryable, is_timeout_error

cli_main = importlib.import_module("repo2readme.cli.main")


# ---------------------------------------------------------------------------
# validate_timeout
# ---------------------------------------------------------------------------


def test_none_means_use_the_default():
    assert validate_timeout(None) is None


@pytest.mark.parametrize("value", [1, 30, 120.5, "45"])
def test_a_usable_value_is_accepted(value):
    assert validate_timeout(value) == float(value)


def test_zero_is_accepted_and_means_no_deadline():
    assert validate_timeout(NO_TIMEOUT) == 0.0


def test_a_negative_value_is_rejected():
    with pytest.raises(InvalidTimeoutError) as excinfo:
        validate_timeout(-1)
    assert "0 or greater" in str(excinfo.value)
    assert "0 for no timeout" in str(excinfo.value)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_value_is_rejected(value):
    with pytest.raises(InvalidTimeoutError):
        validate_timeout(value)


def test_a_non_numeric_value_is_rejected():
    with pytest.raises(InvalidTimeoutError):
        validate_timeout("soon")


# ---------------------------------------------------------------------------
# resolve_timeout
# ---------------------------------------------------------------------------


def test_no_request_uses_the_default():
    assert resolve_timeout(None) == DEFAULT_TIMEOUT_SECONDS


def test_a_request_is_used_as_given():
    assert resolve_timeout(45) == 45.0


def test_zero_disables_the_deadline():
    assert resolve_timeout(0) is None


def test_zero_still_disables_the_deadline_after_scaling():
    """The scaled stages must not resurrect a deadline the user turned off."""
    assert readme_timeout(0) is None


def test_the_readme_stage_gets_longer():
    assert readme_timeout(30) == 30 * README_TIMEOUT_MULTIPLIER
    assert readme_timeout(None) == DEFAULT_TIMEOUT_SECONDS * README_TIMEOUT_MULTIPLIER


def test_the_readme_stage_is_longer_than_a_file_summary():
    assert readme_timeout(None) > resolve_timeout(None)


# ---------------------------------------------------------------------------
# The registry knows each provider's spelling
# ---------------------------------------------------------------------------


def test_every_authenticating_provider_names_its_timeout_argument():
    for spec in PROVIDERS:
        if spec.name == "ollama":
            # Its client takes none; the factory routes through client_kwargs.
            continue
        assert spec.timeout_kwarg, spec.name


@pytest.mark.parametrize(
    "provider, expected",
    [
        ("groq", "request_timeout"),
        ("openai", "request_timeout"),
        ("openrouter", "request_timeout"),
        ("together", "request_timeout"),
        ("google", "timeout"),
        ("anthropic", "default_request_timeout"),
    ],
)
def test_the_registry_matches_the_client(provider, expected):
    assert get_provider(provider).timeout_kwarg == expected


def test_an_alias_resolves_to_the_same_spelling():
    assert get_provider("gemini").timeout_kwarg == get_provider("google").timeout_kwarg


# ---------------------------------------------------------------------------
# _timeout_kwargs
# ---------------------------------------------------------------------------


def test_the_deadline_is_translated_per_provider():
    assert factory._timeout_kwargs(get_provider("groq"), 30, {}) == {
        "request_timeout": 30
    }
    assert factory._timeout_kwargs(get_provider("anthropic"), 30, {}) == {
        "default_request_timeout": 30
    }
    assert factory._timeout_kwargs(get_provider("google"), 30, {}) == {"timeout": 30}


def test_ollama_reaches_its_transport_through_client_kwargs():
    assert factory._timeout_kwargs(get_provider("ollama"), 30, {}) == {
        "client_kwargs": {"timeout": 30}
    }


@pytest.mark.parametrize("value", [None, 0, -5])
def test_no_deadline_adds_no_argument(value):
    assert factory._timeout_kwargs(get_provider("groq"), value, {}) == {}


def test_a_caller_that_set_its_own_timeout_wins():
    existing = {"request_timeout": 9}
    assert factory._timeout_kwargs(get_provider("groq"), 30, existing) == {}


def test_a_caller_that_set_its_own_client_kwargs_wins():
    existing = {"client_kwargs": {"verify": False}}
    assert factory._timeout_kwargs(get_provider("ollama"), 30, existing) == {}


# ---------------------------------------------------------------------------
# create_llm
# ---------------------------------------------------------------------------


@pytest.fixture
def keys(monkeypatch):
    for name in ("GROQ", "OPENAI", "ANTHROPIC", "GOOGLE"):
        monkeypatch.setenv(f"{name}_API_KEY", "x" * 20)


@pytest.mark.parametrize(
    "provider, attribute",
    [
        ("groq", "request_timeout"),
        ("openai", "request_timeout"),
        ("anthropic", "default_request_timeout"),
        ("google", "timeout"),
    ],
)
def test_every_client_is_built_with_a_deadline(provider, attribute, keys):
    """None of them had one, so a hung connection never returned."""
    model = factory.create_llm(provider)
    assert getattr(model, attribute) == DEFAULT_TIMEOUT_SECONDS


def test_an_explicit_deadline_reaches_the_client(keys):
    assert factory.create_llm("groq", timeout=5).request_timeout == 5


def test_a_disabled_deadline_leaves_the_client_alone(keys):
    assert factory.create_llm("groq", timeout=0).request_timeout is None
    assert factory.create_llm("groq", timeout=None).request_timeout is None


def test_an_explicit_keyword_still_wins(keys):
    assert factory.create_llm("groq", request_timeout=7).request_timeout == 7


# ---------------------------------------------------------------------------
# The retry can now see the failure
# ---------------------------------------------------------------------------


class _ReadTimeout(Exception):
    """Stands in for httpx.ReadTimeout, whose message carries only the URL."""


class _APITimeoutError(ConnectionError):
    """Stands in for openai.APITimeoutError, which is not a TimeoutError."""


@pytest.mark.parametrize(
    "exc", [TimeoutError(), _ReadTimeout(""), _APITimeoutError("")]
)
def test_timeout_exceptions_are_recognised(exc):
    assert is_timeout_error(exc) is True


def test_timeout_exceptions_are_retryable(monkeypatch):
    assert is_retryable(_ReadTimeout("")) is True


def test_a_real_httpx_timeout_is_retryable():
    httpx = pytest.importorskip("httpx")
    assert is_retryable(httpx.ReadTimeout("")) is True
    assert is_retryable(httpx.ConnectTimeout("")) is True


def test_ordinary_errors_are_still_not_timeouts():
    assert is_timeout_error(ValueError("bad key")) is False
    assert is_timeout_error(KeyError("language")) is False


# ---------------------------------------------------------------------------
# The value reaches every call site
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module, function",
    [
        ("repo2readme.summarize.summary", "summarize_file"),
        ("repo2readme.summarize.directory_summary", "summarize_directory"),
        ("repo2readme.readme.readme_generator", "generate_readme"),
        ("repo2readme.readme.reviewer_agent", "readme_reviewer"),
        ("repo2readme.services.summarization", "generate_all_summaries"),
        ("repo2readme.services.summarization", "generate_hierarchical_summaries"),
        ("repo2readme.services.orchestrator", "run_pipeline"),
    ],
)
def test_every_stage_accepts_a_timeout(module, function):
    target = getattr(importlib.import_module(module), function)
    assert "timeout" in inspect.signature(target).parameters


def test_the_summarizer_passes_it_to_the_factory(monkeypatch):
    seen = {}

    def fake_create_llm(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr("repo2readme.summarize.summary.create_llm", fake_create_llm)
    from repo2readme.summarize.summary import summarize_file

    summarize_file("a.py", "python", "x = 1", timeout=17)
    assert seen["timeout"] == 17


def test_the_readme_generator_scales_it(monkeypatch):
    seen = {}

    def fake_create_llm(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(
        "repo2readme.readme.readme_generator.create_llm", fake_create_llm
    )
    from repo2readme.readme.readme_generator import generate_readme

    with pytest.raises(RuntimeError):
        generate_readme([], "", [], "", "groq", None, None, timeout=10)
    assert seen["timeout"] == 10 * README_TIMEOUT_MULTIPLIER


def test_the_workflow_carries_it_through_the_graph_state():
    from repo2readme.readme.agent_workflow import ReadmeState

    assert "timeout" in ReadmeState.__annotations__


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_the_flag_is_documented():
    result = CliRunner().invoke(cli_main.main, ["run", "--help"])
    assert "--timeout" in result.output
    assert "0 for no timeout" in result.output


def test_a_negative_timeout_is_rejected_before_the_repository_is_loaded(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        cli_main, "RepoLoader", lambda *a, **k: pytest.fail("repository was loaded")
    )
    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--timeout", "-1"]
    )
    assert result.exit_code == 2
    assert "0 or greater" in result.output


def test_the_flag_reaches_the_summarization_stage(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
    seen = {}

    def fake_generate_all_summaries(documents, summary_cache, **kwargs):
        seen.update(kwargs)
        return [{"file_path": "main.py", "description": "d"}], []

    monkeypatch.setattr(cli_main, "setup_api_keys", lambda provider: None)
    monkeypatch.setattr(
        cli_main, "generate_all_summaries", fake_generate_all_summaries
    )
    monkeypatch.setattr(
        cli_main,
        "generate_hierarchical_summaries",
        lambda file_summaries, **kwargs: file_summaries,
    )
    monkeypatch.setattr(
        cli_main, "run_pipeline", lambda *a, **k: "# readme\n\nbody\n"
    )

    result = CliRunner().invoke(
        cli_main.main, ["run", "--local", str(tmp_path), "--timeout", "42", "--force"]
    )

    assert result.exit_code == 0
    assert seen["timeout"] == 42.0


def test_the_default_is_applied_when_the_flag_is_absent(tmp_path, monkeypatch):
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
    monkeypatch.setattr(
        cli_main, "run_pipeline", lambda *a, **k: "# readme\n\nbody\n"
    )

    CliRunner().invoke(cli_main.main, ["run", "--local", str(tmp_path), "--force"])

    assert seen["timeout"] == DEFAULT_TIMEOUT_SECONDS
