"""Classification of unparseable model answers.

Every chain in the project ends in a parser: ``JsonOutputParser`` on the
summarization and directory roll-up chains, ``PydanticOutputParser`` on the
review chain. When the model returns a truncated object or wraps its answer in a
code fence, those parsers raise, and the next sample usually parses fine - so
these failures have to be classified as retryable.

The subtlety is that ``OutputParserException`` inherits from ``ValueError``,
which the classifier rejects as a programming error. These tests pin the real
class hierarchy rather than a stand-in with the same name.
"""

import json

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError

from repo2readme.providers import UnknownProviderError
from repo2readme.utils.retry import (
    ENV_BASE_DELAY,
    ENV_MAX_RETRIES,
    RetryConfig,
    call_with_retry,
    exception_class_names,
    is_parse_error,
    is_retryable,
)


class VendorParserError(OutputParserException):
    """A provider integration subclassing the parser exception under its own name."""


class Schema(BaseModel):
    score: float


def _validation_error() -> ValidationError:
    try:
        Schema(score="not a number")
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def _json_error() -> json.JSONDecodeError:
    try:
        json.loads("{'not': 'json'")
    except json.JSONDecodeError as exc:
        return exc
    raise AssertionError("expected a JSONDecodeError")


def _no_sleep(_seconds):
    return None


def _config(**kwargs):
    defaults = {"max_retries": 2, "base_delay": 0.0, "jitter": 0.0}
    defaults.update(kwargs)
    return RetryConfig(**defaults)


class TestExceptionClassNames:
    def test_includes_the_class_itself(self):
        assert "outputparserexception" in exception_class_names(
            OutputParserException("boom")
        )

    def test_includes_inherited_classes(self):
        names = exception_class_names(VendorParserError("boom"))
        assert names[0] == "vendorparsererror"
        assert "outputparserexception" in names
        assert "valueerror" in names

    def test_is_lowercased(self):
        assert all(name == name.lower() for name in exception_class_names(TypeError()))


class TestIsParseError:
    @pytest.mark.parametrize(
        "exc",
        [
            OutputParserException("Invalid json output: {\"file_path\":"),
            VendorParserError("Invalid json output"),
            _validation_error(),
            _json_error(),
        ],
        ids=["langchain", "vendor-subclass", "pydantic", "stdlib-json"],
    )
    def test_recognises_parser_failures(self, exc):
        assert is_parse_error(exc)

    @pytest.mark.parametrize(
        "exc",
        [
            TimeoutError("timed out"),
            ConnectionError("connection reset"),
            ValueError("Unsupported provider: nope"),
            UnknownProviderError("vertex"),
            Exception("Error code: 429 - rate limit reached"),
        ],
        ids=["timeout", "connection", "value", "unknown-provider", "rate-limit"],
    )
    def test_ignores_everything_else(self, exc):
        assert not is_parse_error(exc)


class TestClassification:
    def test_parser_failure_is_retryable(self):
        assert is_retryable(OutputParserException("Invalid json output: ```json"))

    def test_vendor_subclass_is_retryable(self):
        assert is_retryable(VendorParserError("could not parse the response"))

    def test_pydantic_validation_failure_is_retryable(self):
        # The review chain parses into a model; a wrong type there is the model
        # answering badly, not a bug in our code.
        assert is_retryable(_validation_error())

    def test_the_model_text_in_the_message_is_not_read_as_an_error(self):
        # A parser exception quotes the answer it could not read. A README that
        # happens to document an auth flow must not be mistaken for an auth
        # failure and marked permanent.
        exc = OutputParserException(
            'Invalid json output: {"description": "handles invalid api key '
            'and unauthorized responses"}'
        )
        assert is_retryable(exc)

    def test_plain_value_errors_are_still_permanent(self):
        assert not is_retryable(ValueError("Unsupported provider: nope"))

    def test_unknown_provider_is_still_permanent(self):
        assert not is_retryable(UnknownProviderError("vertex"))

    def test_other_programming_errors_are_still_permanent(self):
        assert not is_retryable(TypeError("bad argument"))
        assert not is_retryable(KeyError("file_path"))
        assert not is_retryable(ImportError("No module named 'langchain_groq'"))


class TestRetryLoop:
    def test_a_parse_failure_is_retried_and_can_succeed(self):
        calls = {"count": 0}

        def flaky():
            calls["count"] += 1
            if calls["count"] == 1:
                raise OutputParserException("Invalid json output: {")
            return {"file_path": "a.py", "description": "ok"}

        result = call_with_retry(flaky, config=_config(), sleep=_no_sleep)

        assert result == {"file_path": "a.py", "description": "ok"}
        assert calls["count"] == 2

    def test_a_persistent_parse_failure_is_re_raised_after_the_last_attempt(self):
        calls = {"count": 0}

        def always_bad():
            calls["count"] += 1
            raise OutputParserException("Invalid json output: {")

        with pytest.raises(OutputParserException):
            call_with_retry(always_bad, config=_config(max_retries=2), sleep=_no_sleep)

        assert calls["count"] == 3

    def test_retries_can_still_be_switched_off(self):
        calls = {"count": 0}

        def always_bad():
            calls["count"] += 1
            raise OutputParserException("Invalid json output: {")

        with pytest.raises(OutputParserException):
            call_with_retry(always_bad, config=_config(max_retries=0), sleep=_no_sleep)

        assert calls["count"] == 1


class TestSummarizerIntegration:
    def test_the_summarizer_recovers_from_one_unparseable_answer(self, monkeypatch):
        from repo2readme.summarize import summary as summary_module

        state = {"calls": 0}

        class FakeChain:
            def invoke(self, _payload):
                state["calls"] += 1
                if state["calls"] == 1:
                    raise OutputParserException("Invalid json output: {")
                return {"file_path": "a.py", "description": "parsed on retry"}

        monkeypatch.setattr(
            summary_module, "create_summarizer", lambda *a, **k: FakeChain()
        )
        monkeypatch.setenv(ENV_MAX_RETRIES, "1")
        monkeypatch.setenv(ENV_BASE_DELAY, "0")

        result = summary_module.summarize_file("a.py", "python", "x = 1")

        assert result == {"file_path": "a.py", "description": "parsed on retry"}
        assert state["calls"] == 2
        assert "error" not in result

    def test_a_persistently_unparseable_file_still_reports_an_error(self, monkeypatch):
        from repo2readme.summarize import summary as summary_module

        state = {"calls": 0}

        class FakeChain:
            def invoke(self, _payload):
                state["calls"] += 1
                raise OutputParserException("Invalid json output: {")

        monkeypatch.setattr(
            summary_module, "create_summarizer", lambda *a, **k: FakeChain()
        )
        monkeypatch.setenv(ENV_MAX_RETRIES, "1")
        monkeypatch.setenv(ENV_BASE_DELAY, "0")

        result = summary_module.summarize_file("a.py", "python", "x = 1")

        assert result["file_path"] == "a.py"
        assert "error" in result
        assert state["calls"] == 2


class TestReviewerIntegration:
    def test_the_reviewer_recovers_from_one_unparseable_answer(self, monkeypatch):
        # The review chain is prompt | model | PydanticOutputParser, so the fake
        # model returns real message content and the real parser reads it. The
        # first answer is one the parser rejects; the second is valid.
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        from repo2readme.readme import reviewer_agent

        state = {"calls": 0}

        def fake_model(_prompt_value):
            state["calls"] += 1
            if state["calls"] == 1:
                return AIMessage(content="Sure! Here is the review: 9/10, nice work.")
            return AIMessage(content='{"score": 9.0, "feedback": "looks good"}')

        monkeypatch.setattr(
            reviewer_agent, "create_llm", lambda *a, **k: RunnableLambda(fake_model)
        )
        monkeypatch.setenv(ENV_MAX_RETRIES, "1")
        monkeypatch.setenv(ENV_BASE_DELAY, "0")

        review = reviewer_agent.readme_reviewer("# Title")

        assert review.score == 9.0
        assert review.feedback == "looks good"
        assert state["calls"] == 2

    def test_a_persistently_unparseable_review_still_raises(self, monkeypatch):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        from repo2readme.readme import reviewer_agent

        state = {"calls": 0}

        def fake_model(_prompt_value):
            state["calls"] += 1
            return AIMessage(content="I would rate this README about 7 out of 10.")

        monkeypatch.setattr(
            reviewer_agent, "create_llm", lambda *a, **k: RunnableLambda(fake_model)
        )
        monkeypatch.setenv(ENV_MAX_RETRIES, "1")
        monkeypatch.setenv(ENV_BASE_DELAY, "0")

        with pytest.raises(OutputParserException):
            reviewer_agent.readme_reviewer("# Title")

        assert state["calls"] == 2
