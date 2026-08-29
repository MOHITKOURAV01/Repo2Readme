import pytest
from langchain_core.exceptions import (
    OutputParserException as RealOutputParserException,
)

from repo2readme.utils.retry import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_RETRY_AFTER,
    ENV_BASE_DELAY,
    ENV_MAX_RETRIES,
    ENV_MAX_RETRY_AFTER,
    RetryConfig,
    call_with_retry,
    compute_delay,
    is_retryable,
    parse_duration,
    retry_after_seconds,
    status_code_of,
)


class FakeResponse:
    def __init__(self, status_code=None, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class ProviderError(Exception):
    """Stands in for the SDK exceptions, which carry a response object."""

    def __init__(self, message, status_code=None, headers=None):
        super().__init__(message)
        self.response = FakeResponse(status_code, headers)


class OutputParserException(Exception):
    """Same name as LangChain's, which is what the classifier looks at."""


def _no_sleep(_seconds):
    return None


def _config(**kwargs):
    defaults = {"max_retries": 2, "base_delay": 0.0, "jitter": 0.0}
    defaults.update(kwargs)
    return RetryConfig(**defaults)


class TestStatusExtraction:
    def test_reads_status_code_attribute(self):
        exc = type("E", (Exception,), {"status_code": 429})()
        assert status_code_of(exc) == 429

    def test_reads_status_from_response(self):
        assert status_code_of(ProviderError("boom", status_code=503)) == 503

    def test_parses_status_out_of_the_message(self):
        exc = Exception("Error code: 429 - {'error': {'message': 'Rate limit'}}")
        assert status_code_of(exc) == 429

    def test_returns_none_when_there_is_no_status(self):
        assert status_code_of(Exception("something went wrong")) is None

    def test_ignores_out_of_range_values(self):
        exc = type("E", (Exception,), {"code": 99})()
        assert status_code_of(exc) is None


class TestClassification:
    @pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504])
    def test_transient_statuses_are_retryable(self, status):
        assert is_retryable(ProviderError("boom", status_code=status))

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
    def test_client_errors_are_not_retryable(self, status):
        assert not is_retryable(ProviderError("boom", status_code=status))

    def test_builtin_network_errors_are_retryable(self):
        assert is_retryable(TimeoutError("read timed out"))
        assert is_retryable(ConnectionError("connection reset by peer"))

    @pytest.mark.parametrize(
        "message",
        [
            "Rate limit reached for model",
            "Too Many Requests",
            "The server is overloaded, please try again",
            "Service Unavailable",
            "Bad gateway",
            "Connection aborted",
        ],
    )
    def test_transient_messages_are_retryable(self, message):
        assert is_retryable(Exception(message))

    @pytest.mark.parametrize(
        "message",
        [
            "Invalid API key provided",
            "Authentication failed",
            "Unauthorized",
            "Unsupported provider 'vertex'",
            "This model's maximum context length is 8192 tokens",
        ],
    )
    def test_permanent_messages_are_not_retryable(self, message):
        assert not is_retryable(Exception(message))

    def test_configuration_errors_are_never_retried(self):
        assert not is_retryable(ValueError("Unsupported provider: nope"))
        assert not is_retryable(ImportError("No module named 'langchain_groq'"))
        assert not is_retryable(TypeError("bad argument"))

    def test_parser_failures_are_retryable(self):
        assert is_retryable(OutputParserException("Invalid json output: ```json"))

    def test_the_real_langchain_parser_exception_is_retryable(self):
        # The stub above does not inherit from ValueError; the real class does,
        # which is what used to make this classification come out wrong.
        assert is_retryable(RealOutputParserException("Invalid json output: {"))

    def test_unknown_errors_are_not_retried(self):
        # Better to fail fast than to burn three attempts on a real bug.
        assert not is_retryable(Exception("something unusual happened"))

    def test_auth_wins_over_a_retryable_looking_status(self):
        exc = ProviderError("Invalid API key", status_code=401)
        assert not is_retryable(exc)


class TestParseDuration:
    """The duration grammar behind a message hint.

    Providers write these in whatever spelling their own error strings use, so
    the parser is exercised against the shapes that actually appear rather than
    a canonical form.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("20s", 20.0),
            ("6.7s", 6.7),
            ("500ms", 0.5),
            ("2m", 120.0),
            ("1h", 3600.0),
        ],
    )
    def test_reads_a_single_pair(self, text, expected):
        assert parse_duration(text) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Groq reports a per-minute limit in exactly this shape, and it is
            # the case the old single-pair pattern silently read as nothing.
            ("2m59.56s", 179.56),
            ("1m20s", 80.0),
            ("1h30m", 5400.0),
            ("1h30m15s", 5415.0),
        ],
    )
    def test_reads_a_compound_duration(self, text, expected):
        assert parse_duration(text) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("2m 30s", 150.0), ("1 h 30 m", 5400.0), ("500 ms", 0.5)],
    )
    def test_whitespace_between_and_within_pairs_is_allowed(self, text, expected):
        assert parse_duration(text) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("5 minutes", 300.0),
            ("5 mins", 300.0),
            ("1 minute", 60.0),
            ("30 seconds", 30.0),
            ("30 secs", 30.0),
            ("1 second", 1.0),
            ("2 hours", 7200.0),
            ("2 hrs", 7200.0),
            ("250 milliseconds", 0.25),
        ],
    )
    def test_reads_the_spelled_out_units(self, text, expected):
        assert parse_duration(text) == pytest.approx(expected)

    def test_a_long_unit_is_not_read_as_the_short_one(self):
        # "m" ahead of "minutes" in the alternation would match the first
        # letter and leave "inutes" behind, turning five minutes into one.
        assert parse_duration("5 minutes") == 300.0
        assert parse_duration("5 milliseconds") == pytest.approx(0.005)

    def test_stops_at_the_prose_after_the_duration(self):
        assert parse_duration("5 minutes or reduce your request rate") == 300.0
        assert parse_duration("6.7s. Visit the docs for more") == pytest.approx(6.7)

    @pytest.mark.parametrize("text", ["", "soon", "a while", "later, please"])
    def test_returns_none_when_there_is_no_duration(self, text):
        assert parse_duration(text) is None

    def test_a_bare_number_is_not_a_duration(self):
        # Without a unit there is nothing to say what the number means, and
        # guessing seconds would honour a hint the provider never gave.
        assert parse_duration("30") is None

    def test_a_unit_that_begins_a_word_is_rejected(self):
        assert parse_duration("5 monkeys") is None
        assert parse_duration("3 sailors") is None

    def test_zero_is_a_duration_and_not_a_missing_one(self):
        assert parse_duration("0s") == 0.0


class TestRetryAfter:
    def test_reads_the_retry_after_header(self):
        exc = ProviderError("slow down", headers={"retry-after": "7"})
        assert retry_after_seconds(exc) == 7.0

    def test_header_lookup_is_case_insensitive_across_spellings(self):
        exc = ProviderError("slow down", headers={"Retry-After": "3.5"})
        assert retry_after_seconds(exc) == 3.5

    def test_parses_the_hint_groq_puts_in_the_message(self):
        exc = Exception("Rate limit reached. Please try again in 6.7s")
        assert retry_after_seconds(exc) == pytest.approx(6.7)

    def test_understands_millisecond_and_minute_hints(self):
        assert retry_after_seconds(Exception("try again in 500ms")) == 0.5
        assert retry_after_seconds(Exception("try again in 2m")) == 120.0

    def test_returns_none_without_a_hint(self):
        assert retry_after_seconds(Exception("boom")) is None

    def test_ignores_a_header_that_is_neither_a_number_nor_a_date(self):
        # A truncated date is not an HTTP-date, and there is no number in it
        # either, so there is nothing to honour.
        exc = ProviderError("boom", headers={"retry-after": "Wed, 21 Oct 2026"})
        assert retry_after_seconds(exc) is None

    def test_reads_the_compound_hint_the_old_pattern_could_not(self):
        # The wait that matters most: a per-minute limit, reported in the one
        # shape the single value-and-unit pattern silently failed to match.
        exc = Exception(
            "Rate limit reached for model `llama-3.3-70b`. "
            "Please try again in 2m59.56s."
        )
        assert retry_after_seconds(exc) == pytest.approx(179.56)

    def test_reads_a_spelled_out_hint(self):
        exc = Exception("Rate limit exceeded. Please try again in 5 minutes.")
        assert retry_after_seconds(exc) == 300.0

    def test_reads_a_hint_followed_by_prose(self):
        exc = Exception(
            "Please try again in 20s or reduce your request rate. "
            "See https://example.invalid/limits"
        )
        assert retry_after_seconds(exc) == 20.0

    def test_header_wins_over_a_hint_in_the_message(self):
        # Structured, and what the provider's own client reads.
        exc = ProviderError(
            "rate limit, try again in 90s", headers={"retry-after": "12"}
        )
        assert retry_after_seconds(exc) == 12.0

    def test_falls_back_to_the_message_when_the_header_is_unusable(self):
        exc = ProviderError(
            "rate limit, try again in 90s", headers={"retry-after": "soon"}
        )
        assert retry_after_seconds(exc) == 90.0

    def test_reads_an_http_date_header(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime

        deadline = datetime.now(timezone.utc) + timedelta(seconds=45)
        exc = ProviderError(
            "slow down", headers={"retry-after": format_datetime(deadline)}
        )

        # The header carries whole seconds, so the value lands just under the
        # 45 the deadline was built from.
        assert retry_after_seconds(exc) == pytest.approx(45, abs=2)

    def test_an_http_date_already_past_means_no_wait(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime

        past = datetime.now(timezone.utc) - timedelta(seconds=120)
        exc = ProviderError(
            "slow down", headers={"retry-after": format_datetime(past)}
        )

        assert retry_after_seconds(exc) == 0.0

    def test_a_negative_numeric_header_is_ignored(self):
        exc = ProviderError("boom", headers={"retry-after": "-5"})
        assert retry_after_seconds(exc) is None

    def test_a_message_without_the_prefix_is_not_scanned_for_durations(self):
        # "20s" appears, but not as an instruction about when to retry.
        exc = Exception("Request timed out after 20s")
        assert retry_after_seconds(exc) is None

    def test_survives_a_response_without_headers(self):
        exc = ProviderError("boom", status_code=429)
        exc.response.headers = None
        assert retry_after_seconds(exc) is None


class TestDelay:
    def test_backoff_is_exponential(self):
        config = RetryConfig(base_delay=1.0, jitter=0.0, max_delay=100)
        assert compute_delay(0, config) == 1.0
        assert compute_delay(1, config) == 2.0
        assert compute_delay(2, config) == 4.0

    def test_delay_is_capped(self):
        config = RetryConfig(base_delay=10.0, jitter=0.0, max_delay=15.0)
        assert compute_delay(5, config) == 15.0

    def test_jitter_stays_within_bounds(self):
        config = RetryConfig(base_delay=2.0, jitter=0.25, max_delay=100)
        assert compute_delay(0, config, rng=lambda: 0.0) == 2.0
        assert compute_delay(0, config, rng=lambda: 1.0) == 2.5

    def test_provider_hint_overrides_the_computed_backoff(self):
        config = RetryConfig(base_delay=1.0, jitter=0.0, max_delay=100)
        assert compute_delay(0, config, retry_after=9.0) == 9.0

    def test_provider_hint_is_capped_by_its_own_limit(self):
        # max_delay bounds the delays this module invents. The provider's own
        # answer is bounded by max_retry_after instead, so a hint longer than
        # the backoff ceiling is no longer cut down to it.
        config = RetryConfig(
            base_delay=1.0, jitter=0.0, max_delay=5.0, max_retry_after=120.0
        )
        assert compute_delay(0, config, retry_after=60) == 60.0
        assert compute_delay(0, config, retry_after=600) == 120.0


class TestConfigFromEnv:
    def test_defaults_when_unset(self):
        config = RetryConfig.from_env({})
        assert config.max_retries == DEFAULT_MAX_RETRIES
        assert config.base_delay == DEFAULT_BASE_DELAY

    def test_values_are_read_from_the_environment(self):
        config = RetryConfig.from_env(
            {ENV_MAX_RETRIES: "5", ENV_BASE_DELAY: "0.5"}
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5

    def test_zero_retries_is_allowed_and_means_one_attempt(self):
        config = RetryConfig.from_env({ENV_MAX_RETRIES: "0"})
        assert config.max_retries == 0
        assert config.max_attempts == 1

    @pytest.mark.parametrize("value", ["abc", "-1", ""])
    def test_unusable_values_fall_back_to_the_default(self, value):
        config = RetryConfig.from_env({ENV_MAX_RETRIES: value})
        assert config.max_retries == DEFAULT_MAX_RETRIES

    def test_reads_the_real_environment_by_default(self, monkeypatch):
        monkeypatch.setenv(ENV_MAX_RETRIES, "4")
        assert RetryConfig.from_env().max_retries == 4

    def test_max_retry_after_defaults_and_reads_from_the_environment(self):
        assert RetryConfig.from_env({}).max_retry_after == DEFAULT_MAX_RETRY_AFTER
        config = RetryConfig.from_env({ENV_MAX_RETRY_AFTER: "90"})
        assert config.max_retry_after == 90.0

    @pytest.mark.parametrize("value", ["abc", "-1", ""])
    def test_an_unusable_max_retry_after_falls_back_to_the_default(self, value):
        config = RetryConfig.from_env({ENV_MAX_RETRY_AFTER: value})
        assert config.max_retry_after == DEFAULT_MAX_RETRY_AFTER

    def test_zero_max_retry_after_refuses_every_hinted_wait(self):
        # A deliberate "never wait on a provider hint", which is a setting and
        # not a broken value, so it is honoured rather than replaced.
        config = RetryConfig.from_env({ENV_MAX_RETRY_AFTER: "0"})
        assert config.max_retry_after == 0.0

    def test_the_hint_limit_is_independent_of_the_backoff_ceiling(self):
        # The default exists precisely so that a hint is not cut down to the
        # ceiling on the delays this module invents.
        assert DEFAULT_MAX_RETRY_AFTER > RetryConfig().max_delay


class TestCallWithRetry:
    def test_returns_immediately_on_success(self):
        calls = []

        def func():
            calls.append(1)
            return "ok"

        assert call_with_retry(func, config=_config(), sleep=_no_sleep) == "ok"
        assert len(calls) == 1

    def test_retries_a_transient_failure_then_succeeds(self):
        calls = []

        def func():
            calls.append(1)
            if len(calls) < 3:
                raise ProviderError("rate limit", status_code=429)
            return "ok"

        assert call_with_retry(func, config=_config(), sleep=_no_sleep) == "ok"
        assert len(calls) == 3

    def test_reraises_after_exhausting_attempts(self):
        calls = []

        def func():
            calls.append(1)
            raise ProviderError("rate limit", status_code=429)

        with pytest.raises(ProviderError):
            call_with_retry(func, config=_config(max_retries=2), sleep=_no_sleep)

        assert len(calls) == 3

    def test_permanent_failures_are_not_retried(self):
        calls = []

        def func():
            calls.append(1)
            raise ProviderError("Invalid API key", status_code=401)

        with pytest.raises(ProviderError):
            call_with_retry(func, config=_config(), sleep=_no_sleep)

        assert len(calls) == 1

    def test_zero_retries_makes_exactly_one_attempt(self):
        calls = []

        def func():
            calls.append(1)
            raise ProviderError("rate limit", status_code=429)

        with pytest.raises(ProviderError):
            call_with_retry(func, config=_config(max_retries=0), sleep=_no_sleep)

        assert len(calls) == 1

    def test_sleeps_between_attempts_with_growing_delays(self):
        slept = []
        calls = []

        def func():
            calls.append(1)
            raise ProviderError("rate limit", status_code=429)

        config = RetryConfig(max_retries=2, base_delay=1.0, jitter=0.0, max_delay=60)
        with pytest.raises(ProviderError):
            call_with_retry(func, config=config, sleep=slept.append)

        assert slept == [1.0, 2.0]

    def test_honours_the_provider_retry_after_hint(self):
        slept = []
        state = {"calls": 0}

        def func():
            state["calls"] += 1
            if state["calls"] == 1:
                raise ProviderError(
                    "rate limit", status_code=429, headers={"retry-after": "4"}
                )
            return "ok"

        config = RetryConfig(max_retries=2, base_delay=1.0, jitter=0.0, max_delay=60)
        assert call_with_retry(func, config=config, sleep=slept.append) == "ok"
        assert slept == [4.0]

    def test_a_hint_longer_than_the_backoff_ceiling_is_waited_out_in_full(self):
        # The bug this replaces: max_delay cut a 60s hint down to 30s, and the
        # retry landed inside the window the provider had just closed.
        slept = []
        state = {"calls": 0}

        def func():
            state["calls"] += 1
            if state["calls"] == 1:
                raise ProviderError(
                    "rate limit", status_code=429, headers={"retry-after": "60"}
                )
            return "ok"

        config = RetryConfig(
            max_retries=2,
            base_delay=1.0,
            jitter=0.0,
            max_delay=30.0,
            max_retry_after=300.0,
        )
        assert call_with_retry(func, config=config, sleep=slept.append) == "ok"
        assert slept == [60.0]

    def test_stops_when_the_hint_is_longer_than_the_run_will_wait(self):
        slept = []
        calls = []

        def func():
            calls.append(1)
            raise ProviderError(
                "rate limit", status_code=429, headers={"retry-after": "3600"}
            )

        config = RetryConfig(
            max_retries=3, base_delay=1.0, jitter=0.0, max_retry_after=300.0
        )
        with pytest.raises(ProviderError):
            call_with_retry(func, config=config, sleep=slept.append)

        # One attempt, no sleep: every later attempt would land inside the
        # window the provider said it cannot serve.
        assert len(calls) == 1
        assert slept == []

    def test_a_hint_exactly_at_the_limit_is_still_waited_out(self):
        slept = []
        state = {"calls": 0}

        def func():
            state["calls"] += 1
            if state["calls"] == 1:
                raise ProviderError(
                    "rate limit", status_code=429, headers={"retry-after": "300"}
                )
            return "ok"

        config = RetryConfig(
            max_retries=2, base_delay=1.0, jitter=0.0, max_retry_after=300.0
        )
        assert call_with_retry(func, config=config, sleep=slept.append) == "ok"
        assert slept == [300.0]

    def test_a_long_hint_in_the_message_also_stops_the_run(self):
        calls = []

        def func():
            calls.append(1)
            raise ProviderError(
                "Rate limit reached. Please try again in 1h30m.", status_code=429
            )

        config = RetryConfig(
            max_retries=3, base_delay=0.0, jitter=0.0, max_retry_after=300.0
        )
        with pytest.raises(ProviderError):
            call_with_retry(func, config=config, sleep=_no_sleep)

        assert len(calls) == 1

    def test_the_reason_for_stopping_early_is_logged(self, caplog):
        import logging

        def func():
            raise ProviderError(
                "rate limit", status_code=429, headers={"retry-after": "3600"}
            )

        config = RetryConfig(max_retries=3, base_delay=0.0, max_retry_after=300.0)
        with (
            caplog.at_level(logging.DEBUG, logger="repo2readme.utils.retry"),
            pytest.raises(ProviderError),
        ):
            call_with_retry(
                func, config=config, description="summary for a.py", sleep=_no_sleep
            )

        assert "not retrying" in caplog.text
        assert "3600s" in caplog.text

    def test_a_short_hint_does_not_stop_the_run(self):
        state = {"calls": 0}

        def func():
            state["calls"] += 1
            if state["calls"] < 3:
                raise ProviderError(
                    "rate limit", status_code=429, headers={"retry-after": "2"}
                )
            return "ok"

        config = RetryConfig(
            max_retries=3, base_delay=0.0, jitter=0.0, max_retry_after=300.0
        )
        assert call_with_retry(func, config=config, sleep=_no_sleep) == "ok"
        assert state["calls"] == 3

    def test_does_not_sleep_after_the_final_attempt(self):
        slept = []

        def func():
            raise ProviderError("rate limit", status_code=429)

        with pytest.raises(ProviderError):
            call_with_retry(func, config=_config(max_retries=1), sleep=slept.append)

        assert len(slept) == 1

    def test_custom_predicate_is_used(self):
        calls = []

        def func():
            calls.append(1)
            raise RuntimeError("weird")

        with pytest.raises(RuntimeError):
            call_with_retry(
                func,
                config=_config(max_retries=2),
                retryable=lambda exc: True,
                sleep=_no_sleep,
            )

        assert len(calls) == 3

    def test_logs_each_retry(self, caplog):
        import logging

        state = {"calls": 0}

        def func():
            state["calls"] += 1
            if state["calls"] == 1:
                raise ProviderError("rate limit", status_code=429)
            return "ok"

        with caplog.at_level(logging.DEBUG, logger="repo2readme.utils.retry"):
            call_with_retry(
                func, config=_config(), description="summary for a.py", sleep=_no_sleep
            )

        assert "summary for a.py" in caplog.text
        assert "attempt 1/3" in caplog.text


class TestCallSites:
    def test_summarize_file_retries_and_returns_the_summary(self, monkeypatch):
        from repo2readme.summarize import summary as summary_module

        state = {"calls": 0}

        class FakeChain:
            def invoke(self, _payload):
                state["calls"] += 1
                if state["calls"] < 3:
                    raise ProviderError("rate limit", status_code=429)
                return {"file_path": "a.py", "description": "ok"}

        monkeypatch.setattr(
            summary_module, "create_summarizer", lambda *a, **k: FakeChain()
        )
        monkeypatch.setenv(ENV_BASE_DELAY, "0")

        result = summary_module.summarize_file("a.py", "python", "x = 1")

        assert result == {"file_path": "a.py", "description": "ok"}
        assert state["calls"] == 3

    def test_summarize_file_still_returns_an_error_dict_when_retries_run_out(
        self, monkeypatch
    ):
        from repo2readme.summarize import summary as summary_module

        state = {"calls": 0}

        class FakeChain:
            def invoke(self, _payload):
                state["calls"] += 1
                raise ProviderError("rate limit", status_code=429)

        monkeypatch.setattr(
            summary_module, "create_summarizer", lambda *a, **k: FakeChain()
        )
        monkeypatch.setenv(ENV_MAX_RETRIES, "1")
        monkeypatch.setenv(ENV_BASE_DELAY, "0")

        result = summary_module.summarize_file("a.py", "python", "x = 1")

        assert "error" in result
        assert state["calls"] == 2

    def test_permanent_error_is_not_retried_by_the_summarizer(self, monkeypatch):
        from repo2readme.summarize import summary as summary_module

        state = {"calls": 0}

        class FakeChain:
            def invoke(self, _payload):
                state["calls"] += 1
                raise ProviderError("Invalid API key", status_code=401)

        monkeypatch.setattr(
            summary_module, "create_summarizer", lambda *a, **k: FakeChain()
        )
        monkeypatch.setenv(ENV_BASE_DELAY, "0")

        result = summary_module.summarize_file("a.py", "python", "x = 1")

        assert "error" in result
        assert state["calls"] == 1
