"""Where an API key is looked for, and in what order.

The regression these cover: a key exported in the environment - which is what
``.env`` becomes once ``python-dotenv`` has run - was invisible to
``config.load_env``, so the CLI reported it missing and prompted anyway.
"""

import io
import json

import pytest

from repo2readme import config
from repo2readme.config import (
    ApiKeyError,
    MissingApiKeyError,
    ResolvedApiKey,
    SOURCE_ENVIRONMENT,
    SOURCE_PROMPT,
    SOURCE_STORE,
)
from repo2readme.providers import UnknownProviderError
from repo2readme.services import environment


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the on-disk key store at a temporary file."""
    path = tmp_path / "env.json"
    monkeypatch.setattr(config, "ENV_PATH", str(path))
    return path


@pytest.fixture
def no_prompt(monkeypatch):
    """Fail loudly if anything tries to prompt."""

    def fail(*args, **kwargs):
        raise AssertionError("the run must not prompt for an API key here")

    monkeypatch.setattr(config, "_prompt_for_key", fail)


class TestPrecedence:
    def test_environment_wins_over_the_store(self, store, monkeypatch, no_prompt):
        store.write_text(json.dumps({"GROQ_API_KEY": "from-the-store"}))
        monkeypatch.setenv("GROQ_API_KEY", "from-the-environment")

        resolved = config.resolve_api_key("groq")

        assert resolved.value == "from-the-environment"
        assert resolved.source == SOURCE_ENVIRONMENT
        assert resolved.from_environment

    def test_environment_key_is_used_without_prompting(
        self, store, monkeypatch, no_prompt
    ):
        # The reported bug: no store, key exported, and the CLI prompted.
        monkeypatch.setenv("GROQ_API_KEY", "gsk_exported_key")

        assert config.get_api_key("groq") == "gsk_exported_key"

    def test_environment_key_is_not_copied_into_the_store(
        self, store, monkeypatch, no_prompt
    ):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_exported_key")

        config.resolve_api_key("groq")

        assert not store.exists()

    def test_store_is_used_when_nothing_is_exported(self, store, no_prompt):
        store.write_text(json.dumps({"GROQ_API_KEY": "from-the-store"}))

        resolved = config.resolve_api_key("groq")

        assert resolved.value == "from-the-store"
        assert resolved.source == SOURCE_STORE

    def test_blank_environment_value_falls_through_to_the_store(
        self, store, monkeypatch, no_prompt
    ):
        # An unfilled `.env` line (GROQ_API_KEY=) exports an empty string.
        store.write_text(json.dumps({"GROQ_API_KEY": "from-the-store"}))
        monkeypatch.setenv("GROQ_API_KEY", "   ")

        assert config.resolve_api_key("groq").value == "from-the-store"

    def test_whitespace_is_stripped_from_an_exported_key(
        self, store, monkeypatch, no_prompt
    ):
        monkeypatch.setenv("GROQ_API_KEY", "  gsk_padded_key\n")

        assert config.resolve_api_key("groq").value == "gsk_padded_key"

    def test_alias_reads_the_canonical_variable(self, store, monkeypatch, no_prompt):
        monkeypatch.setenv("GOOGLE_API_KEY", "gemini-secret")

        assert config.get_api_key("gemini") == "gemini-secret"

    def test_unknown_provider_is_rejected(self, store):
        with pytest.raises(UnknownProviderError):
            config.resolve_api_key("nope")


class TestKeylessProviders:
    def test_ollama_needs_no_key(self, store, no_prompt):
        resolved = config.resolve_api_key("ollama")

        assert resolved.value is None
        assert resolved.env_var is None
        assert resolved.source is None


class TestNonInteractive:
    def test_missing_key_without_a_terminal_is_an_error_not_a_prompt(
        self, store, monkeypatch
    ):
        monkeypatch.setattr(config, "can_prompt", lambda stream=None: False)

        with pytest.raises(MissingApiKeyError) as excinfo:
            config.resolve_api_key("groq")

        message = str(excinfo.value)
        assert "GROQ_API_KEY" in message
        assert ".env" in message
        assert excinfo.value.provider == "groq"
        assert excinfo.value.env_var == "GROQ_API_KEY"

    def test_prompting_can_be_declined_by_the_caller(self, store, no_prompt):
        with pytest.raises(MissingApiKeyError):
            config.resolve_api_key("groq", prompt_if_missing=False)

    def test_a_missing_key_error_is_a_value_error(self, store, monkeypatch):
        # The CLI catches ValueError around key setup; keep that working.
        monkeypatch.setattr(config, "can_prompt", lambda stream=None: False)

        with pytest.raises(ValueError):
            config.resolve_api_key("groq")

    @pytest.mark.parametrize(
        "stream, expected",
        [
            (io.StringIO(), False),
            (None, False),
        ],
    )
    def test_can_prompt_rejects_a_non_terminal(self, stream, expected):
        assert config.can_prompt(stream) is expected

    def test_can_prompt_accepts_a_terminal(self):
        class Terminal:
            def isatty(self):
                return True

        assert config.can_prompt(Terminal()) is True

    def test_can_prompt_survives_a_closed_stdin(self):
        stream = io.StringIO()
        stream.close()

        assert config.can_prompt(stream) is False


class TestPrompting:
    def test_a_prompted_key_is_saved_for_next_time(self, store, monkeypatch):
        monkeypatch.setattr(config, "can_prompt", lambda stream=None: True)
        monkeypatch.setattr(config, "_prompt_for_key", lambda label: "gsk_typed_key")

        resolved = config.resolve_api_key("groq")

        assert resolved.source == SOURCE_PROMPT
        assert json.loads(store.read_text())["GROQ_API_KEY"] == "gsk_typed_key"

    def test_saving_a_key_keeps_the_others(self, store, monkeypatch):
        store.write_text(json.dumps({"GOOGLE_API_KEY": "gemini-secret"}))
        monkeypatch.setattr(config, "can_prompt", lambda stream=None: True)
        monkeypatch.setattr(config, "_prompt_for_key", lambda label: "gsk_typed_key")

        config.resolve_api_key("groq")

        saved = json.loads(store.read_text())
        assert saved == {
            "GOOGLE_API_KEY": "gemini-secret",
            "GROQ_API_KEY": "gsk_typed_key",
        }

    def test_a_short_answer_is_rejected(self, store, monkeypatch):
        monkeypatch.setattr(config, "can_prompt", lambda stream=None: True)
        monkeypatch.setattr(
            config, "click", type("c", (), {"prompt": staticmethod(lambda *a, **k: "abc")})
        )

        with pytest.raises(ApiKeyError, match="at least 8 characters"):
            config.resolve_api_key("groq")


class TestStoreReading:
    def test_a_corrupt_store_is_ignored_rather_than_raised(self, store, no_prompt):
        store.write_text("{not json")

        assert config.load_env() == {}

    def test_a_store_that_is_not_an_object_is_ignored(self, store, no_prompt):
        store.write_text(json.dumps(["GROQ_API_KEY"]))

        assert config.load_env() == {}

    def test_a_store_that_is_not_valid_utf8_is_ignored(self, store, no_prompt):
        # UnicodeDecodeError is neither a JSONDecodeError nor an OSError, so it
        # was the one kind of unreadable store that still ended the run.
        store.write_bytes(b'{"GROQ_API_KEY": "\xff\xfe not utf-8"}')

        assert config.load_env() == {}

    def test_a_store_written_here_reads_back(self, store):
        config.save_env({"GROQ_API_KEY": "gsk_key", "NOTE": "caf\u00e9"})

        assert config.load_env()["NOTE"] == "caf\u00e9"

    def test_a_missing_store_reads_as_empty(self, store):
        assert config.load_env() == {}


class TestDefaultProviders:
    def test_both_default_keys_come_from_the_environment(
        self, store, monkeypatch, no_prompt
    ):
        monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
        monkeypatch.setenv("GOOGLE_API_KEY", "gemini-secret")

        assert config.get_api_keys() == ("groq-secret", "gemini-secret")

    def test_a_run_with_one_key_exported_only_prompts_for_the_other(
        self, store, monkeypatch
    ):
        monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
        monkeypatch.setattr(config, "can_prompt", lambda stream=None: True)

        asked = []

        def prompt(label):
            asked.append(label)
            return "gemini-secret"

        monkeypatch.setattr(config, "_prompt_for_key", prompt)

        assert config.get_api_keys() == ("groq-secret", "gemini-secret")
        assert asked == ["Google Gemini"]


class TestSetupApiKeys:
    def test_exported_key_is_left_alone(self, store, monkeypatch, no_prompt):
        import os

        monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
        monkeypatch.setenv("GOOGLE_API_KEY", "gemini-secret")

        environment.setup_api_keys(None)

        assert os.environ["GROQ_API_KEY"] == "groq-secret"
        assert os.environ["GOOGLE_API_KEY"] == "gemini-secret"

    def test_stored_key_is_exported_for_the_sdks(self, store, no_prompt):
        import os

        store.write_text(json.dumps({"TOGETHER_API_KEY": "together-secret"}))

        environment.setup_api_keys("together")

        assert os.environ["TOGETHER_API_KEY"] == "together-secret"

    def test_a_keyless_provider_exports_nothing(self, store, no_prompt):
        import os

        before = dict(os.environ)

        environment.setup_api_keys("ollama")

        assert dict(os.environ) == before

    def test_a_none_key_is_never_assigned_to_the_environment(self, store):
        # os.environ[x] = None raises TypeError; a keyless provider used to
        # take exactly that path.
        assert (
            environment.export_key(
                ResolvedApiKey(
                    provider="ollama", env_var=None, value=None, source=None
                )
            )
            is False
        )

    def test_export_is_a_no_op_when_the_value_is_already_there(
        self, store, monkeypatch
    ):
        monkeypatch.setenv("GROQ_API_KEY", "groq-secret")

        changed = environment.export_key(
            ResolvedApiKey(
                provider="groq",
                env_var="GROQ_API_KEY",
                value="groq-secret",
                source=SOURCE_ENVIRONMENT,
            )
        )

        assert changed is False

    def test_missing_key_without_a_terminal_reaches_the_caller(
        self, store, monkeypatch
    ):
        monkeypatch.setattr(config, "can_prompt", lambda stream=None: False)

        with pytest.raises(MissingApiKeyError):
            environment.setup_api_keys("groq")

    def test_unknown_provider_is_rejected_before_any_key_work(self, store, no_prompt):
        with pytest.raises(UnknownProviderError):
            environment.setup_api_keys("nope")
