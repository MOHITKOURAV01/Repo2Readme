"""``--base-url`` has to reach the client, or be refused.

``create_llm`` resolved a base URL at the top and then never read it again on
the Groq, Google and Anthropic branches::

    base_url = base_url or spec.default_base_url
    ...
    if name == "groq":
        return ChatGroq(model=model, api_key=resolved_key, **kwargs)

So a flag that is declared in the CLI, documented, threaded through four call
sites and keyed into the summary cache had no effect on where the requests went
for three of the six providers. Its one observable effect was invalidating the
cache.

Which keyword a client takes the base URL under is provider knowledge, so it
lives in ``ProviderSpec`` next to ``default_base_url`` and ``env_var``, and every
branch of the factory applies it the same way.
"""

import sys
import types
from typing import ClassVar

import pytest

from repo2readme.providers import (
    PROVIDERS,
    BaseUrlNotSupportedError,
    UnknownProviderError,
    get_provider,
    providers_supporting_base_url,
    resolve_base_url,
    supports_base_url,
)

PROXY = "http://my-proxy:8000/v1"

# Every provider, with the module and class its factory branch imports.
CLIENTS = {
    "groq": ("langchain_groq", "ChatGroq"),
    "google": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic"),
    "openai": ("langchain_openai", "ChatOpenAI"),
    "openrouter": ("langchain_openai", "ChatOpenAI"),
    "together": ("langchain_openai", "ChatOpenAI"),
    "ollama": ("langchain_ollama", "ChatOllama"),
}

BASE_URL_PROVIDERS = sorted(
    name for name in CLIENTS if get_provider(name).supports_base_url
)


@pytest.fixture
def client(monkeypatch):
    """Install a fake chat client and return what it was constructed with."""

    def install(provider):
        module_name, class_name = CLIENTS[provider]
        captured: dict = {}

        module = types.ModuleType(module_name)
        setattr(
            module,
            class_name,
            type(class_name, (), {"__init__": lambda self, **kw: captured.update(kw)}),
        )
        monkeypatch.setitem(sys.modules, module_name, module)
        return captured

    return install


def create(provider, **kwargs):
    from repo2readme.llm import factory

    return factory.create_llm(provider=provider, model="m", api_key="k", **kwargs)


class TestRegistry:
    def test_every_provider_declares_whether_it_takes_a_base_url(self):
        for spec in PROVIDERS:
            assert spec.base_url_param is None or isinstance(spec.base_url_param, str)

    def test_google_is_the_only_provider_without_one(self):
        # Not an oversight: ChatGoogleGenerativeAI is not an OpenAI-protocol
        # client and its endpoint is configured through the transport instead.
        assert [s.name for s in PROVIDERS if not s.supports_base_url] == ["google"]

    def test_a_provider_with_a_default_base_url_must_accept_one(self):
        # A default that could never be applied would be dead configuration.
        for spec in PROVIDERS:
            if spec.default_base_url is not None:
                assert spec.supports_base_url, spec.name

    def test_supports_base_url_accepts_an_alias(self):
        assert supports_base_url("gemini") is False
        assert supports_base_url("claude") is True

    def test_the_supported_list_is_canonical_names(self):
        assert providers_supporting_base_url() == [
            s.name for s in PROVIDERS if s.name != "google"
        ]


class TestResolveBaseUrl:
    def test_an_explicit_value_wins_over_the_default(self):
        assert resolve_base_url("together", PROXY) == PROXY

    def test_the_default_applies_when_nothing_is_given(self):
        assert resolve_base_url("together", None) == "https://api.together.xyz/v1"
        assert resolve_base_url("ollama", None) == "http://localhost:11434"

    def test_a_provider_without_a_default_returns_none(self):
        assert resolve_base_url("groq", None) is None
        assert resolve_base_url("google", None) is None

    def test_an_explicit_value_is_refused_where_it_cannot_be_used(self):
        with pytest.raises(BaseUrlNotSupportedError):
            resolve_base_url("google", PROXY)

    def test_the_refusal_names_the_alternatives(self):
        with pytest.raises(BaseUrlNotSupportedError) as excinfo:
            resolve_base_url("google", PROXY)
        message = str(excinfo.value)
        assert "google" in message
        assert "groq" in message and "openrouter" in message

    def test_the_refusal_explains_the_consequence(self):
        # The point of refusing rather than ignoring is that the traffic would
        # otherwise go somewhere the user did not choose.
        with pytest.raises(BaseUrlNotSupportedError) as excinfo:
            resolve_base_url("gemini", PROXY)
        assert "would be ignored" in str(excinfo.value)

    def test_an_empty_string_is_not_an_explicit_value(self):
        assert resolve_base_url("google", "") is None

    def test_an_unknown_provider_still_raises_its_own_error(self):
        with pytest.raises(UnknownProviderError):
            resolve_base_url("wat", PROXY)


class TestFactoryPassesItThrough:
    @pytest.mark.parametrize("provider", BASE_URL_PROVIDERS)
    def test_an_explicit_base_url_reaches_the_client(self, provider, client):
        captured = client(provider)
        create(provider, base_url=PROXY)
        assert captured.get("base_url") == PROXY

    @pytest.mark.parametrize("provider", ["groq", "anthropic"])
    def test_the_providers_that_used_to_drop_it(self, provider, client):
        # The regression this fixes, pinned by name.
        captured = client(provider)
        create(provider, base_url=PROXY)
        assert "base_url" in captured

    @pytest.mark.parametrize(
        ("provider", "expected"),
        [
            ("together", "https://api.together.xyz/v1"),
            ("openrouter", "https://openrouter.ai/api/v1"),
            ("ollama", "http://localhost:11434"),
        ],
    )
    def test_the_registry_default_is_still_applied(self, provider, expected, client):
        captured = client(provider)
        create(provider)
        assert captured["base_url"] == expected

    @pytest.mark.parametrize("provider", ["groq", "openai", "anthropic"])
    def test_no_base_url_is_passed_when_there_is_none(self, provider, client):
        # Passing base_url=None would override a client's own default, which is
        # not the same as leaving the argument out.
        captured = client(provider)
        create(provider)
        assert "base_url" not in captured

    def test_google_gets_no_base_url_argument(self, client):
        captured = client("google")
        create("google")
        assert "base_url" not in captured
        assert captured["model"] == "m"

    def test_google_refuses_an_explicit_base_url(self, client):
        client("google")
        with pytest.raises(BaseUrlNotSupportedError):
            create("google", base_url=PROXY)

    def test_the_alias_refuses_it_too(self, client):
        client("google")
        with pytest.raises(BaseUrlNotSupportedError):
            create("gemini", base_url=PROXY)

    def test_the_refusal_happens_before_the_client_is_imported(self, monkeypatch):
        # Nothing should be constructed for a call that cannot be honoured.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "langchain_google_genai":
                raise AssertionError("client imported despite an unusable base URL")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(BaseUrlNotSupportedError):
            create("google", base_url=PROXY)

    @pytest.mark.parametrize("provider", sorted(CLIENTS))
    def test_the_model_and_key_are_unaffected(self, provider, client):
        captured = client(provider)
        create(provider)
        assert captured["model"] == "m"

    def test_extra_kwargs_still_reach_the_client(self, client):
        captured = client("groq")
        create("groq", base_url=PROXY, temperature=0.2)
        assert captured["temperature"] == 0.2
        assert captured["base_url"] == PROXY

    def test_an_unknown_provider_raises_before_anything_else(self):
        with pytest.raises(UnknownProviderError):
            create("wat", base_url=PROXY)


class TestNoBranchNamesTheKeywordItself:
    """The durable part: no branch may spell the keyword out again.

    Three of six branches forgot it once. They forgot it because each one wrote
    its own constructor call, so the only thing keeping them in agreement was
    that somebody remembered.
    """

    def test_the_factory_applies_one_resolved_mapping(self):
        import inspect

        from repo2readme.llm import factory

        source = inspect.getsource(factory.create_llm)
        assert source.count("**endpoint") == len(
            [line for line in source.splitlines() if "return Chat" in line]
        )

    def test_no_branch_hardcodes_base_url_as_a_keyword(self):
        import inspect

        from repo2readme.llm import factory

        source = inspect.getsource(factory.create_llm)
        assert "base_url=base_url" not in source


class TestCliRejectsItEarly:
    def _run(self, tmp_path, monkeypatch, args):
        import importlib

        from click.testing import CliRunner

        cli_main = importlib.import_module("repo2readme.cli.main")
        monkeypatch.chdir(tmp_path)

        source = tmp_path / "repo"
        source.mkdir(exist_ok=True)
        (source / "main.py").write_text("print('hello')\n")

        return CliRunner().invoke(
            cli_main.main, ["run", "--local", str(source), *args]
        )

    def test_an_unusable_base_url_stops_the_run(self, tmp_path, monkeypatch):
        result = self._run(
            tmp_path, monkeypatch, ["--provider", "google", "--base-url", PROXY]
        )
        assert result.exit_code == 2
        assert "not supported" in result.output

    def test_it_stops_before_the_repository_is_loaded(self, tmp_path, monkeypatch):
        # The error used to arrive at the first API call, after the clone, the
        # traversal and the confirmation of the token estimate.
        result = self._run(
            tmp_path, monkeypatch, ["--provider", "gemini", "--base-url", PROXY]
        )
        assert result.exit_code == 2
        assert "Loading repository" not in result.output

    def test_a_usable_base_url_is_accepted(self, tmp_path, monkeypatch):
        result = self._run(
            tmp_path,
            monkeypatch,
            ["--provider", "groq", "--base-url", PROXY, "--dry-run"],
        )
        assert result.exit_code == 0

    def test_no_base_url_is_always_fine(self, tmp_path, monkeypatch):
        result = self._run(
            tmp_path, monkeypatch, ["--provider", "google", "--dry-run"]
        )
        assert result.exit_code == 0

    def test_an_unknown_provider_with_a_base_url_is_reported(
        self, tmp_path, monkeypatch
    ):
        result = self._run(
            tmp_path, monkeypatch, ["--provider", "wat", "--base-url", PROXY]
        )
        assert result.exit_code == 2
        assert "wat" in result.output

    def test_without_an_explicit_provider_the_cli_does_not_guess(
        self, tmp_path, monkeypatch
    ):
        # No --provider means the call sites still apply their own defaults;
        # reporting an error about a provider the user never chose would be
        # worse than letting create_llm check the one it is actually handed.
        result = self._run(tmp_path, monkeypatch, ["--base-url", PROXY, "--dry-run"])
        assert result.exit_code == 0


class TestCacheKeyingIsNowHonest:
    """``base_url`` keys the summary cache, so it had better mean something.

    ``_compute_config_hash`` mixes the base URL into the hash that decides
    whether cached summaries are still valid. On Groq that meant changing
    ``--base-url`` re-summarized the entire repository while changing nothing
    about where the requests went: the one observable effect of the flag was the
    one effect it should not have.
    """

    CONFIG: ClassVar[dict] = {"provider": "groq", "model": "openai/gpt-oss-120b"}

    def _cache(self, tmp_path, base_url):
        from repo2readme.cache import SummaryCache

        return SummaryCache(str(tmp_path), {**self.CONFIG, "base_url": base_url}, "h")

    def test_the_base_url_is_part_of_the_cache_key(self, tmp_path):
        # Unchanged behaviour, asserted because the rest of this class depends
        # on it: a different endpoint really is a different configuration.
        first = self._cache(tmp_path, None)._compute_config_hash()
        second = self._cache(tmp_path, PROXY)._compute_config_hash()
        assert first != second

    def test_changing_it_invalidates_the_cache(self, tmp_path):
        original = self._cache(tmp_path, None)
        original.put("a.py", "content", "python", {"description": "d"}, 1.0)
        original.flush()

        moved = self._cache(tmp_path, PROXY)
        assert moved.get("a.py", "content", "python") is None

    def test_and_the_endpoint_now_actually_changes(self, tmp_path, client):
        # The other half of the bargain. Invalidating the cache for a value that
        # is then dropped is pure cost; this asserts the value is not dropped.
        captured = client("groq")
        create("groq", base_url=PROXY)
        assert captured["base_url"] == PROXY

    def test_an_unusable_base_url_never_reaches_the_cache_key(
        self, tmp_path, monkeypatch
    ):
        # The CLI refuses it before the cache is constructed, so google users
        # cannot pay a cache invalidation for a flag that does nothing.
        import importlib

        from click.testing import CliRunner

        cli_main = importlib.import_module("repo2readme.cli.main")
        monkeypatch.chdir(tmp_path)

        source = tmp_path / "repo"
        source.mkdir()
        (source / "main.py").write_text("print('hello')\n")

        result = CliRunner().invoke(
            cli_main.main,
            ["run", "--local", str(source), "--provider", "google",
             "--base-url", PROXY],
        )

        assert result.exit_code == 2
        assert not (tmp_path / ".repo2readme").exists()


class TestFactoryCoversEveryProvider:
    def test_every_registered_provider_can_be_constructed(self, client):
        # The registry and the factory must stay in sync; a provider added
        # without a branch would raise NotImplementedError here.
        for name in sorted(CLIENTS):
            captured = client(name)
            create(name)
            assert captured["model"] == "m"

    def test_the_registry_and_this_test_agree_on_the_provider_list(self):
        assert sorted(CLIENTS) == sorted(spec.name for spec in PROVIDERS)

    def test_a_provider_with_no_branch_fails_loudly(self, monkeypatch):
        from repo2readme.llm import factory
        from repo2readme.providers import ProviderSpec

        phantom = ProviderSpec(
            name="phantom", label="Phantom", default_model="x", env_var=None
        )
        monkeypatch.setattr(factory, "get_provider", lambda _p: phantom)

        with pytest.raises(NotImplementedError):
            create("phantom")

    def test_a_phantom_provider_still_refuses_an_unusable_base_url_first(
        self, monkeypatch
    ):
        from repo2readme.llm import factory
        from repo2readme.providers import ProviderSpec

        phantom = ProviderSpec(
            name="phantom",
            label="Phantom",
            default_model="x",
            env_var=None,
            base_url_param=None,
        )
        monkeypatch.setattr(factory, "get_provider", lambda _p: phantom)

        with pytest.raises(BaseUrlNotSupportedError):
            create("phantom", base_url=PROXY)
