"""One chat client per configuration, not one per request.

From the issue: create_llm was called once per file, once per directory roll-up
and once per iteration of the review loop. Twenty-five files built twenty-five
clients for one configuration, each with its own httpx connection pool and TLS
context, from --max-workers threads at once.
"""

import sys
import threading

import pytest
from langchain_core.runnables import RunnableLambda

from repo2readme.llm import factory
from repo2readme.llm.client_cache import (
    cache_key,
    cached_client_count,
    clear_client_cache,
    client_cache_stats,
    get_or_create,
)
from repo2readme.providers import UnknownProviderError


class Built:
    """Counts how many times a client was actually constructed."""

    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return object()


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def test_an_alias_and_its_provider_share_a_key():
    assert cache_key("gemini") == cache_key("google")


def test_a_default_model_and_the_same_model_spelled_out_share_a_key():
    assert cache_key("google", model="gemini-2.5-flash") == cache_key("google")


def test_a_default_base_url_and_the_same_url_spelled_out_share_a_key():
    assert cache_key("together", base_url="https://api.together.xyz/v1") == cache_key(
        "together"
    )


@pytest.mark.parametrize(
    "left,right",
    [
        (dict(provider="groq"), dict(provider="google")),
        (dict(provider="groq", model="a"), dict(provider="groq", model="b")),
        (
            dict(provider="openrouter", base_url="http://one"),
            dict(provider="openrouter", base_url="http://two"),
        ),
        (
            dict(provider="groq", api_key="first"),
            dict(provider="groq", api_key="second"),
        ),
        (
            dict(provider="groq", temperature=0.0),
            dict(provider="groq", temperature=0.7),
        ),
    ],
)
def test_different_configurations_get_different_keys(left, right):
    assert cache_key(**left) != cache_key(**right)


def test_the_api_key_is_not_stored_in_the_key():
    key = cache_key("groq", api_key="sk-super-secret-value")
    assert "sk-super-secret-value" not in repr(key)


def test_keyword_order_does_not_matter():
    assert cache_key("groq", temperature=0.0, top_p=1.0) == cache_key(
        "groq", top_p=1.0, temperature=0.0
    )


def test_an_unknown_provider_raises_rather_than_producing_a_key():
    with pytest.raises(UnknownProviderError):
        cache_key("wat")


def test_unhashable_keyword_arguments_have_no_key():
    assert cache_key("groq", callbacks=["a", "b"]) is None


# ---------------------------------------------------------------------------
# get_or_create
# ---------------------------------------------------------------------------


def test_the_same_configuration_returns_the_same_object():
    build = Built()

    first = get_or_create(build, "groq", model="m")
    second = get_or_create(build, "groq", model="m")

    assert first is second
    assert build.count == 1


def test_twenty_five_calls_build_one_client():
    build = Built()

    clients = [get_or_create(build, "groq", model="m") for _ in range(25)]

    assert build.count == 1
    assert len({id(client) for client in clients}) == 1


def test_an_alias_reuses_the_clients_of_its_provider():
    build = Built()

    first = get_or_create(build, "google")
    second = get_or_create(build, "gemini")

    assert first is second
    assert build.count == 1


def test_different_configurations_get_different_clients():
    build = Built()

    a = get_or_create(build, "groq", model="one")
    b = get_or_create(build, "groq", model="two")

    assert a is not b
    assert build.count == 2
    assert cached_client_count() == 2


def test_a_failed_construction_is_not_cached():
    attempts = {"n": 0}

    def explode():
        attempts["n"] += 1
        raise ImportError("pip install langchain-groq")

    for _ in range(3):
        with pytest.raises(ImportError):
            get_or_create(explode, "groq")

    assert attempts["n"] == 3
    assert cached_client_count() == 0


def test_an_unknown_provider_never_reaches_the_builder():
    build = Built()

    with pytest.raises(UnknownProviderError):
        get_or_create(build, "wat")

    assert build.count == 0


def test_uncacheable_arguments_build_every_time():
    build = Built()

    first = get_or_create(build, "groq", callbacks=["a"])
    second = get_or_create(build, "groq", callbacks=["a"])

    assert first is not second
    assert build.count == 2
    assert client_cache_stats()["uncacheable"] == 2
    assert cached_client_count() == 0


def test_concurrent_callers_share_one_client():
    """The whole point: --max-workers threads must not build one each."""
    barrier = threading.Barrier(16)
    build = Built()
    lock = threading.Lock()
    results = []

    def counted_build():
        with lock:
            build.count += 1
        return "the client"

    def worker():
        barrier.wait()
        results.append(get_or_create(counted_build, "groq", model="m"))

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 16
    assert build.count == 1
    assert cached_client_count() == 1


def test_clearing_forces_a_rebuild():
    build = Built()

    first = get_or_create(build, "groq")
    clear_client_cache()
    second = get_or_create(build, "groq")

    assert first is not second
    assert build.count == 2


def test_stats_count_hits_and_misses():
    build = Built()

    get_or_create(build, "groq")
    get_or_create(build, "groq")
    get_or_create(build, "google")

    stats = client_cache_stats()
    assert stats["misses"] == 2
    assert stats["hits"] == 1
    assert stats["uncacheable"] == 0


def test_clearing_resets_the_counters():
    get_or_create(Built(), "groq")
    clear_client_cache()

    assert client_cache_stats() == {"hits": 0, "misses": 0, "uncacheable": 0}


# ---------------------------------------------------------------------------
# Through create_llm
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_openai(monkeypatch):
    """Records every construction, and is a real Runnable so chains can be built."""
    built = []

    class FakeChatOpenAI(RunnableLambda):
        def __init__(self, **kwargs):
            super().__init__(lambda value: '{"file_path": "f", "description": "d"}')
            built.append(kwargs)
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules, "langchain_openai", type("mod", (), {"ChatOpenAI": FakeChatOpenAI})
    )
    return built


def test_create_llm_reuses_one_client_across_many_calls(fake_openai):
    clients = [
        factory.create_llm(provider="openai", api_key="k") for _ in range(25)
    ]

    assert len(fake_openai) == 1
    assert all(client is clients[0] for client in clients)


def test_create_llm_still_builds_per_distinct_model(fake_openai):
    factory.create_llm(provider="openai", model="a", api_key="k")
    factory.create_llm(provider="openai", model="b", api_key="k")
    factory.create_llm(provider="openai", model="a", api_key="k")

    assert [kwargs["model"] for kwargs in fake_openai] == ["a", "b"]


def test_create_llm_separates_clients_by_api_key(fake_openai):
    first = factory.create_llm(provider="openai", api_key="one")
    second = factory.create_llm(provider="openai", api_key="two")

    assert first is not second
    assert [kwargs["api_key"] for kwargs in fake_openai] == ["one", "two"]


def test_create_llm_still_resolves_the_default_base_url(fake_openai, monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "together-key")

    client = factory.create_llm(provider="together")

    assert client.kwargs["base_url"] == "https://api.together.xyz/v1"
    assert client.kwargs["api_key"] == "together-key"


def test_build_llm_bypasses_the_cache(fake_openai):
    first = factory.build_llm(provider="openai", api_key="k")
    second = factory.build_llm(provider="openai", api_key="k")

    assert first is not second
    assert len(fake_openai) == 2
    assert cached_client_count() == 0


def test_create_llm_still_raises_for_an_unknown_provider():
    with pytest.raises(UnknownProviderError):
        factory.create_llm(provider="wat", model="x")


def test_summarizing_twenty_five_files_builds_one_client(fake_openai):
    """The shape the issue measured: 25 files, 25 clients, one configuration."""
    from repo2readme.summarize.summary import create_summarizer

    for index in range(25):
        create_summarizer(f"f{index}.py", "python", "code", provider="openai")

    assert len(fake_openai) == 1


def test_a_whole_summarization_run_builds_one_client(fake_openai):
    from repo2readme.summarize.summary import summarize_file

    results = [
        summarize_file(f"f{index}.py", "python", "code", provider="openai")
        for index in range(25)
    ]

    assert len(fake_openai) == 1
    assert all("error" not in result for result in results)


def test_the_review_loop_shares_the_summarizer_client(fake_openai):
    """Same provider and model across call sites means one client for the run."""
    from repo2readme.summarize.summary import create_summarizer

    create_summarizer("f.py", "python", "code", provider="openai", model_name="m")
    factory.create_llm(provider="openai", model="m")

    assert len(fake_openai) == 1
