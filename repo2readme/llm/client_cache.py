"""One chat client per configuration, instead of one per request.

``create_llm`` was called on every request: once per file in ``summarize_file``,
once per directory in ``summarize_directory``, and once per iteration of the
review loop in ``generate_readme`` and ``readme_reviewer``. Twenty-five files
built twenty-five clients for one configuration; a four-hundred-file repository
built four hundred, from ``--max-workers`` threads at once.

``ChatGroq``, ``ChatOpenAI``, ``ChatAnthropic`` and ``ChatGoogleGenerativeAI``
are not thin wrappers. Constructing one builds the provider SDK client, which
builds an ``httpx.Client`` with its own connection pool and TLS context. Per
request that means a fresh pool - so the handshake to the provider is repeated
for every call instead of being amortised over a keep-alive connection - full
Pydantic validation of the model configuration, and a pool's worth of sockets
held until the garbage collector catches up.

Nothing in the configuration varies between those calls. ``provider``, ``model``
and ``base_url`` come from the command line and are fixed for the whole run,
which the summary cache already relies on: it hashes exactly those three values
to decide whether its entries are still valid.

A chat model is stateless with respect to a request - LangChain treats a
``BaseChatModel`` as a reusable Runnable - so one instance can serve the whole
run. This module keys them on the *resolved* configuration, so ``--provider
gemini`` and ``--provider google`` share a client, and so does ``--model
gemini-2.5-flash`` with no ``--model`` at all.

What is deliberately not cached:

* a failed construction. An unknown provider or a missing package raises, and
  the exception leaves nothing behind, so the next call tries again and reports
  the same thing.
* a call whose keyword arguments cannot be hashed. Rather than guess at
  equality, those bypass the cache and build a client the old way.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Callable

from repo2readme.providers import get_provider

# Length of the API key fingerprint kept in a cache key. The key itself is never
# stored here - a rotated key has to produce a different client, and that is all
# the fingerprint is for.
_FINGERPRINT_LENGTH = 16

_lock = threading.Lock()
_clients: dict[tuple, Any] = {}
_stats = {"hits": 0, "misses": 0, "uncacheable": 0}


def _fingerprint(api_key: str | None) -> str:
    """A stable, non-reversible marker for an API key."""
    if not api_key:
        return ""
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return digest[:_FINGERPRINT_LENGTH]


def _kwargs_signature(kwargs: dict[str, Any]) -> tuple | None:
    """A hashable stand-in for ``kwargs``, or None when there is not one.

    Most callers pass nothing here. The ones that do pass scalars - a
    temperature, a timeout - which hash fine. Anything else (a callback list, a
    client object) is reported as uncacheable rather than compared by identity,
    which would key on an object that is rebuilt per call and never hit.
    """
    if not kwargs:
        return ()

    signature = tuple(sorted(kwargs.items()))
    try:
        hash(signature)
    except TypeError:
        return None
    return signature


def cache_key(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> tuple | None:
    """The key a client is stored under, or None when it cannot be cached.

    The provider name, model and base URL are resolved first, so every spelling
    of the same configuration lands on one entry.

    Raises
    ------
    UnknownProviderError
        If the provider is not in the registry. Raised here so an unusable
        provider fails before anything is imported, exactly as it did before.
    """
    spec = get_provider(provider)

    signature = _kwargs_signature(kwargs)
    if signature is None:
        return None

    return (
        spec.name,
        model or spec.default_model,
        base_url or spec.default_base_url,
        _fingerprint(api_key),
        signature,
    )


def get_or_create(
    build: Callable[[], Any],
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Any:
    """Return the client for this configuration, building it once if needed.

    ``build`` is called at most once per key. The lock is held across the call
    so concurrent workers share one client rather than racing to build several;
    it is contended once per configuration, not once per request.
    """
    key = cache_key(
        provider, model=model, api_key=api_key, base_url=base_url, **kwargs
    )

    if key is None:
        with _lock:
            _stats["uncacheable"] += 1
        return build()

    with _lock:
        client = _clients.get(key)
        if client is not None:
            _stats["hits"] += 1
            return client

        # Built under the lock, so a second caller waits for the first client
        # rather than building its own. A failure propagates and stores nothing.
        client = build()
        _clients[key] = client
        _stats["misses"] += 1
        return client


def clear_client_cache() -> None:
    """Forget every client and reset the counters.

    Needed by tests, and by any long-lived caller that changes configuration -
    a rotated key or a different model should not be served from an instance
    built for the previous one.
    """
    with _lock:
        _clients.clear()
        for name in _stats:
            _stats[name] = 0


def client_cache_stats() -> dict[str, int]:
    """Hits, misses and calls that could not be cached, for this process."""
    with _lock:
        return dict(_stats)


def cached_client_count() -> int:
    """How many distinct clients are currently held."""
    with _lock:
        return len(_clients)
