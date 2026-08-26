from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from repo2readme.llm.client_cache import get_or_create
from repo2readme.providers import get_provider


def _missing_package(package: str, provider_label: str) -> ImportError:
    return ImportError(
        f"{provider_label} support requires the '{package}' package. "
        f"Install it with: pip install {package}"
    )


def create_llm(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """
    Return a LangChain chat model for this configuration.

    The client is built once per distinct configuration and reused. It used to
    be built per request - once per file, once per directory roll-up, once per
    iteration of the review loop - and each one carries its own HTTP connection
    pool, so a four-hundred-file repository paid four hundred TLS handshakes for
    work that fits on one keep-alive connection. See
    ``repo2readme.llm.client_cache``.

    Parameters
    ----------
    provider : str
        LLM provider name or alias, as listed in ``repo2readme.providers``.
    model : str | None
        Model name. Falls back to the provider's default model.
    api_key : str | None
        Optional API key. Falls back to the provider's environment variable.
    base_url : str | None
        Optional base URL. Falls back to the provider's default base URL,
        which is what makes the OpenAI-compatible providers work out of the box.

    Raises
    ------
    UnknownProviderError
        If the provider is not in the registry. The message lists every
        supported provider. Raised before anything is imported, and never
        cached.
    """
    return get_or_create(
        lambda: build_llm(
            provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        ),
        provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


def build_llm(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """Construct a chat model, without consulting the cache.

    :func:`create_llm` is the entry point everything else should use. This is
    kept separate so a caller that genuinely needs its own instance - and the
    cache itself - has a way to ask for one.
    """
    spec = get_provider(provider)
    name = spec.name
    model = model or spec.default_model
    base_url = base_url or spec.default_base_url
    resolved_key = api_key or (os.getenv(spec.env_var) if spec.env_var else None)

    if name == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:  # pragma: no cover - depends on install
            raise _missing_package("langchain-groq", spec.label) from exc
        return ChatGroq(model=model, api_key=resolved_key, **kwargs)

    if name == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - depends on install
            raise _missing_package("langchain-google-genai", spec.label) from exc
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=resolved_key,
            **kwargs,
        )

    if name == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - depends on install
            raise _missing_package("langchain-anthropic", spec.label) from exc
        return ChatAnthropic(model=model, api_key=resolved_key, **kwargs)

    if name == "ollama":
        # Ollama runs locally and needs no key, but langchain-ollama is not a
        # hard dependency, so fail with an actionable message.
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise _missing_package("langchain-ollama", spec.label) from exc
        return ChatOllama(model=model, base_url=base_url, **kwargs)

    # openai, openrouter and together all speak the OpenAI wire protocol and
    # differ only by base URL and key.
    if name in ("openai", "openrouter", "together"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depends on install
            raise _missing_package("langchain-openai", spec.label) from exc
        return ChatOpenAI(
            model=model,
            api_key=resolved_key,
            base_url=base_url,
            **kwargs,
        )

    # Unreachable while the registry and this function stay in sync; kept as a
    # guard so a newly registered provider fails loudly instead of silently.
    raise NotImplementedError(
        f"Provider {spec.name!r} is registered but has no factory branch."
    )
