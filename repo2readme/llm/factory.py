from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from repo2readme.providers import (
    BaseUrlNotSupportedError,
    ProviderSpec,
    get_provider,
    providers_supporting_base_url,
)


def _missing_package(package: str, provider_label: str) -> ImportError:
    return ImportError(
        f"{provider_label} support requires the '{package}' package. "
        f"Install it with: pip install {package}"
    )


def _base_url_kwargs(
    spec: ProviderSpec, base_url: str | None, explicit: bool
) -> dict[str, str]:
    """The base URL argument for this provider's client, if it takes one.

    Every branch below builds its client with ``**_base_url_kwargs(...)`` rather
    than naming the keyword itself. Three of them used to omit it: ``base_url``
    was resolved at the top of :func:`create_llm` and then never read again on
    the Groq, Google and Anthropic paths, so a flag that is declared in the CLI,
    documented, threaded through four call sites and keyed into the summary
    cache had no effect on where the requests went.

    ``explicit`` distinguishes a value the user asked for from the registry
    default. A provider with no base URL option and no default has nothing to
    refuse when the user did not ask for anything.
    """
    if spec.base_url_param is None:
        if explicit:
            raise BaseUrlNotSupportedError(
                spec.name, providers_supporting_base_url()
            )
        return {}

    if base_url is None:
        return {}

    return {spec.base_url_param: base_url}


def create_llm(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """
    Factory function to create a LangChain chat model.

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
        Passed to the client under the keyword named by the provider's
        ``base_url_param``.

    Raises
    ------
    UnknownProviderError
        If the provider is not in the registry. The message lists every
        supported provider.
    BaseUrlNotSupportedError
        If ``base_url`` is given for a provider whose client takes no base URL.
    """

    spec = get_provider(provider)
    name = spec.name
    model = model or spec.default_model
    explicit_base_url = base_url is not None
    base_url = base_url or spec.default_base_url
    resolved_key = api_key or (os.getenv(spec.env_var) if spec.env_var else None)

    # Resolved once, applied by every branch. Naming the keyword per branch is
    # what let three of them forget it.
    endpoint = _base_url_kwargs(spec, base_url, explicit_base_url)

    if name == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:  # pragma: no cover - depends on install
            raise _missing_package("langchain-groq", spec.label) from exc
        return ChatGroq(model=model, api_key=resolved_key, **endpoint, **kwargs)

    if name == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - depends on install
            raise _missing_package("langchain-google-genai", spec.label) from exc
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=resolved_key,
            **endpoint,
            **kwargs,
        )

    if name == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - depends on install
            raise _missing_package("langchain-anthropic", spec.label) from exc
        return ChatAnthropic(
            model=model, api_key=resolved_key, **endpoint, **kwargs
        )

    if name == "ollama":
        # Ollama runs locally and needs no key, but langchain-ollama is not a
        # hard dependency, so fail with an actionable message.
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise _missing_package("langchain-ollama", spec.label) from exc
        return ChatOllama(model=model, **endpoint, **kwargs)

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
            **endpoint,
            **kwargs,
        )

    # Unreachable while the registry and this function stay in sync; kept as a
    # guard so a newly registered provider fails loudly instead of silently.
    raise NotImplementedError(
        f"Provider {spec.name!r} is registered but has no factory branch."
    )
