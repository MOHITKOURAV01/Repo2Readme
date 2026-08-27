"""Make the resolved API keys visible to the provider SDKs.

``llm.factory`` reads ``os.getenv(spec.env_var)`` when it is not handed a key
directly, so whatever this module exports is what the chains authenticate with.

The exports used to be written unconditionally::

    groq_key, gemini_key = get_api_keys()
    os.environ["GROQ_API_KEY"] = groq_key

which fails with ``TypeError: str expected, not NoneType`` the moment a key
resolves to ``None``, and hard-codes two variable names that the provider
registry already owns.
"""

import logging
import os

from repo2readme.config import ResolvedApiKey, resolve_api_key
from repo2readme.providers import get_provider

logger = logging.getLogger(__name__)

# Resolved when the user does not pass --provider: the summarizer falls back to
# Groq and the reviewer to Google, so a default run needs both.
DEFAULT_PROVIDERS = ("groq", "google")


def export_key(resolved: ResolvedApiKey) -> bool:
    """Put a resolved key in the environment. Returns whether anything changed.

    A key that came *from* the environment is already there, so it is left
    alone rather than rewritten with an identical value.
    """
    if resolved.env_var is None or not resolved.value:
        return False

    if os.environ.get(resolved.env_var) == resolved.value:
        return False

    os.environ[resolved.env_var] = resolved.value
    return True


def setup_api_keys(provider: str | None) -> None:
    """Configure API keys and export them as environment variables.

    When ``provider`` is given, only that provider's key is resolved. Providers
    that do not authenticate (a local Ollama server, for example) are accepted
    and simply skip the export. Without an explicit provider the historic
    Groq + Google defaults are used, since those are the models the summarizer
    and the reviewer fall back to.

    Raises
    ------
    UnknownProviderError
        If ``provider`` is not in the registry. The message lists the
        supported names.
    MissingApiKeyError
        If a key is needed and neither the environment, the saved store, nor a
        prompt can supply one.
    """
    names = (get_provider(provider).name,) if provider else DEFAULT_PROVIDERS

    for name in names:
        resolved = resolve_api_key(name)

        if resolved.env_var is None:
            logger.debug("%s needs no API key", name)
            continue

        export_key(resolved)
        logger.debug(
            "Using the %s API key from the %s", name, resolved.source
        )
