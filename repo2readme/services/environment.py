import os

from repo2readme.config import get_api_keys, get_api_key
from repo2readme.providers import get_provider


def setup_api_keys(provider: str | None) -> None:
    """Configure API keys and export them as environment variables.

    When ``provider`` is given, only that provider's key is resolved. Providers
    that do not authenticate (a local Ollama server, for example) are accepted
    and simply skip the export. Without an explicit provider the historic
    Groq + Google defaults are used, since those are the models the summarizer
    and the reviewer fall back to.
    """
    if not provider:
        groq_key, gemini_key = get_api_keys()
        os.environ["GROQ_API_KEY"] = groq_key
        os.environ["GOOGLE_API_KEY"] = gemini_key
        return

    # Raises UnknownProviderError, which lists the supported providers.
    spec = get_provider(provider)

    if not spec.requires_api_key:
        return

    api_key = get_api_key(spec.name)
    if api_key:
        os.environ[spec.env_var] = api_key
