import os
from repo2readme.config import get_api_keys, get_api_key

def setup_api_keys(provider: str | None) -> None:
    """Configures the API keys and sets the required environment variables."""
    if provider:
        api_key = get_api_key(provider)

        provider_env = {
            "groq": "GROQ_API_KEY",
            "google": "GOOGLE_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "together": "TOGETHER_API_KEY",
        }

        os.environ[provider_env[provider.lower()]] = api_key
    else:
        groq_key, gemini_key = get_api_keys()
        os.environ["GROQ_API_KEY"] = groq_key
        os.environ["GOOGLE_API_KEY"] = gemini_key
