"""Single source of truth for the LLM providers repo2readme supports.

Provider knowledge used to live in three places that drifted apart: the API key
map in ``config.py``, the environment map in ``services/environment.py`` and the
if/elif chain in ``llm/factory.py``. Everything now reads from the registry
below, so adding a provider is a one-line change and the CLI help text can be
generated instead of hand-maintained.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class UnknownProviderError(ValueError):
    """Raised when a provider name is not present in the registry."""

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(
            f"Unsupported provider: {provider!r}. "
            f"Supported providers: {', '.join(supported_providers())}."
        )


@dataclass(frozen=True)
class ProviderSpec:
    """Everything repo2readme needs to know about one provider.

    Attributes
    ----------
    name:
        Canonical lowercase identifier, e.g. ``"google"``.
    label:
        Human readable name used in prompts and error messages.
    default_model:
        Model used when the user does not pass ``--model``.
    env_var:
        Environment variable holding the API key, or ``None`` for providers
        that do not authenticate (local runtimes such as Ollama).
    aliases:
        Alternative names accepted on the command line, e.g. ``"gemini"``.
    default_base_url:
        Base URL applied when the user does not pass ``--base-url``.
    timeout_kwarg:
        Constructor argument this provider's client takes a per-request
        deadline through. The four libraries spell it four different ways -
        ``request_timeout``, ``default_request_timeout``, ``timeout`` - and
        Ollama takes none at all, reaching its transport through
        ``client_kwargs`` instead, which the factory handles.
    """

    name: str
    label: str
    default_model: str
    env_var: str | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    default_base_url: str | None = None
    timeout_kwarg: str | None = None

    @property
    def requires_api_key(self) -> bool:
        return self.env_var is not None


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="groq",
        label="Groq",
        default_model="openai/gpt-oss-120b",
        env_var="GROQ_API_KEY",
        timeout_kwarg="request_timeout",
    ),
    ProviderSpec(
        name="google",
        label="Google Gemini",
        default_model="gemini-2.5-flash",
        env_var="GOOGLE_API_KEY",
        aliases=("gemini",),
        timeout_kwarg="timeout",
    ),
    ProviderSpec(
        name="openai",
        label="OpenAI",
        default_model="gpt-4o-mini",
        env_var="OPENAI_API_KEY",
        timeout_kwarg="request_timeout",
    ),
    ProviderSpec(
        name="anthropic",
        label="Anthropic",
        default_model="claude-sonnet-4-5",
        env_var="ANTHROPIC_API_KEY",
        aliases=("claude",),
        timeout_kwarg="default_request_timeout",
    ),
    ProviderSpec(
        name="openrouter",
        label="OpenRouter",
        default_model="openai/gpt-4o-mini",
        env_var="OPENROUTER_API_KEY",
        default_base_url="https://openrouter.ai/api/v1",
        timeout_kwarg="request_timeout",
    ),
    ProviderSpec(
        name="together",
        label="Together AI",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        env_var="TOGETHER_API_KEY",
        default_base_url="https://api.together.xyz/v1",
        timeout_kwarg="request_timeout",
    ),
    ProviderSpec(
        name="ollama",
        label="Ollama (local)",
        default_model="llama3",
        env_var=None,
        default_base_url="http://localhost:11434",
    ),
)


def _build_index() -> dict[str, ProviderSpec]:
    index: dict[str, ProviderSpec] = {}
    for spec in PROVIDERS:
        for key in (spec.name, *spec.aliases):
            if key in index:
                raise RuntimeError(f"Duplicate provider key in registry: {key}")
            index[key] = spec
    return index


_INDEX = _build_index()


def supported_providers(include_aliases: bool = False) -> list[str]:
    """Return the provider names accepted on the command line."""
    if include_aliases:
        return sorted(_INDEX)
    return [spec.name for spec in PROVIDERS]


def is_supported(provider: str | None) -> bool:
    """Return True when ``provider`` resolves to a registry entry."""
    if not provider:
        return False
    return provider.strip().lower() in _INDEX


def get_provider(provider: str) -> ProviderSpec:
    """Resolve a provider name (or alias) to its :class:`ProviderSpec`.

    Raises
    ------
    UnknownProviderError
        If the name is not in the registry.
    """
    if not provider or not provider.strip():
        raise UnknownProviderError(provider or "")

    key = provider.strip().lower()
    try:
        return _INDEX[key]
    except KeyError:
        raise UnknownProviderError(provider) from None


def normalize_provider_name(provider: str) -> str:
    """Map an alias to its canonical provider name (``gemini`` -> ``google``)."""
    return get_provider(provider).name


def api_key_env_var(provider: str) -> str | None:
    """Environment variable holding the API key, or None if not needed."""
    return get_provider(provider).env_var


def requires_api_key(provider: str) -> bool:
    """Whether the provider needs an API key before it can be used."""
    return get_provider(provider).requires_api_key


def resolve_model(provider: str, model: str | None) -> str:
    """Return ``model`` when given, otherwise the provider default."""
    if model:
        return model
    return get_provider(provider).default_model


def resolve_base_url(provider: str, base_url: str | None) -> str | None:
    """Return ``base_url`` when given, otherwise the provider default."""
    if base_url:
        return base_url
    return get_provider(provider).default_base_url


def provider_choices_help() -> str:
    """Comma separated provider list used in the ``--provider`` help text."""
    return ", ".join(supported_providers())
