"""Where an API key comes from, and in what order.

``repo2readme`` documents three ways to supply a key - export it, put it in a
``.env`` file, or let the CLI ask once and remember it - but only the last one
ever worked. ``load_env`` read ``~/.repo2readme_env.json`` and nothing else, so
an exported ``GROQ_API_KEY`` was invisible: the CLI announced the key was
missing and prompted for it anyway. On a machine with no terminal attached -
which is exactly where an exported key is the *only* option - that prompt
aborted the run.

The order below is the one the documentation always described:

1. The process environment. ``python-dotenv`` loads ``.env`` into it before the
   CLI starts, so ``.env`` and ``export`` are the same source by the time
   anything here runs.
2. ``~/.repo2readme_env.json``, written by an earlier interactive run.
3. A prompt, and only when there is someone at the other end of stdin to answer
   it.

A key that came from the environment is never written to the store. The user is
already managing it somewhere; copying it into a second file only creates a
stale duplicate for the next time they rotate it.
"""

from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass
from rich import print as rprint

from repo2readme.providers import get_provider

try:
    import click
except Exception:
    click = None

ENV_PATH = os.path.join(os.path.expanduser("~"), ".repo2readme_env.json")

# The shortest key any supported provider issues is comfortably longer than
# this; the check exists to catch an empty prompt or a pasted placeholder.
MIN_API_KEY_LENGTH = 8

# Where a resolved key came from, for logging and for deciding whether it needs
# to be persisted.
SOURCE_ENVIRONMENT = "environment"
SOURCE_STORE = "stored"
SOURCE_PROMPT = "prompt"


class ApiKeyError(ValueError):
    """A key could not be resolved.

    Subclasses ``ValueError`` because that is what the previous invalid-key
    path raised, and the CLI already handles it.
    """


class MissingApiKeyError(ApiKeyError):
    """No key was available and there was no way to ask for one.

    Carries the provider and its environment variable so a caller can build a
    message of its own; the default one already says what to do.
    """

    def __init__(self, provider: str, env_var: str, label: str):
        self.provider = provider
        self.env_var = env_var
        self.label = label
        super().__init__(
            f"No {label} API key found, and there is no terminal to prompt on. "
            f"Set {env_var} in the environment, put it in a .env file, or run "
            f"repo2readme from a terminal once to enter and save it."
        )


@dataclass(frozen=True)
class ResolvedApiKey:
    """A key together with where it was found.

    ``value`` is ``None`` for providers that do not authenticate, which is not
    an error - a local Ollama server needs no key.
    """

    provider: str
    env_var: str | None
    value: str | None
    source: str | None

    @property
    def from_environment(self) -> bool:
        return self.source == SOURCE_ENVIRONMENT


def load_env():
    """Contents of the on-disk key store, or ``{}`` when there is none."""
    if not os.path.exists(ENV_PATH):
        return {}
    try:
        with open(ENV_PATH, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # A truncated or hand-edited store should send the user to the prompt,
        # not end the run with a traceback about JSON.
        return {}
    return data if isinstance(data, dict) else {}


def save_env(data):
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(ENV_PATH, flags, 0o600)
    if hasattr(os, 'fchmod'):
        os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=4)


def _clean(value) -> str | None:
    """A usable key, or ``None``. A whitespace-only value counts as absent."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def key_from_environment(env_var: str, environ=None) -> str | None:
    """The key exported in the process environment, if any.

    ``.env`` files arrive here too: the CLI calls ``load_dotenv()`` on import,
    which copies them into ``os.environ`` before anything asks this question.
    """
    source = os.environ if environ is None else environ
    return _clean(source.get(env_var))


def key_from_store(env_var: str) -> str | None:
    """The key saved by an earlier interactive run, if any."""
    return _clean(load_env().get(env_var))


def can_prompt(stream=None) -> bool:
    """Whether there is a terminal to ask on.

    ``click.prompt`` raises ``Abort`` when stdin is not a terminal, which
    surfaces as an empty failure message part way through a run. Asking first
    turns that into an error that names the variable to set.
    """
    stream = sys.stdin if stream is None else stream
    try:
        return bool(stream) and stream.isatty()
    except (AttributeError, ValueError):
        # A closed or replaced stdin: treat it as unusable rather than guessing.
        return False


def _prompt_for_key(label: str) -> str:
    """Ask for a key and validate the answer."""
    rprint(f"[yellow]{label} API key is missing![/yellow]\n")

    # Use Click prompt when available so CLI prompting integrates with Click
    # and is testable via click.testing.CliRunner. Fall back to built-in
    # input() if Click isn't present.
    if click is not None:
        # Let click.Abort (Ctrl-C, EOF) propagate: catching it here would
        # prompt a second time or surface a bare EOFError.
        api_key = click.prompt(
            f"Enter your {label} API key",
            hide_input=True,
            default="",
            show_default=False,
        ).strip()
    else:
        api_key = input(f"Enter your {label} API key: ").strip()

    if not api_key or len(api_key) < MIN_API_KEY_LENGTH:
        raise ApiKeyError(
            f"Invalid {label} API key. Key must be at least "
            f"{MIN_API_KEY_LENGTH} characters."
        )

    return api_key


def resolve_api_key(
    provider: str,
    *,
    prompt_if_missing: bool = True,
    environ=None,
) -> ResolvedApiKey:
    """Find the API key for ``provider``, and say where it came from.

    Parameters
    ----------
    provider:
        Provider name or alias, as accepted by ``repo2readme.providers``.
    prompt_if_missing:
        Ask for the key when neither the environment nor the store has one.
        Pass ``False`` to look without side effects.
    environ:
        Environment mapping to read. Defaults to ``os.environ``.

    Raises
    ------
    MissingApiKeyError
        When a key is needed, none was found, and prompting is impossible or
        was not asked for.
    ApiKeyError
        When the answer to the prompt was not a plausible key.
    """
    spec = get_provider(provider)

    if not spec.requires_api_key:
        return ResolvedApiKey(
            provider=spec.name, env_var=None, value=None, source=None
        )

    exported = key_from_environment(spec.env_var, environ=environ)
    if exported:
        return ResolvedApiKey(
            provider=spec.name,
            env_var=spec.env_var,
            value=exported,
            source=SOURCE_ENVIRONMENT,
        )

    stored = key_from_store(spec.env_var)
    if stored:
        return ResolvedApiKey(
            provider=spec.name,
            env_var=spec.env_var,
            value=stored,
            source=SOURCE_STORE,
        )

    if not prompt_if_missing or not can_prompt():
        raise MissingApiKeyError(spec.name, spec.env_var, spec.label)

    api_key = _prompt_for_key(spec.label)

    env = load_env()
    env[spec.env_var] = api_key
    save_env(env)

    rprint("[green]API key saved successfully![/green]")

    return ResolvedApiKey(
        provider=spec.name,
        env_var=spec.env_var,
        value=api_key,
        source=SOURCE_PROMPT,
    )


def get_api_key(provider: str):
    """Return the API key for a provider, prompting for it if missing.

    Providers that do not authenticate (for example a local Ollama server)
    return ``None`` instead of prompting.
    """
    return resolve_api_key(provider).value


def get_api_keys():
    """The Groq and Google keys, in that order.

    These are the two defaults: without ``--provider`` the summarizer falls
    back to Groq and the reviewer to Google, so such a run needs both.
    """
    groq = resolve_api_key("groq").value
    gemini = resolve_api_key("google").value
    return groq, gemini


def reset_api_keys():
    """Delete the on-disk key store. Returns whether there was one."""
    if os.path.exists(ENV_PATH):
        os.remove(ENV_PATH)
        return True
    return False
