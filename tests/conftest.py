"""Shared fixtures.

API keys are read from the process environment now, which makes the suite
sensitive to the machine it runs on: a developer with ``GROQ_API_KEY`` exported
would see key-resolution tests take a different branch from CI. Every test
starts from an environment with no provider keys in it, and opts back in by
setting the ones it needs.
"""

import pytest

from repo2readme.providers import PROVIDERS

PROVIDER_ENV_VARS = tuple(
    spec.env_var for spec in PROVIDERS if spec.env_var is not None
)


@pytest.fixture(autouse=True)
def _clear_provider_api_keys(monkeypatch):
    """Remove every provider API key from the environment for one test."""
    for env_var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
