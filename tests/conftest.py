"""Shared fixtures.

The chat client cache is process-level, so without this a client built by one
test would be handed to the next one - including a fake injected through
``sys.modules``. Clearing it around every test keeps them independent.
"""

import pytest

from repo2readme.llm.client_cache import clear_client_cache


@pytest.fixture(autouse=True)
def isolate_llm_client_cache():
    clear_client_cache()
    yield
    clear_client_cache()
