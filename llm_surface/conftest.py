"""Pytest configuration for llm_surface tests.

Loads the repo-root .env before the test session so OPENAI_API_KEY and other
secrets are available to both unit tests (via the _mock_api_key fixture) and
integration tests (which use the real API).

``find_dotenv(usecwd=True)`` walks up from the current working directory, so
this works whether pytest is invoked from ``llm_surface/`` or from the repo root.
``override=False`` ensures that environment variables already set (e.g. CI
pipeline secrets) are never overwritten by the file.
"""

import os

import pytest
from dotenv import find_dotenv, load_dotenv

# Load once at collection time — before any fixture or test runs.
load_dotenv(find_dotenv(usecwd=True), override=False)


@pytest.fixture(autouse=True)
def _mock_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure OPENAI_API_KEY is set so the server lifespan can initialise.

    Only injects a dummy sentinel when the variable is absent (i.e. no .env
    was found). When conftest.py has already loaded the real key from .env,
    this is a no-op — leaving integration tests free to use the genuine key.
    Defined here (root conftest.py) so it applies to all test files regardless
    of the order in which modules are collected.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "test-sk-not-real")
