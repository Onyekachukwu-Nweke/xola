"""Pytest configuration for llm_surface tests.

Loads the repo-root .env before the test session so OPENAI_API_KEY and other
secrets are available to both unit tests (via the _mock_api_key fixture) and
integration tests (which use the real API).

``find_dotenv(usecwd=True)`` walks up from the current working directory, so
this works whether pytest is invoked from ``llm_surface/`` or from the repo root.
``override=False`` ensures that environment variables already set (e.g. CI
pipeline secrets) are never overwritten by the file.
"""

from dotenv import find_dotenv, load_dotenv

# Load once at collection time — before any fixture or test runs.
load_dotenv(find_dotenv(usecwd=True), override=False)
