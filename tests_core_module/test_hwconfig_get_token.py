from __future__ import annotations

import os
import pytest


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Reset in-memory token store and env between tests."""
    from tyxonq.devices.hardware import config as hwcfg
    hwcfg._TOKENS.clear()
    monkeypatch.delenv("TYXONQ_API_KEY", raising=False)


def test_env_fallback_default_returns_global_key(monkeypatch):
    from tyxonq.devices.hardware import config as hwcfg

    monkeypatch.setenv("TYXONQ_API_KEY", "global-key")
    # Default behavior: provider with no in-memory token falls back to env.
    assert hwcfg.get_token(provider="anything") == "global-key"


def test_env_fallback_disabled_returns_none(monkeypatch):
    from tyxonq.devices.hardware import config as hwcfg

    monkeypatch.setenv("TYXONQ_API_KEY", "global-key")
    # Quafu driver opts out of the env fallback so it can run its own chain.
    assert hwcfg.get_token(provider="quafu", env_fallback=False) is None


def test_env_fallback_disabled_still_returns_in_memory_token():
    from tyxonq.devices.hardware import config as hwcfg

    hwcfg.set_token("explicit-quafu-token", provider="quafu")
    assert hwcfg.get_token(provider="quafu", env_fallback=False) == "explicit-quafu-token"


def test_existing_tyxonq_behavior_preserved(monkeypatch):
    """Default env_fallback=True must keep the tyxonq driver working unchanged."""
    from tyxonq.devices.hardware import config as hwcfg

    monkeypatch.setenv("TYXONQ_API_KEY", "tyxonq-key")
    assert hwcfg.get_token(provider="tyxonq") == "tyxonq-key"


def test_no_token_no_env_returns_none():
    """Default fallback path must return None when neither in-memory nor env has a token."""
    from tyxonq.devices.hardware import config as hwcfg

    assert hwcfg.get_token(provider="anything") is None
