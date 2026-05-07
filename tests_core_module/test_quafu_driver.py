"""Unit tests for the quafu driver. All HTTPS is mocked — these run on any
Python in TyxonQ's supported range without a live token."""
from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _reset_quafu_state(monkeypatch):
    """Clear singleton + env vars + in-memory tokens around each test."""
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task
    from tyxonq.devices.hardware import config as hwcfg

    def _cleanup():
        if hasattr(Task, "instance"):
            del Task.instance
        hwcfg._TOKENS.clear()

    _cleanup()
    for var in ("TYXONQ_API_KEY", "TYXONQ_QUAFU_TOKEN", "QPU_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    yield
    _cleanup()


# ---------- Token resolution ----------

def test_resolve_token_prefers_explicit_kwarg(monkeypatch):
    monkeypatch.setenv("TYXONQ_QUAFU_TOKEN", "from-env")
    from tyxonq.devices.hardware.quafu.driver import _resolve_token

    assert _resolve_token("from-kwarg") == "from-kwarg"


def test_resolve_token_falls_back_to_set_token():
    import tyxonq as tq
    from tyxonq.devices.hardware.quafu.driver import _resolve_token

    tq.set_token("from-set-token", provider="quafu")
    assert _resolve_token(None) == "from-set-token"


def test_resolve_token_falls_back_to_tyxonq_quafu_env(monkeypatch):
    monkeypatch.setenv("TYXONQ_QUAFU_TOKEN", "from-tq-env")
    from tyxonq.devices.hardware.quafu.driver import _resolve_token

    assert _resolve_token(None) == "from-tq-env"


def test_resolve_token_falls_back_to_qpu_api_token(monkeypatch):
    monkeypatch.setenv("QPU_API_TOKEN", "from-qpu-env")
    from tyxonq.devices.hardware.quafu.driver import _resolve_token

    assert _resolve_token(None) == "from-qpu-env"


def test_resolve_token_prefers_tyxonq_env_over_upstream_env(monkeypatch):
    monkeypatch.setenv("TYXONQ_QUAFU_TOKEN", "tq-wins")
    monkeypatch.setenv("QPU_API_TOKEN", "qpu-loses")
    from tyxonq.devices.hardware.quafu.driver import _resolve_token

    assert _resolve_token(None) == "tq-wins"


def test_resolve_token_ignores_global_tyxonq_api_key(monkeypatch):
    """Regression: TYXONQ_API_KEY must NOT be sent to Quafu."""
    monkeypatch.setenv("TYXONQ_API_KEY", "tyxonq-cloud-token")
    from tyxonq.devices.hardware.quafu.driver import _resolve_token

    with pytest.raises(RuntimeError, match="Quafu token required"):
        _resolve_token(None)


def test_resolve_token_raises_with_helpful_message_when_missing():
    from tyxonq.devices.hardware.quafu.driver import _resolve_token

    with pytest.raises(RuntimeError) as exc_info:
        _resolve_token(None)
    msg = str(exc_info.value)
    assert "quafu-sqc.baqis.ac.cn" in msg
    assert "TYXONQ_QUAFU_TOKEN" in msg
    assert "tq.set_token" in msg
