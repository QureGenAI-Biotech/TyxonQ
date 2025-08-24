from __future__ import annotations

from typing import Dict, Optional
import json
import os


_TOKENS: Dict[str, str] = {}
_DEFAULTS: Dict[str, str] = {"provider": "tyxonq", "device": "tyxonq::simulator:mps"}
_AUTH_FILE = os.path.join(os.path.expanduser("~"), ".tyxonq.auth.json")

ENDPOINTS: Dict[str, Dict[str, str]] = {
    "tyxonq": {
        "base_url": os.getenv("TYXONQ_BASE_URL", "https://api.tyxonq.com/qau-cloud/tyxonq/"),
        "api_version": os.getenv("TYXONQ_API_VERSION", "v1"),
    }
}


def _save_tokens() -> None:
    try:
        with open(_AUTH_FILE, "w") as f:
            json.dump(_TOKENS, f)
    except Exception:
        pass


def _load_tokens() -> None:
    global _TOKENS
    if os.path.exists(_AUTH_FILE):
        try:
            with open(_AUTH_FILE, "r") as f:
                _TOKENS = json.load(f)
        except Exception:
            _TOKENS = {}


def set_token(token: str, *, provider: Optional[str] = None, device: Optional[str] = None, persist: bool = True) -> Dict[str, str]:
    prov = (provider or _DEFAULTS.get("provider") or "tyxonq").lower()
    key = prov + "::" + (device or "")
    _TOKENS[key] = token
    if persist:
        _save_tokens()
    return dict(_TOKENS)


def get_token(*, provider: Optional[str] = None, device: Optional[str] = None) -> Optional[str]:
    prov = (provider or _DEFAULTS.get("provider") or "tyxonq").lower()
    key = prov + "::" + (device or "")
    if not _TOKENS:
        _load_tokens()
    return _TOKENS.get(key)


def set_default(*, provider: Optional[str] = None, device: Optional[str] = None) -> None:
    if provider is not None:
        _DEFAULTS["provider"] = provider
    if device is not None:
        _DEFAULTS["device"] = device


def get_default_provider() -> str:
    return _DEFAULTS.get("provider", "tyxonq")


def get_default_device() -> str:
    return _DEFAULTS.get("device", "tyxonq::simulator:mps")


__all__ = [
    "ENDPOINTS",
    "set_token",
    "get_token",
    "set_default",
    "get_default_provider",
    "get_default_device",
]


