from __future__ import annotations

from typing import Dict, Optional
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




def set_token(token: str, *, provider: Optional[str] = None, device: Optional[str] = None) -> Dict[str, str]:
    prov = (provider or _DEFAULTS.get("provider") or "tyxonq").lower()
    # Store both device-scoped and provider-scoped entries for flexibility
    key_device = prov + "::" + (device or "")
    key_provider = prov + "::"
    _TOKENS[key_device] = token
    _TOKENS[key_provider] = token
    return dict(_TOKENS)


def get_token(*, provider: Optional[str] = None, device: Optional[str] = None) -> Optional[str]:
    prov = (provider or _DEFAULTS.get("provider") or "tyxonq").lower()
    key_device = prov + "::" + (device or "")
    key_provider = prov + "::"
    # Prefer in-memory tokens
    tok = _TOKENS.get(key_device) or _TOKENS.get(key_provider)
    if tok:
        return tok
    # Fallback to environment variable
    return os.getenv("TYXONQ_API_KEY")


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


