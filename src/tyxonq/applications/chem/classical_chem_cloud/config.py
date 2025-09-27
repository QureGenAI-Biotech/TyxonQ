"""Core configuration and factory functions for classical chemistry cloud computation."""

from __future__ import annotations

from typing import Dict, Any


class CloudClassicalConfig:
    """Configuration for cloud classical computation providers."""
    
    def __init__(self):
        self.providers = {
            "tyxonq": {
                # "endpoint": "https://classical.tyxonq.com",
                "endpoint": "http://127.0.0.1:8009",
                "engine": "tyxonq_classical",
                "default_config": {
                    "timeout": 7200
                }
            }
        }
    
    def get_provider_config(self, provider: str, device: str) -> Dict[str, Any]:
        """Get configuration for specified provider (device ignored, reserved)."""
        if provider not in self.providers:
            raise ValueError(f"Unknown classical provider: {provider}")
        return self.providers[provider]


def create_classical_client(provider: str, device: str, config: CloudClassicalConfig = None):
    """Factory function to create classical computation client.

    Device is passed through to server via payload; scheduling is handled remotely.
    """
    from .clients import TyxonQClassicalClient

    if provider != "tyxonq":
        raise ValueError(f"Unknown classical provider: {provider}")
    return TyxonQClassicalClient(device=str(device or "auto"), config=config)


__all__ = ["CloudClassicalConfig", "create_classical_client"]