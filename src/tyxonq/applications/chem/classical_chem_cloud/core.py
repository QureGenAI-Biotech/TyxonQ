"""Core configuration and factory functions for classical chemistry cloud computation."""

from __future__ import annotations

from typing import Dict, Any


class CloudClassicalConfig:
    """Configuration for cloud classical computation providers."""
    
    def __init__(self):
        self.providers = {
            "tyxonq": {
                "gpu": {
                    "endpoint": "https://classical-gpu.tyxonq.com",
                    "engine": "tyxonq_classical_gpu",  # Renamed from ByteQC
                    "default_config": {
                        "precision": "double",
                        "memory_limit": "32GB",
                        "timeout": 3600
                    }
                },
                "cpu": {
                    "endpoint": "https://classical-cpu.tyxonq.com", 
                    "engine": "tyxonq_classical_cpu",  # Renamed from PySCF remote
                    "default_config": {
                        "num_threads": 32,
                        "memory_limit": "64GB",
                        "timeout": 7200
                    }
                }
            }
        }
    
    def get_provider_config(self, provider: str, device: str) -> Dict[str, Any]:
        """Get configuration for specified provider and device."""
        if provider not in self.providers:
            raise ValueError(f"Unknown classical provider: {provider}")
        if device not in self.providers[provider]:
            raise ValueError(f"Unknown device '{device}' for provider '{provider}'")
        return self.providers[provider][device]


def create_classical_client(provider: str, device: str, config: CloudClassicalConfig = None):
    """Factory function to create appropriate classical computation client.

    Supports device in {"gpu", "cpu", "auto"}. For "auto", prefer GPU.
    """
    from .clients import TyxonQClassicalGPUClient, TyxonQClassicalCPUClient

    if provider != "tyxonq":
        raise ValueError(f"Unknown classical provider: {provider}")

    dev = str(device or "auto").lower()
    if dev == "auto":
        # Prefer GPU by default; cloud will schedule appropriately
        dev = "gpu"

    if dev == "gpu":
        return TyxonQClassicalGPUClient(config)
    if dev == "cpu":
        return TyxonQClassicalCPUClient(config)
    raise ValueError(f"Unknown device '{device}' for provider '{provider}'")


__all__ = ["CloudClassicalConfig", "create_classical_client"]