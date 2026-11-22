"""Shared protocols and data structures for numerics GPU solver backends.

These interfaces are deliberately small: hardware-specific code lives in each
backend module, while the SQD business layer only depends on these contracts.
"""

from __future__ import annotations

from dataclasses import dataclass


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested backend cannot be used."""


@dataclass(frozen=True)
class MemoryInfo:
    free_bytes: int
    total_bytes: int
    allocated_bytes: int = 0
    reserved_bytes: int = 0

    @property
    def available_bytes(self) -> int:
        return max(0, self.free_bytes + max(0, self.reserved_bytes - self.allocated_bytes))


@dataclass(frozen=True)
class BackendCapabilities:
    device_kind: str
    supports_fp64: bool = True
    supports_autograd: bool = False
    distributed: bool = False


__all__ = ["BackendCapabilities", "BackendUnavailableError", "MemoryInfo"]
