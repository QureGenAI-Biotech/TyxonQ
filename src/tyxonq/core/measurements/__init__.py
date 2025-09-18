from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Expectation:
    """Expectation measurement of an observable on given wires."""

    obs: str
    wires: Tuple[int, ...]


@dataclass(frozen=True)
class Probability:
    """Probability measurement over the computational basis on wires."""

    wires: Tuple[int, ...]


@dataclass(frozen=True)
class Sample:
    """Sample measurement on given wires with a number of shots."""

    wires: Tuple[int, ...]
    shots: int


__all__ = ["Expectation", "Probability", "Sample"]


