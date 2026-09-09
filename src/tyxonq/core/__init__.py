"""Core package exposing the device-independent IR primitives.

Only `ir` (Circuit / Hamiltonian) is provided. The former `operations` and
`measurements` scaffolding packages were removed as dead code (no production
consumers); measurement/observable handling lives with the IR and postprocessing.
"""

from .ir import Circuit, Hamiltonian

__all__ = ["Circuit", "Hamiltonian"]


