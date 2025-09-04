"""
Hamiltonian encoding library (problem → qubit operators)
-------------------------------------------------------

This package provides stable import paths for operator/Hamiltonian encoding utilities
that are domain-agnostic and used prior to gate-level construction.

Note: during migration this package re-exports implementations from
tyxonq.libs.operator_library to avoid breaking callers. New code should
prefer this package name.
"""

from tyxonq.libs.hamiltonian_encoding.operator_encoding import *  # noqa: F401,F403
from tyxonq.libs.hamiltonian_encoding.pauli_io import *  # noqa: F401,F403


