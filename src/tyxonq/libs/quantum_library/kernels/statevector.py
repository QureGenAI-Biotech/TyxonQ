from __future__ import annotations

from typing import Any, Sequence
import numpy as np
from ....numerics import NumericBackend as nb
from ....numerics.api import ArrayBackend


def init_statevector(num_qubits: int, backend: ArrayBackend | None = None) -> Any:
    K = backend or nb
    if num_qubits <= 0:
        return K.array([1.0 + 0.0j], dtype=K.complex128)
    dim = 1 << num_qubits
    data = [1.0 + 0.0j] + [0.0 + 0.0j] * (dim - 1)
    return K.array(data, dtype=K.complex128)


def apply_1q_statevector(backend: Any, state: Any, gate2: Any, qubit: int, num_qubits: int) -> Any:
    K = backend or nb
    psi = K.reshape(K.asarray(state), (2,) * num_qubits)
    g2 = K.asarray(gate2)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    # Reserve 'a','b' for gate indices; use distinct axis symbols starting from 'c'
    axes = letters[2:2 + num_qubits]
    in_axes = axes.copy(); in_axes[qubit] = 'b'
    out_axes = axes.copy(); out_axes[qubit] = 'a'
    spec = f"ab,{''.join(in_axes)}->{''.join(out_axes)}"
    arr = K.einsum(spec, g2, psi)
    return K.reshape(K.asarray(arr), (-1,))


def apply_2q_statevector(backend: Any, state: Any, gate4: Any, q0: int, q1: int, num_qubits: int) -> Any:
    if q0 == q1:
        return state
    K = backend or nb
    psi = K.reshape(K.asarray(state), (2,) * num_qubits)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    # Reserve 'a','b','c','d' for gate indices; use distinct axis symbols starting from 'e'
    axes = letters[4:4 + num_qubits]
    in_axes = axes.copy(); in_axes[q0] = 'c'; in_axes[q1] = 'd'
    out_axes = axes.copy(); out_axes[q0] = 'a'; out_axes[q1] = 'b'
    spec = f"abcd,{''.join(in_axes)}->{''.join(out_axes)}"
    g4 = K.reshape(K.asarray(gate4), (2, 2, 2, 2))
    arr = K.einsum(spec, g4, psi)
    return K.reshape(K.asarray(arr), (-1,))


def expect_z_statevector(state: Any, qubit: int, num_qubits: int, backend: ArrayBackend | None = None) -> Any:
    K = backend or nb
    s = K.reshape(K.asarray(state), (2,) * num_qubits)
    s_perm = K.moveaxis(s, qubit, 0)
    s2 = K.abs(K.reshape(s_perm, (2, -1))) ** 2  # type: ignore[operator]
    sums = K.sum(s2, axis=1)
    return sums[0] - sums[1]
