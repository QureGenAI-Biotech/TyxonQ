from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from .common import _einsum_backend 

def init_statevector(num_qubits: int) -> np.ndarray:
    if num_qubits <= 0:
        return np.array([1.0 + 0.0j])
    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    state[0] = 1.0 + 0.0j
    return state


def apply_1q_statevector(backend: Any, state: np.ndarray, gate2: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    psi = state.reshape([2] * num_qubits)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    # Reserve 'a','b' for gate indices; use distinct axis symbols starting from 'c'
    axes = letters[2:2 + num_qubits]
    in_axes = axes.copy(); in_axes[qubit] = 'b'
    out_axes = axes.copy(); out_axes[qubit] = 'a'
    spec = f"ab,{''.join(in_axes)}->{''.join(out_axes)}"
    arr = _einsum_backend(backend, spec, gate2, psi)
    return arr.reshape(-1)


def apply_2q_statevector(backend: Any, state: np.ndarray, gate4: np.ndarray, q0: int, q1: int, num_qubits: int) -> np.ndarray:
    if q0 == q1:
        return state
    psi = state.reshape([2] * num_qubits)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    # Reserve 'a','b','c','d' for gate indices; use distinct axis symbols starting from 'e'
    axes = letters[4:4 + num_qubits]
    in_axes = axes.copy(); in_axes[q0] = 'c'; in_axes[q1] = 'd'
    out_axes = axes.copy(); out_axes[q0] = 'a'; out_axes[q1] = 'b'
    spec = f"abcd,{''.join(in_axes)}->{''.join(out_axes)}"
    arr = _einsum_backend(backend, spec, gate4.reshape(2, 2, 2, 2), psi)
    return arr.reshape(-1)


def expect_z_statevector(state: np.ndarray, qubit: int, num_qubits: int) -> float:
    s_perm = np.moveaxis(state.reshape([2] * num_qubits), qubit, 0)
    s2 = np.abs(s_perm.reshape(2, -1)) ** 2
    probs = np.sum(s2, axis=1)
    return float(probs[0] - probs[1])
