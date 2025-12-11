"""Circuit building blocks.

Keep functions backend-agnostic: operate only on `tyxonq.core.ir.circuit.Circuit`
and plain Python/sequence types. Do not depend on NumPy or torch here.
"""

from __future__ import annotations

from typing import Any, Sequence

from ...core.ir.circuit import Circuit


def example_block(c: Circuit, params: Any, *, nlayers: int) -> Circuit:
    """Simple hardware-efficient block.

    params is expected to be flattenable to length 2 * nlayers * n where
    n = c.num_qubits. Layout per layer j:
      - RZ angles: params[j*2*n + 0 : j*2*n + n]
      - RX angles: params[j*2*n + n : j*2*n + 2*n]
    """

    n = int(c.num_qubits)

    # Convert params to a flat Python list only; avoid backend deps here
    if isinstance(params, (list, tuple)):
        flat: Sequence[float] = params  # type: ignore[assignment]
    else:
        try:
            # Best-effort: objects with .reshape/.ravel
            flat = list(params.reshape(-1))  # type: ignore[attr-defined]
        except Exception:
            try:
                flat = list(params)  # type: ignore[arg-type]
            except Exception:
                raise TypeError("params must be sequence-like or reshape-able")

    # Initial H layer
    for q in range(n):
        c.h(q)

    # Layers
    for j in range(nlayers):
        base = j * 2 * n
        # Entangling CX chain
        for q in range(n - 1):
            c.cx(q, q + 1)
        # Parameterized 1q rotations
        for q in range(n):
            # Don't convert to float to preserve autograd chain
            theta_rz = flat[base + q]
            theta_rx = flat[base + n + q]
            c.rz(q, theta_rz)
            c.rx(q, theta_rx)

    return c


# Hardware-efficient RY ansatz: interleaved RY layers and CX entanglers.
def build_hwe_ry_ops(n_qubits: int, n_layers: int, params) -> Circuit:
    """Construct HWE RY ansatz as a Circuit.

    params shape: ((n_layers+1) * n_qubits,)
    """
    c = Circuit(n_qubits)
    try:
        mat = params.reshape(n_layers + 1, n_qubits)  # type: ignore[attr-defined]
    except Exception:
        # fall back to python slicing
        flat = list(params)
        mat = [flat[i * n_qubits : (i + 1) * n_qubits] for i in range(n_layers + 1)]
    # initial RY layer
    for i in range(n_qubits):
        # Don't convert to float to preserve autograd chain
        c.ry(i, theta=mat[0][i])
    c = c.add_barrier(*range(n_qubits))
    # entangling + RY layers
    for l in range(n_layers):
        for i in range(n_qubits - 1):
            c.cnot(i, i + 1)
        for i in range(n_qubits):
            # Don't convert to float to preserve autograd chain
            c.ry(i, theta=mat[l + 1][i])
        c = c.add_barrier(*range(n_qubits))
    return c

