"""Circuit building blocks (migrated from legacy templates.blocks).

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
            theta_rz = float(flat[base + q])
            theta_rx = float(flat[base + n + q])
            c.rz(q, theta_rz)
            c.rx(q, theta_rx)

    return c


