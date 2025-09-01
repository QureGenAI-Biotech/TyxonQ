"""Generic VQE components (example and scaffolding).

Design goals:
- Only depend on the construction and chain API execution of `tyxonq.core.ir.circuit.Circuit`;
- Decouple from specific numerical backends;
- Provide hardware-efficient ansatz, energy calculation from counts, parameter-shift gradients, and a one-shot evaluation wrapper.

Notes:
- Hamiltonian is expressed in a lightweight structure: List[Tuple[float, List[Tuple[str, int]]]]
  For example: H = 0.5 * Z0 + 0.5 * Z1 + 0.7 * Z0Z1 is represented as:
    [ (0.5, [("Z", 0)]), (0.5, [("Z", 1)]), (0.7, [("Z", 0), ("Z", 1)]) ]
"""

from __future__ import annotations

from typing import List, Tuple, Sequence, Dict
from ...core.ir.circuit import Circuit


Hamiltonian = List[Tuple[float, List[Tuple[str, int]]]]


def build_hwe_ansatz_ops(n: int, layers: int, params: Sequence[float]) -> List[Tuple]:
    """Construct a list of ops for a simple hardware-efficient ansatz:
    each layer: CX chain → RX(n) → RZ(n) → RX(n). The length of params should be 3*n*layers.
    """
    ops: List[Tuple] = []
    idx = 0
    for _ in range(layers):
        for i in range(n - 1):
            ops.append(("cx", i, i + 1))
        for i in range(n):
            ops.append(("rx", i, float(params[idx])))
            idx += 1
        for i in range(n):
            ops.append(("rz", i, float(params[idx])))
            idx += 1
        for i in range(n):
            ops.append(("rx", i, float(params[idx])))
            idx += 1
    return ops


def energy_from_counts(counts: Dict[str, int], n: int, hamiltonian: Hamiltonian) -> float:
    """Estimate the energy expectation E = <H> from measurement counts.

    - Assumes the corresponding basis transformations (commonly X→H, Y→HS, etc.) and Z-basis measurement have already been applied.
    - No basis-transformation inference is performed here; the caller must ensure the measurement basis is correct.
    """
    total = sum(counts.values()) or 1
    acc = 0.0
    for bitstr, cnt in counts.items():
        # bit '0' => +1, '1' => -1 on Z 期望
        val = 0.0
        for coeff, ops in hamiltonian:
            term = 1.0
            for (pauli, q) in ops:
                if pauli.upper() == "Z":
                    term *= (1.0 if bitstr[q] == '0' else -1.0)
                else:
                    # 这里仅示例 Z 项；X/Y 需在电路端做基变换
                    raise NotImplementedError("Only Z-terms supported in energy_from_counts; rotate basis in circuit.")
            val += coeff * term
        acc += val * cnt
    return acc / total


def _shift_for_gate(name: str) -> float:
    # RX/RZ 标准参数移位 π/2
    if name in ("rx", "rz"):
        from math import pi
        return 0.5 * pi
    return 0.0


def _param_layout(n: int, layers: int) -> List[Tuple[str, int]]:
    layout: List[Tuple[str, int]] = []
    for _ in range(layers):
        layout += [("rx", i) for i in range(n)]
        layout += [("rz", i) for i in range(n)]
        layout += [("rx", i) for i in range(n)]
    return layout


def parameter_shift_gradient(
    n: int,
    layers: int,
    params: Sequence[float],
    hamiltonian: Hamiltonian,
    *,
    shots: int = 4096,
    device_provider: str = "simulator",
    device_name: str = "statevector",
) -> List[float]:
    """Compute the gradient using the parameter-shift rule (for RX/RZ).

    - The caller is expected to have applied the necessary basis rotations in the circuit so that the Hamiltonian terms align with the measurement basis.
    - Internally, this function always measures in the Z basis and estimates the energy from the resulting counts.
    """
    layout = _param_layout(n, layers)
    base = list(params)
    grads: List[float] = []
    for k, (gname, _) in enumerate(layout):
        s = _shift_for_gate(gname)
        if s == 0.0:
            grads.append(0.0)
            continue
        p_plus = list(base); p_plus[k] = base[k] + s
        p_minus = list(base); p_minus[k] = base[k] - s

        # +shift
        c_p = Circuit(n, ops=build_hwe_ansatz_ops(n, layers, p_plus))
        for q in range(n):
            c_p.ops.append(("h", q))
            c_p.ops.append(("measure_z", q))
        r_p = c_p.device(provider=device_provider, device=device_name, shots=shots).postprocessing(method=None).run()
        counts_p = r_p[0]["result"] if isinstance(r_p, list) else r_p.get("result", {})
        e_p = energy_from_counts(counts_p, n, hamiltonian)

        # -shift
        c_m = Circuit(n, ops=build_hwe_ansatz_ops(n, layers, p_minus))
        for q in range(n):
            c_m.ops.append(("h", q))
            c_m.ops.append(("measure_z", q))
        r_m = c_m.device(provider=device_provider, device=device_name, shots=shots).postprocessing(method=None).run()
        counts_m = r_m[0]["result"] if isinstance(r_m, list) else r_m.get("result", {})
        e_m = energy_from_counts(counts_m, n, hamiltonian)

        grads.append(0.5 * (e_p - e_m))
    return grads


def evaluate_energy(
    n: int,
    layers: int,
    params: Sequence[float],
    hamiltonian: Hamiltonian,
    *,
    shots: int = 4096,
    device_provider: str = "simulator",
    device_name: str = "statevector",
) -> float:
    """Evaluate the energy in one shot: build circuit → H-basis transformation → Z measurement → counts → E[H]."""
    c = Circuit(n, ops=build_hwe_ansatz_ops(n, layers, params))
    for q in range(n):
        c.ops.append(("h", q))
        c.ops.append(("measure_z", q))
    res = c.device(provider=device_provider, device=device_name, shots=shots).postprocessing(method=None).run()
    counts = res[0]["result"] if isinstance(res, list) else res.get("result", {})
    return energy_from_counts(counts, n, hamiltonian)


