"""
Benchmark TFIM Hamiltonian construction and counts-based energy evaluation (chain API).

- Hamiltonian represented as lightweight Pauli-term list: List[Tuple[float, List[Tuple[str, int]]]]
- Energy estimated via measurement counts with basis rotations (X via H, Z native)
"""

from __future__ import annotations

import time
from typing import List, Tuple

import tyxonq as tq

Hamiltonian = List[Tuple[float, List[Tuple[str, int]]]]


def build_tfim_terms(n: int, hzz: float = 1.0, hx: float = -1.0, pbc: bool = False) -> Hamiltonian:
    terms: Hamiltonian = []
    # Z Z couplings
    for i in range(n - 1):
        terms.append((hzz, [("Z", i), ("Z", i + 1)]))
    if pbc and n > 1:
        terms.append((hzz, [("Z", n - 1), ("Z", 0)]))
    # X local fields
    for i in range(n):
        terms.append((hx, [("X", i)]))
    return terms


def _counts_from_circuit(c: tq.Circuit, shots: int) -> dict:
    out = (
        c.device(provider="simulator", device="statevector", shots=shots)
         .postprocessing(method=None)
         .run()
    )
    return out[0]["result"] if isinstance(out, list) else out.get("result", {})


def _expectation_from_counts_z(counts: dict, n: int, sites: List[int]) -> float:
    total = sum(counts.values()) or 1
    acc = 0.0
    for bitstr, cnt in counts.items():
        val = 1.0
        for q in sites:
            val *= (1.0 if bitstr[q] == "0" else -1.0)
        acc += val * cnt
    return acc / total


def energy_counts_tfim(n: int, terms: Hamiltonian, shots: int = 4096) -> float:
    # One shot for all Z-only terms
    cz = tq.Circuit(n)
    for q in range(n):
        cz.measure_z(q)
    counts_z = _counts_from_circuit(cz, shots)

    # Per-qubit shot for X terms (basis rotation H)
    counts_x = {}
    for i in range(n):
        cx = tq.Circuit(n)
        cx.h(i)
        for q in range(n):
            cx.measure_z(q)
        counts_x[i] = _counts_from_circuit(cx, shots)

    energy = 0.0
    for coeff, ops in terms:
        axes = [p for p, _ in ops]
        qs = [q for _, q in ops]
        if all(ax == "Z" for ax in axes):
            energy += coeff * _expectation_from_counts_z(counts_z, n, qs)
        elif len(ops) == 1 and axes[0] == "X":
            energy += coeff * _expectation_from_counts_z(counts_x[qs[0]], n, [qs[0]])
        else:
            raise NotImplementedError("Only Z and X terms supported in this demo.")
    return energy


if __name__ == "__main__":
    n = 10
    shots = 4096
    print("---- TFIM Hamiltonian building (list-of-terms) ----")
    t0 = time.perf_counter()
    H = build_tfim_terms(n, hzz=1.0, hx=-1.0, pbc=False)
    t1 = time.perf_counter()
    print({"n": n, "num_terms": len(H), "build_time_s": t1 - t0})

    print("---- Counts-based energy estimation ----")
    t2 = time.perf_counter()
    e = energy_counts_tfim(n, H, shots=shots)
    t3 = time.perf_counter()
    print({"shots": shots, "energy": e, "eval_time_s": t3 - t2})
