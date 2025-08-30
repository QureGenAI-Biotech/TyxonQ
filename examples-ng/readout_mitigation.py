"""
Readout error mitigation demo (refactored to new API).

This example builds a small Bell-like circuit, constructs synthetic measured counts
with readout errors, and applies readout mitigation to recover the ideal counts.
"""

from __future__ import annotations

import tyxonq as tq
from tyxonq.postprocessing.readout import ReadoutMit

from tyxonq.numerics import NumericBackend as nb

import numpy as np


def build_bell_circuit(nqubit: int = 2) -> tq.Circuit:
    c = tq.Circuit(nqubit)
    c.h(0).cx(0, 1)
    c.measure_z(0).measure_z(1)
    return c


def ideal_counts_for_bell(shots: int) -> dict[str, int]:
    # |Φ+> = (|00>+|11>)/sqrt(2) measured in Z basis → 00 and 11 with 50%-50%
    half = shots // 2
    return {"00": half, "11": shots - half}


    # Removed old local channel; now provided by postprocessing.readout.apply_readout_noise_to_counts
    pass


def apply_readout_noise_to_counts(raw_counts: Dict[str, int], single_qubit_cals: Dict[int, Any]) -> Dict[str, int]:
    """Apply per-qubit readout calibration matrices to ideal counts to simulate readout noise.

    Parameters:
        raw_counts: counts dict mapping bitstring→frequency (ideal ground truth)
        single_qubit_cals: mapping qubit index→2x2 matrix A so that measured_prob = A @ true_prob

    Returns:
        counts dict after applying readout channel.
    """
    if not raw_counts:
        return {}
    n = len(next(iter(raw_counts.keys())))
    size = 2**n
    prob_true = nb.zeros((size,), dtype=nb.float64)
    shots = 0.0
    for bstr, cnt in raw_counts.items():
        idx = int(bstr, 2)
        arr = nb.to_numpy(prob_true)
        arr[idx] = float(cnt)
        prob_true = nb.asarray(arr)
        shots += float(cnt)
    if shots <= 0:
        return dict(raw_counts)
    prob_true = nb.asarray(nb.to_numpy(prob_true) / shots)
    # kron A on ascending wires (0..n-1)
    A = None
    for q in range(n):
        m = single_qubit_cals.get(q)
        if m is None:
            m = nb.eye(2)
        A = m if A is None else nb.kron(A, m)
    # Convert to numpy for final rounding and indexing ops
    prob_meas_np = np.asarray(nb.to_numpy(A) @ nb.to_numpy(prob_true), dtype=float)
    prob_meas_np = np.clip(prob_meas_np, 0.0, 1.0)
    s = float(np.sum(prob_meas_np))
    if s <= 1e-12:
        s = 1.0
    prob_meas_np = prob_meas_np / s
    vecm = np.rint(prob_meas_np * shots).astype(int)
    out: Dict[str, int] = {}
    nz_idx = np.nonzero(vecm)[0]
    for idx in nz_idx:
        out[format(int(idx), f"0{n}b")] = int(vecm[int(idx)])
    return out


def demo_readout_mitigation():
    c = build_bell_circuit(2)
    shots = 10000

    # Ideal counts (synthetic)
    ideal = ideal_counts_for_bell(shots)

    # Define per-qubit calibration matrices (measured_prob = A @ true_prob), row-stochastic
    # Example: mild flip biases
    nb = tq.set_backend("numpy")  # choose numeric backend
    A0 = nb.array([[0.97, 0.03], [0.05, 0.95]], dtype=nb.float64)
    A1 = nb.array([[0.98, 0.02], [0.04, 0.96]], dtype=nb.float64)

    # Synthesize raw counts as if readout errors were applied
    raw = apply_readout_noise_to_counts(ideal, {0: A0, 1: A1})

    # Configure mitigation with known calibrations (numeric path)
    mit = ReadoutMit()
    # nb.to_numpy exists via API: convert backend arrays to numpy for scipy interop inside mitigation
    mit.set_single_qubit_cals({0: nb.to_numpy(A0), 1: nb.to_numpy(A1)})
    corrected = mit.apply_readout_mitigation(raw, method="inverse", qubits=[0, 1], shots=shots)

    # Chainable circuit execution path (proper noise injection via device layer)
    run_results = (
        c
        .device(
            provider="simulator",
            device="statevector",
            shots=shots,
            use_noise=True,
            noise={"type": "readout", "cals": {0: A0, 1: A1}},
        )
        .postprocessing(method="readout_mitigation", cals={0: nb.to_numpy(A0), 1: nb.to_numpy(A1)}, mitigation="inverse")
        .run()
    )
    counts_from_run = run_results.get("results", {}) if isinstance(run_results, dict) else run_results[0].get("results", {})
    # Postprocessing is performed by Circuit.run when method="readout_mitigation" and cals provided

    print("Ideal:", ideal)
    print("Raw (with readout error):", raw)
    print("Corrected (mitigated, numeric):", corrected)
    print("Run(raw with device-injected readout noise):", counts_from_run)
    postproc_payload = (
        run_results.get("postprocessing", {})
        if isinstance(run_results, dict)
        else (run_results[0].get("postprocessing", {}) if run_results else {})
    )
    print("Postprocessing payload:", postproc_payload)


if __name__ == "__main__":
    demo_readout_mitigation()
