"""Profile representative TyxonQ numerical workloads with reproducible samples.

The benchmark is intentionally separate from pytest: it records observations and
never turns timing thresholds into correctness assertions. Run suites serially,
and repeat the command in a fresh process before accepting an optimization.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np


SCALES = ("small", "medium", "large")
SUITES = ("all", "mps", "quantum-info", "ewf", "lucj", "skqd", "readout")


def synchronize(backend: Any = None) -> None:
    if backend is None:
        return
    if hasattr(backend, "synchronize"):
        backend.synchronize()
    elif hasattr(backend, "evaluate"):
        backend.evaluate()


def measure(
    suite: str,
    scale: str,
    fn: Callable[[], dict[str, Any]],
    *,
    repeats: int,
    metadata: dict[str, Any],
    backend: Any = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    cold_result = fn()
    synchronize(backend)
    cold_seconds = time.perf_counter() - started

    samples = []
    result = cold_result
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        synchronize(backend)
        samples.append(time.perf_counter() - started)

    # Memory instrumentation has observable overhead, so it is deliberately
    # excluded from both cold and steady-state timing samples.
    tracemalloc.start()
    fn()
    synchronize(backend)
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        peak_rss *= 1024

    return {
        "suite": suite,
        "scale": scale,
        "metadata": metadata,
        "cold_seconds": cold_seconds,
        "steady_seconds": samples,
        "steady_median_seconds": statistics.median(samples),
        "steady_min_seconds": min(samples),
        "steady_max_seconds": max(samples),
        "python_peak_bytes_dedicated_run": int(python_peak),
        "process_peak_rss_bytes": peak_rss,
        "cold_result": cold_result,
        "result": result,
    }


def mps_workload(scale: str, repeats: int) -> dict[str, Any]:
    from tyxonq.core.ir import Circuit
    from tyxonq.devices.simulators.matrix_product_state import MatrixProductStateEngine

    settings = {
        "small": (6, 2, 4),
        "medium": (10, 4, 16),
        "large": (14, 6, 32),
    }
    n, depth, max_bond = settings[scale]
    ops: list[tuple[Any, ...]] = [("h", q) for q in range(n)]
    for layer in range(depth):
        for q in range(layer % 2, n - 1, 2):
            ops.append(("cx", q, q + 1))
        for q in range(n):
            ops.extend((("rz", q, 0.07 * (layer + 1)), ("rx", q, 0.03 * (q + 1))))
    circuit = Circuit(num_qubits=n, ops=ops)
    backend_name = "numpy"

    def run() -> dict[str, Any]:
        engine = MatrixProductStateEngine(backend_name=backend_name, max_bond=max_bond)
        started = time.perf_counter()
        state = engine.state(circuit)
        synchronize(engine.backend)
        evolve_seconds = time.perf_counter() - started
        return {
            "phases": {"evolve": evolve_seconds},
            "bond_dimensions": [list(map(int, tensor.shape[::2])) for tensor in state.tensors],
            "checksum": float(sum(float(np.sum(np.abs(tensor))) for tensor in state.tensors)),
        }

    return measure(
        "mps",
        scale,
        run,
        repeats=repeats,
        backend=None,
        metadata={
            "qubits": n,
            "depth": depth,
            "max_bond": max_bond,
            "two_qubit_gates": sum(op[0] == "cx" for op in ops),
            "backend": backend_name,
            "dtype": "complex128",
            "estimated_mps_tensor_bytes": int(n * max_bond * 2 * max_bond * 16),
        },
    )


def quantum_info_workload(scale: str, repeats: int) -> dict[str, Any]:
    from tyxonq.libs.quantum_library.kernels import quantum_info as qi

    n = {"small": 4, "medium": 6, "large": 8}[scale]
    dim = 2**n
    rng = np.random.default_rng(20261119 + n)
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi /= np.linalg.norm(psi)
    other = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    other /= np.linalg.norm(other)
    rho = 0.75 * np.outer(psi, psi.conj()) + 0.25 * np.outer(other, other.conj())
    cut = list(range(n // 2))

    def run() -> dict[str, Any]:
        phases: dict[str, float] = {}

        def stage(name: str, call: Callable[[], Any]) -> Any:
            started = time.perf_counter()
            value = call()
            phases[name] = time.perf_counter() - started
            return value

        reduced_pure = stage("reduced_pure", lambda: qi.reduced_density_matrix(psi, cut))
        reduced_mixed = stage("reduced_mixed", lambda: qi.reduced_density_matrix(rho, cut))
        entropy = stage("entropy", lambda: qi.entropy(rho))
        fidelity = stage("fidelity", lambda: qi.fidelity(rho, rho))
        negativity = stage("negativity", lambda: qi.entanglement_negativity(rho, cut))
        distance = stage("trace_distance", lambda: qi.trace_distance(rho, 0.5 * (rho + np.eye(dim) / dim)))
        return {
            "phases": phases,
            "entropy": entropy,
            "fidelity": fidelity,
            "negativity": negativity,
            "trace_distance": distance,
            "reduced_pure_trace": float(np.trace(reduced_pure).real),
            "reduced_mixed_trace": float(np.trace(reduced_mixed).real),
        }

    return measure(
        "quantum-info",
        scale,
        run,
        repeats=repeats,
        metadata={
            "qubits": n,
            "dense_dimension": dim,
            "dtype": "complex128",
            "state_kind": "pure_and_rank_two_mixed",
            "dense_matrix_bytes": int(dim * dim * 16),
            "traced_identity_bytes": int((2 ** len(cut)) ** 2 * 16),
        },
    )


def ewf_workload(scale: str, repeats: int) -> dict[str, Any]:
    from pyscf import gto, scf
    from tyxonq.applications.chem.algorithms.ewf import EWF
    from tyxonq.applications.chem.algorithms.ewf import driver as driver_module
    from tyxonq.applications.chem.algorithms.ewf import energy as energy_module

    settings = {
        "small": ("H 0 0 0; H 0 0 0.74", "sto-3g", "FCI"),
        "medium": ("Li 0 0 0; H 0 0 1.6", "sto-3g", "CCSD"),
        "large": ("H 0 0 0; H 0 0 0.74", "cc-pvdz", "CCSD"),
    }
    atom, basis, solver = settings[scale]
    setup_started = time.perf_counter()
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.run()
    setup_seconds = time.perf_counter() - setup_started

    original_fragment = energy_module.fragment_dmet_energy
    original_driver_fragment = driver_module.fragment_dmet_energy
    original_total = driver_module.dmet_total_energy
    counters: dict[str, float] = {}

    def timed_fragment(*args: Any, **kwargs: Any) -> float:
        started = time.perf_counter()
        value = original_fragment(*args, **kwargs)
        counters["fragment_calls"] = counters.get("fragment_calls", 0) + 1
        counters["fragment_seconds"] = counters.get("fragment_seconds", 0.0) + time.perf_counter() - started
        return value

    def timed_total(*args: Any, **kwargs: Any) -> float:
        started = time.perf_counter()
        value = original_total(*args, **kwargs)
        counters["total_seconds"] = time.perf_counter() - started
        return value

    driver_module.fragment_dmet_energy = timed_fragment
    energy_module.fragment_dmet_energy = timed_fragment
    driver_module.dmet_total_energy = timed_total
    try:
        def run() -> dict[str, Any]:
            counters.clear()
            emb = EWF(mf, solver=solver, energy_functional="dmet")
            energy = emb.kernel()
            fragments = emb.result.fragments if emb.result is not None else ()
            build = sum(f.diagnostics["build_seconds"] for f in fragments)
            solve = sum(f.diagnostics["solve_seconds"] for f in fragments)
            return {
                "phases": {
                    "fragment_build": build,
                    "cluster_solve": solve,
                    "fragment_dmet_energy": counters.get("fragment_seconds", 0.0),
                    "dmet_total_including_fragments": counters.get("total_seconds", 0.0),
                },
                "energy": energy,
                "converged": emb.converged,
                "fragment_count": len(fragments),
                "cluster_orbitals": [f.cluster.norb for f in fragments],
                "fragment_dmet_calls": int(counters.get("fragment_calls", 0)),
            }

        record = measure(
            "ewf",
            scale,
            run,
            repeats=repeats,
            metadata={
                "atom": atom,
                "basis": basis,
                "solver": solver,
                "energy_functional": "dmet",
                "scf_setup_seconds": setup_seconds,
                "ao_orbitals": int(mol.nao),
                "dtype": "float64",
            },
        )
    finally:
        driver_module.fragment_dmet_energy = original_driver_fragment
        energy_module.fragment_dmet_energy = original_fragment
        driver_module.dmet_total_energy = original_total
    return record


def lucj_workload(scale: str, repeats: int) -> dict[str, Any]:
    from tyxonq.applications.chem.algorithms.lucj import double_factorization as df

    settings = {
        "small": (1, 2, 1, 2),
        "medium": (2, 2, 1, 3),
        "large": (2, 3, 2, 4),
    }
    nocc, nvir, max_terms, maxiter = settings[scale]
    norb = nocc + nvir
    rng = np.random.default_rng(9100 + norb)
    t2 = 0.01 * rng.normal(size=(nocc, nocc, nvir, nvir))
    t2ab = 0.01 * rng.normal(size=(nocc, max(1, nocc - 1), nvir, norb - max(1, nocc - 1)))
    counters: dict[str, float] = {}
    original_restricted = df._torch_reconstruct_from_params
    original_unrestricted = df._torch_reconstruct_alpha_beta_from_params

    def timed_restricted(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        value = original_restricted(*args, **kwargs)
        counters["restricted_calls"] = counters.get("restricted_calls", 0) + 1
        counters["restricted_seconds"] = counters.get("restricted_seconds", 0.0) + time.perf_counter() - started
        return value

    def timed_unrestricted(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        value = original_unrestricted(*args, **kwargs)
        counters["unrestricted_calls"] = counters.get("unrestricted_calls", 0) + 1
        counters["unrestricted_seconds"] = counters.get("unrestricted_seconds", 0.0) + time.perf_counter() - started
        return value

    df._torch_reconstruct_from_params = timed_restricted
    df._torch_reconstruct_alpha_beta_from_params = timed_unrestricted
    try:
        def run() -> dict[str, Any]:
            counters.clear()
            started = time.perf_counter()
            restricted_output = cast(Any, df.double_factorized_t2(
                t2,
                max_terms=max_terms,
                optimize=True,
                options={"maxiter": maxiter},
                return_optimize_result=True,
            ))
            restricted = restricted_output[2]
            restricted_seconds = time.perf_counter() - started
            started = time.perf_counter()
            unrestricted_output = cast(Any, df.double_factorized_t2_alpha_beta(
                t2ab,
                max_terms=max_terms,
                optimize=True,
                options={"maxiter": maxiter},
                return_optimize_result=True,
            ))
            unrestricted = unrestricted_output[2]
            unrestricted_seconds = time.perf_counter() - started
            return {
                "phases": {
                    "restricted_optimization": restricted_seconds,
                    "alpha_beta_optimization": unrestricted_seconds,
                    "restricted_reconstruct_forward": counters.get("restricted_seconds", 0.0),
                    "alpha_beta_reconstruct_forward": counters.get("unrestricted_seconds", 0.0),
                },
                "restricted_loss": float(restricted.fun),
                "alpha_beta_loss": float(unrestricted.fun),
                "restricted_nfev": int(restricted.nfev),
                "alpha_beta_nfev": int(unrestricted.nfev),
                "restricted_objective_calls": int(counters.get("restricted_calls", 0)),
                "alpha_beta_objective_calls": int(counters.get("unrestricted_calls", 0)),
                "restricted_loss_history": np.asarray(restricted.loss_history).tolist(),
                "alpha_beta_loss_history": np.asarray(unrestricted.loss_history).tolist(),
            }

        return measure(
            "lucj",
            scale,
            run,
            repeats=repeats,
            metadata={
                "spatial_orbitals": norb,
                "occupied_orbitals": nocc,
                "virtual_orbitals": nvir,
                "max_terms": max_terms,
                "maxiter": maxiter,
                "dtype": "float64_complex128",
                "optimizer": "scipy_lbfgsb_with_pytorch_autograd",
                "t2_elements": int(t2.size + t2ab.size),
            },
        )
    finally:
        df._torch_reconstruct_from_params = original_restricted
        df._torch_reconstruct_alpha_beta_from_params = original_unrestricted


def skqd_workload(scale: str, repeats: int) -> dict[str, Any]:
    from tyxonq.applications.chem.algorithms.skqd.basis import rotate_integrals

    norb = {"small": 4, "medium": 6, "large": 8}[scale]
    rng = np.random.default_rng(12000 + norb)
    h1 = rng.normal(size=(norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    chol = rng.normal(size=(norb, norb, norb))
    chol = 0.5 * (chol + chol.swapaxes(0, 1))
    eri = np.einsum("pqx,rsx->pqrs", chol, chol, optimize=True)
    rotation = np.linalg.qr(rng.normal(size=(norb, norb)))[0]

    def run() -> dict[str, Any]:
        started = time.perf_counter()
        h_rotated, eri_rotated = rotate_integrals(h1, eri, rotation)
        rotation_seconds = time.perf_counter() - started
        return {
            "phases": {"integral_rotation": rotation_seconds},
            "h1_checksum": float(np.sum(np.abs(h_rotated))),
            "eri_checksum": float(np.sum(np.abs(eri_rotated))),
        }

    return measure(
        "skqd",
        scale,
        run,
        repeats=repeats,
        metadata={
            "spatial_orbitals": norb,
            "eri_elements": int(eri.size),
            "eri_bytes": int(eri.nbytes),
            "dtype": "float64",
            "operation": "five_operand_optimized_einsum",
        },
    )


def readout_workload(scale: str, repeats: int) -> dict[str, Any]:
    from tyxonq.postprocessing.readout import ReadoutMit

    n = {"small": 4, "medium": 6, "large": 8}[scale]
    dim = 2**n
    rng = np.random.default_rng(13000 + n)
    shots = 100000
    counts_array = rng.multinomial(shots, np.full(dim, 1.0 / dim))
    counts = {
        format(index, f"0{n}b"): int(count)
        for index, count in enumerate(counts_array)
        if count
    }
    calibration = np.array([[0.98, 0.02], [0.02, 0.98]])

    def run() -> dict[str, Any]:
        mitigation = ReadoutMit()
        mitigation.set_single_qubit_cals({q: calibration for q in range(n)})
        started = time.perf_counter()
        mitigated = mitigation.apply_readout_mitigation(counts)
        inverse_seconds = time.perf_counter() - started
        return {
            "phases": {"inverse_mitigation": inverse_seconds},
            "mitigated_shots": int(sum(mitigated.values())),
            "nonzero_counts": len(mitigated),
        }

    return measure(
        "readout",
        scale,
        run,
        repeats=repeats,
        metadata={
            "qubits": n,
            "calibration_dimension": dim,
            "calibration_bytes": int(dim * dim * 8),
            "shots": shots,
            "method": "inverse_scipy_pinv",
            "dtype": "float64",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="all")
    parser.add_argument("--scale", choices=(*SCALES, "all"), default="all")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("--repeats must be at least 5 for steady-state acceptance data")

    selected = SUITES[1:] if args.suite == "all" else (args.suite,)
    scales = SCALES if args.scale == "all" else (args.scale,)
    runners = {
        "mps": mps_workload,
        "quantum-info": quantum_info_workload,
        "ewf": ewf_workload,
        "lucj": lucj_workload,
        "skqd": skqd_workload,
        "readout": readout_workload,
    }
    records = []
    for suite in selected:
        for scale in scales:
            records.append(runners[suite](scale, args.repeats))

    report = {
        "schema_version": 1,
        "created_unix": time.time(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "blas_threads": "use the invoking shell environment; no implicit changes",
        },
        "protocol": {
            "steady_repeats": args.repeats,
            "serial": True,
            "timing_threshold_assertions": False,
            "memory_timing_separated": True,
            "speedup_acceptance_percent": 5.0,
        },
        "workloads": records,
    }
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
