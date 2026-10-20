"""自旋动力学本地流程；独立拆分基态、Trotter 与 AQC 三种误差。"""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import eigsh, expm_multiply

from .aqc import AQCOptions, compress_dynamics
from .circuits import blocks_to_circuit, build_evolution, circuit_stats, evolution_blocks
from .ground_state import GroundStateOptions, prepare_ground_state
from .models import SpinChainModel, site_operator
from .simulation import pauli_expectations, simulate_observables, statevector
from .spectra import compute_dsf, error_metrics, response_from_kicks


@dataclass
class SpinDynamicsResult:
    """数值数组和可序列化摘要分开；绑定线路与参数对象供后续复查。"""

    arrays: dict
    summary: dict
    ground_state: object
    circuits: dict
    baseline_circuits: dict
    aqc_results: dict


def exact_response(model, initial_state, times, center, perturbation_axis="Z", measurement_axis="Z"):
    """小体系精确 Hamiltonian 演化；与 Trotter 构造完全独立。"""
    hamiltonian = model.sparse_matrix()
    state = np.asarray(initial_state, dtype=complex).reshape(-1)
    state = state / np.linalg.norm(state)
    perturbation = site_operator(model.n, center, perturbation_axis)
    values = []
    for sign in (1, -1):
        kicked = (state - 1j * sign * (perturbation @ state)) / np.sqrt(2)
        evolved = expm_multiply(-1j * hamiltonian, kicked, start=0, stop=float(times[-1]),
                                num=len(times), endpoint=True, traceA=0.0)
        values.append(np.array([pauli_expectations(psi, model.n, measurement_axis) for psi in evolved]))
    return response_from_kicks(*values)


def run_spin_dynamics(
    model: SpinChainModel | None = None, *, dt=0.6, steps=10, order=2,
    ground_state_options: GroundStateOptions | None = None,
    aqc_options: AQCOptions | None = None, simulator="statevector",
    center=None, perturbation_axis="Z", measurement_axis="Z",
    n_momentum=100, n_frequency=100, progress=None,
) -> SpinDynamicsResult:
    """运行完整本地流程；aqc_options=None 时保留未压缩基准。

    首版包含小体系精确参考，限定 2--10 个偶数位，避免误触指数级精确解。
    """
    model = model or SpinChainModel(10)
    gs_opts = ground_state_options or GroundStateOptions()
    if model.n > 10:
        raise ValueError("This workflow includes exact references and supports n <= 10")
    if simulator not in ("statevector", "quimb_mps"):
        raise ValueError("simulator must be 'statevector' or 'quimb_mps'")
    # 在 DMRG 或优化前检查模型、时间和测量设置，避免无效输入耗费计算。
    if steps < 1:
        raise ValueError("workflow steps must be >= 1")
    build_evolution(model, dt, steps, order)
    center = model.n // 2 - 1 if center is None else center
    if isinstance(center, bool) or not isinstance(center, (int, np.integer)) or not 0 <= center < model.n:
        raise ValueError("center must be an integer site inside the chain")
    perturbation_axis, measurement_axis = perturbation_axis.upper(), measurement_axis.upper()
    if perturbation_axis not in ("X", "Y", "Z") or measurement_axis not in ("X", "Y", "Z"):
        raise ValueError("axes must be X/Y/Z")
    if aqc_options is not None and (sum(s[0] for s in aqc_options.segments) > steps or
                                    max(s[1] for s in aqc_options.segments) > steps):
        raise ValueError("AQC plan exceeds the requested time steps")

    started = perf_counter()
    if progress:
        progress({"stage": "ground_state_start"})
    ground = prepare_ground_state(model, gs_opts)
    if progress:
        progress({"stage": "ground_state_done", "energy": ground.reference_energy,
                  "fidelity": ground.fidelity})
    times = np.arange(steps + 1) * dt
    baseline, executed, aqc_results = {}, {}, {}
    baseline_values, measured_values, fidelities = {}, {}, {}
    for label, sign in (("plus", 1), ("minus", -1)):
        initial_blocks = ground.blocks + [("r" + perturbation_axis.lower(), center, sign * np.pi / 2)]
        baseline[label] = [blocks_to_circuit(model.n, initial_blocks + evolution_blocks(model, dt, k, order))
                           for k in range(steps + 1)]
        if aqc_options is None:
            executed[label] = baseline[label]
        else:
            def report(event):
                if progress:
                    progress({**event, "branch": label})
            aqc_results[label] = compress_dynamics(model, initial_blocks, dt, steps, aqc_options,
                                                   order=order, progress=report)
            executed[label] = aqc_results[label].circuits
        baseline_values[label] = np.array([
            simulate_observables(c, measurement_axis, simulator, gs_opts.max_bond, gs_opts.cutoff)
            for c in baseline[label]])
        measured_values[label] = (baseline_values[label].copy() if aqc_options is None else np.array([
            simulate_observables(c, measurement_axis, simulator, gs_opts.max_bond, gs_opts.cutoff)
            for c in executed[label]]))
        # 最终线路的保真度用独立 TyxonQ 态矢量路径核对，包括压缩窗口以后的时间点。
        fidelities[label] = np.array([
            float(abs(np.vdot(statevector(a), statevector(b))) ** 2)
            for a, b in zip(baseline[label], executed[label])])

    green = response_from_kicks(measured_values["plus"], measured_values["minus"])
    full_green = response_from_kicks(baseline_values["plus"], baseline_values["minus"])
    exact_prepared = exact_response(model, statevector(ground.circuit), times, center,
                                   perturbation_axis, measurement_axis)
    # 精确时间演化使用 DMRG 选中的基态；简并时不会错误地换成另一个精确基态。
    exact_ground = exact_response(model, ground.reference_mps.to_dense().reshape(-1), times, center,
                                 perturbation_axis, measurement_axis)
    q, w, dsf = compute_dsf(green, times, center, n_momentum=n_momentum, n_frequency=n_frequency)
    spectra = {"dsf": dsf}
    for name, value in (("full_trotter_dsf", full_green), ("exact_prepared_dsf", exact_prepared),
                        ("exact_ground_dsf", exact_ground)):
        spectra[name] = compute_dsf(value, times, center, n_momentum=n_momentum, n_frequency=n_frequency)[2]
    arrays = {
        "times": times, "sites": np.arange(model.n), "momenta": q, "frequencies": w,
        "magnetization_plus": measured_values["plus"], "magnetization_minus": measured_values["minus"],
        "full_magnetization_plus": baseline_values["plus"], "full_magnetization_minus": baseline_values["minus"],
        "green": green, "full_trotter_green": full_green, "exact_prepared_green": exact_prepared,
        "exact_ground_green": exact_ground, "fidelity_plus": fidelities["plus"],
        "fidelity_minus": fidelities["minus"], **spectra,
    }
    errors = {}
    for stage, actual, reference in (("ground_state", "exact_prepared", "exact_ground"),
                                    ("trotter", "full_trotter", "exact_prepared"),
                                    ("aqc", "", "full_trotter")):
        errors[stage] = {}
        for quantity in ("green", "dsf"):
            actual_key = f"{actual}_{quantity}" if actual else quantity
            errors[stage][quantity] = error_metrics(arrays[actual_key], arrays[f"{reference}_{quantity}"])
    stats = {label: [{"full_trotter": circuit_stats(a), "executed": circuit_stats(b)}
                     for a, b in zip(baseline[label], executed[label])] for label in baseline}
    exact_energy = float(eigsh(model.sparse_matrix(), k=1, which="SA", return_eigenvectors=False,
                               v0=np.random.default_rng(gs_opts.seed).normal(size=2**model.n), tol=1e-12)[0])
    checks = {
        "dmrg_energy": abs(ground.reference_energy - exact_energy) < 1e-8,
        "ground_state_fidelity": ground.fidelity >= 0.98,
    }
    if aqc_results:
        checks["compressed_fidelity"] = all(r["fidelity"] >= 0.95 for a in aqc_results.values() for r in a.records)
        checks["independent_statevector_fidelity"] = all(np.min(f) >= 0.95 for f in fidelities.values())
        checks["reduced_final_gates_and_depth"] = all(
            stats[label][-1]["executed"][key] < stats[label][-1]["full_trotter"][key]
            for label in stats for key in ("num_2q_gates", "depth_2q", "depth"))
    summary = {
        "model": asdict(model), "dt": dt, "steps": steps, "order": order, "center": center,
        "perturbation_axis": perturbation_axis, "measurement_axis": measurement_axis,
        "simulator": simulator, "ground_state_options": asdict(gs_opts),
        "aqc_options": asdict(aqc_options) if aqc_options else None,
        "ground_state": {"reference_energy": ground.reference_energy, "exact_energy": exact_energy,
                         "prepared_energy": ground.prepared_energy, "fidelity": ground.fidelity,
                         "dmrg_converged": ground.dmrg_converged, "optimizer_success": ground.optimizer_success,
                         "optimizer_message": ground.optimizer_message, "iterations": ground.optimizer_iterations},
        "aqc": {label: result.records for label, result in aqc_results.items()},
        "circuit_stats": stats, "errors": errors, "acceptance_checks": checks,
        "acceptance_passed": all(checks.values()), "elapsed_seconds": perf_counter() - started,
        "conventions": {"spin": "S=sigma/2", "bit_order": "q0 is leftmost",
                        "retarded_response": "-i<[S_alpha(t),S_beta]>", "kick_difference_factor": 0.25,
                        "dsf": "-Im(G_R)/pi at zero temperature", "time_integral": "dt * sum over t>0",
                        "center_site_approximation": True, "mirror": False, "clip_negative": False,
                        "cost_basis": "H, X, RX, RY, RZ, CX; same local rotation folding for both paths",
                        "energy_unit": "coupling unit, hbar=1",
                        "momentum_resolution": 2 * np.pi / model.n,
                        "frequency_resolution": 2 * np.pi / times[-1]},
    }
    if progress:
        progress({"stage": "complete", "acceptance_passed": summary["acceptance_passed"],
                  "elapsed_seconds": summary["elapsed_seconds"]})
    return SpinDynamicsResult(arrays, summary, ground, executed, baseline, aqc_results)
