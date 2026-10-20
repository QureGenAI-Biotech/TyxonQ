"""闭壳层 EWF–LUCJ–SQD 的单点、有限差分检查和近似几何优化示例。

流程：几何 → RHF → 原子片段与 MP2-BNO 浴 → FCI 或 LUCJ 采样/SQD →
全局密度与能量 → 近似梯度 → ASE/Sella 优化。仅运行本地计算。

从仓库根目录运行：
    export PYTHONPATH=src
    python examples/ewf_geometry.py singlepoint
    python examples/ewf_geometry.py singlepoint --molecule h2 --force-sqd
    python examples/ewf_geometry.py gradient-check --molecule h2
    python examples/ewf_geometry.py optimize --output /tmp/ewf_water.json
几何优化另需安装可选依赖：pip install -e '.[geomopt]'。

默认 H₂O/STO-3G，空间轨道数不超过 12 的片段使用 FCI；--force-sqd 强制使用
CCSD 振幅初始化、LUCJ 态矢量采样和 SQD。small 配置使用 2048 shots、
2 批×256 样本、3 轮 SQD 和 50 步拟合。--profile paper 使用浴阈值 1e-5、
单层 LUCJ、1000 步拟合、100000 shots、4 批×2000 样本及最多 5 轮 SQD，
并显式限制同/异自旋耦合对；全局密度默认 rdm_t_lambda。

每批注入 HF 组态；扩展筛选为概率 c²>1e-5，carryover 为振幅阈值 1e-4。
保留历轮最优解及原组态空间，不保证与其他实现逐轮随机轨迹一致。
每个新几何重建嵌入并重新采样，同一几何共享能量和梯度结果。

态矢量采样最多 20 qubits，内存随 2**nqubits 增长；全局密度/Λ 张量为
O(N_MO**4) 存储。paper 配置不代表本机可模拟论文的大线路。
--backend cuda 仅用 A100 加速 SQD 片段求解，不加速采样、全局密度或梯度。

梯度为 frozen-bath（固定浴）近似，缺少完整浴轨道和片段投影响应，
全局 Λ 不会使其成为完整重建 EWF 能量的严格导数。gradient-check 用多个
步长重建嵌入做中心差分；固定种子不能消除采样或空间变化的不连续。
优化力阈值为 0.1 eV/Å，默认最多 25 步；optimizer_converged 仅表示
近似力满足优化器停止条件，不证明完整能量达到驻点，也不保证动力学能量守恒。

--output 保存实际配置、结构、收敛状态、空间变化、阶段耗时和内存，拒绝覆盖。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import platform
from time import perf_counter

import numpy as np
from pyscf import cc, gto, lib

from tyxonq.applications.chem.algorithms.ewf.solver import build_cluster_rhf, solve_cluster
from tyxonq.applications.chem.algorithms.lucj import LUCJ, initialize_lucj_parameters_from_ccsd
from tyxonq.applications.chem.algorithms.sqd import (
    reverse_bitstring_halves, run_sqd_fermion, sci_result_to_solver_result,
)
from tyxonq.applications.chem.interfaces.ewf_scanner import EWFScanner
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine


def settings(profile="small"):
    """论文配置与缩小验证配置分开；输出使用的实际参数。"""
    config = dict(bath_threshold=1e-5, fci_max_orbitals=12, layers=1,
                  init_maxiter=1000, shots=100000, samples_per_batch=2000,
                  num_batches=4, max_iterations=5, energy_tol=1e-8,
                  occupancies_tol=1e-5, carryover_threshold=1e-4,
                  ext_sqd_threshold=1e-5, ext_sqd_threshold_mode="probability",
                  fmax=0.1, steps=25, seed=7, max_qubits=20, backend="cpu")
    if profile == "small":
        config.update(init_maxiter=50, shots=2048, samples_per_batch=256,
                      num_batches=2, max_iterations=3)
    elif profile != "paper":
        raise ValueError("profile must be 'small' or 'paper'.")
    return config


def make_cluster_solver(config):
    """按片段大小选择 FCI 或从实际 LUCJ 线路采样后执行 SQD。"""
    def cluster_solver(fragment, cluster, mf):
        if cluster.norb <= config["fci_max_orbitals"]:
            return solve_cluster(cluster, mf, "FCI")
        n = cluster.norb
        if 2 * n > config["max_qubits"]:
            raise ValueError(
                f"Local statevector sampling needs {2*n} qubits; limit is "
                f"{config['max_qubits']}. This example does not submit hardware jobs."
            )
        start = perf_counter()
        cluster_mf = build_cluster_rhf(cluster, mf)
        if cluster.nvir:
            cc_solver = cc.CCSD(cluster_mf)
            cc_solver.conv_tol = 1e-10
            cc_solver.kernel()
            if not cc_solver.converged:
                raise RuntimeError("Cluster CCSD initialization did not converge.")
            t2 = cc_solver.t2
        else:
            t2 = np.zeros((cluster.nocc, cluster.nocc, 0, 0))
        cc_seconds = perf_counter() - start
        # 与论文一致，只用 t2 初始化，不额外加入 t1 末尾轨道旋转。
        pairs = ([(p, p + 1) for p in range(n - 1)],
                 [(p, p) for p in range(0, min(n, 16), 4)])
        start = perf_counter()
        params = initialize_lucj_parameters_from_ccsd(
            t2, n_spatial_orbitals=n, n_layers=config["layers"],
            interaction_pairs=pairs, optimize=True, maxiter=config["init_maxiter"],
        )
        circuit = LUCJ(n, sum(cluster.nelec), config["layers"],
                       interaction_pairs=pairs).get_circuit(params)
        init_seconds = perf_counter() - start
        start = perf_counter()
        probabilities = np.asarray(StatevectorEngine().probability(circuit), dtype=float).ravel()
        probabilities /= probabilities.sum()
        # 每个几何重新采样，固定种子减少有限差分的随机差异，不复用旧轨道样本。
        rng = np.random.default_rng(config["seed"] + fragment.id)
        indices = rng.choice(probabilities.size, size=config["shots"], p=probabilities)
        values, multiplicities = np.unique(indices, return_counts=True)
        counts = {reverse_bitstring_halves(format(int(i), f"0{2*n}b")): int(k)
                  for i, k in zip(values, multiplicities)}
        sampling_seconds = perf_counter() - start
        iteration_results = []
        def record_iteration(results):
            best = min(results, key=lambda result: result.energy)
            iteration_results.append((best.energy, np.asarray(best.orbital_occupancies)))
        solver_kwargs = {}
        if config["backend"] == "cuda":
            from tyxonq.applications.chem.algorithms.sqd.gpu import make_cuda_sci_solver
            solver_kwargs["sci_solver"] = make_cuda_sci_solver(
                threads=1, chunk_size=128, tolerance=1e-10, max_cycles=100,
                max_space=24, operator_residual_tolerance=1e-7, rdm_energy_tolerance=1e-8,
            )
        start = perf_counter()
        hf = (1 << cluster.nocc) - 1
        sci_result = run_sqd_fermion(
            cluster.h_eff, cluster.eri, counts, config["samples_per_batch"],
            n, cluster.nelec, num_batches=config["num_batches"],
            max_iterations=config["max_iterations"], energy_tol=config["energy_tol"],
            occupancies_tol=config["occupancies_tol"], initial_occupancies=None,
            include_configurations=([hf], [hf]), average_batch_occupancies=False,
            carryover_threshold=config["carryover_threshold"], ext_sqd=True,
            ext_sqd_threshold=config["ext_sqd_threshold"],
            ext_sqd_threshold_mode=config["ext_sqd_threshold_mode"],
            seed=config["seed"] + fragment.id, callback=record_iteration, **solver_kwargs,
        )
        sqd_seconds = perf_counter() - start
        recovery_converged = len(iteration_results) > 1 and (
            abs(iteration_results[-1][0] - iteration_results[-2][0]) < config["energy_tol"]
            and np.max(np.abs(iteration_results[-1][1] - iteration_results[-2][1]))
            < config["occupancies_tol"]
        )
        state = sci_result.sci_state
        signature = hashlib.sha256()
        for strings in (state.ci_strs_a, state.ci_strs_b):
            signature.update(np.asarray(strings, dtype=np.int64).tobytes())
        result = sci_result_to_solver_result(sci_result, cluster, include_cisd=False)
        return replace(result, diagnostics={
            "source": "local_lucj_statevector", "shots": config["shots"],
            "seed": config["seed"] + fragment.id, "backend": config["backend"],
            "ccsd_seconds": cc_seconds, "lucj_seconds": init_seconds,
            "sampling_seconds": sampling_seconds, "sqd_seconds": sqd_seconds,
            "statevector_bytes": int(16 * probabilities.size),
            "selected_dimensions": list(state.amplitudes.shape),
            "selected_space_signature": signature.hexdigest(),
            "recovery_iterations": len(iteration_results),
            "recovery_converged": bool(recovery_converged),
            "iteration_energies_hartree": [float(item[0]) for item in iteration_results],
            "spin_square": float(sci_result.spin_square()),
        })
    return cluster_solver


def make_scanner(mol, config, assembly="rdm_t_lambda"):
    """固定计算配方，每个新几何重建原子片段与浴轨道。"""
    return EWFScanner(mol, ewf_options={
        "bath_options": {"threshold": config["bath_threshold"]},
        "energy_functional": "density", "density_assembly": assembly,
        "cluster_solver": make_cluster_solver(config),
    })


def gradient_check(scanner, coords, steps=(1e-3, 5e-4, 2.5e-4)):
    """重建嵌入的中心差分；偏差是近似诊断，不据此修改力阈值。"""
    energy, gradient = scanner(coords)
    diagnostics = []
    for step in steps:
        numeric = np.zeros_like(coords)
        start = len(scanner.history)
        for atom in range(len(coords)):
            for axis in range(3):
                displacement = np.zeros_like(coords)
                displacement[atom, axis] = step
                numeric[atom, axis] = (
                    scanner.energy(coords + displacement) - scanner.energy(coords - displacement)
                ) / (2 * step)
        error = numeric - gradient
        diagnostics.append({"step_bohr": step, "max_error_hartree_per_bohr": float(np.max(abs(error))),
                            "rms_error_hartree_per_bohr": float(np.sqrt(np.mean(error**2))),
                            "finite_difference": numeric.tolist(),
                            "spaces_changed": any(x["spaces_changed"] for x in scanner.history[start:])})
    scanner(coords)
    return {"energy_hartree": energy, "analytic_gradient": gradient.tolist(),
            "approximation": "frozen_bath_and_projectors", "rebuilt_embedding_checks": diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("singlepoint", "gradient-check", "optimize"), nargs="?", default="singlepoint")
    parser.add_argument("--profile", choices=("small", "paper"), default="small")
    parser.add_argument("--force-sqd", action="store_true", help="所有片段使用实际 LUCJ 采样与 SQD")
    parser.add_argument("--molecule", choices=("h2", "lih", "h2o"), default="h2o")
    parser.add_argument("--assembly", choices=("rdm_t", "rdm_t_lambda"), default="rdm_t_lambda")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--output", help="可选 JSON 结果路径；省略则仅输出到终端")
    args = parser.parse_args()
    lib.num_threads(1)
    config = settings(args.profile)
    config.update(backend=args.backend, steps=args.steps)
    if args.force_sqd:
        config["fci_max_orbitals"] = 0
    geometries = {"h2": "H 0 0 0; H 0 0 0.78", "lih": "Li 0 0 0; H 0 0 1.65",
                  "h2o": "O 0 0 0; H 0 0.8 0.65; H 0 -0.8 0.65"}
    mol = gto.M(atom=geometries[args.molecule], basis="sto-3g", unit="Angstrom", verbose=0)
    scanner = make_scanner(mol, config, args.assembly)
    coords = mol.atom_coords()
    print(json.dumps({"mode": args.mode, "profile": args.profile, "config": config}, ensure_ascii=False), flush=True)
    if args.mode == "singlepoint":
        result = {"energy_hartree": scanner.energy(coords)}
    elif args.mode == "gradient-check":
        result = gradient_check(scanner, coords)
    else:
        # ASE/Sella 仅在需要几何优化时导入。
        from ase import Atoms
        from sella import Sella
        from tyxonq.applications.chem.interfaces.ewf_calculator import EWFCalculator
        from tyxonq.applications.chem.interfaces.ase_calculator import BOHR_TO_ANGSTROM
        atoms = Atoms(scanner.symbols, positions=coords * BOHR_TO_ANGSTROM)
        atoms.calc = EWFCalculator(scanner)
        optimizer = Sella(atoms, order=0, internal=True, logfile=None)
        converged = optimizer.run(fmax=config["fmax"], steps=config["steps"])
        final_coords = atoms.get_positions() / BOHR_TO_ANGSTROM
        result = {"optimizer_converged": bool(converged), "steps": optimizer.nsteps,
                  "stationarity_of_rebuilt_energy_verified": False,
                  "energy_hartree": scanner.energy(final_coords),
                  "max_force_ev_per_angstrom": float(np.max(np.linalg.norm(atoms.get_forces(), axis=1))),
                  "positions_angstrom": atoms.get_positions().tolist(),
                  "rms_displacement_angstrom": float(np.sqrt(np.mean((atoms.get_positions() - coords * BOHR_TO_ANGSTROM)**2)))}
    result.update(config=config, profile=args.profile, molecule=args.molecule, basis="sto-3g",
                  assembly=args.assembly, approximation="frozen_bath_and_projectors", history=scanner.history)
    packages = ["numpy", "pyscf"] + (["ase", "sella"] if args.mode == "optimize" else [])
    result["environment"] = {"python": platform.python_version(),
                             **{name: version(name) for name in packages}}
    rendered = json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    print(rendered)
    if args.output:
        from pathlib import Path
        with Path(args.output).open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")


if __name__ == "__main__":
    main()
