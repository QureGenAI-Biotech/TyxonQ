"""SKQD 分步示例：体系准备、线路采样、SQD 后处理及自然轨道第二轮。

需要 TyxonQ 的 NumPy、SciPy、PySCF 和 OpenFermion 依赖；采样使用本地模拟。
从仓库根目录运行：
    export PYTHONPATH=src
    python examples/SKQD/workflow.py prepare --config examples/SKQD/siam.json --output /tmp/skqd-siam
    python examples/SKQD/workflow.py sample --input /tmp/skqd-siam
    python examples/SKQD/workflow.py postprocess --input /tmp/skqd-siam --exact

prepare 保存积分、参数与可重载线路，拒绝覆盖已有目录；sample 保存 counts、
seed 和 shots；postprocess 保存能量、RDM 和 SCI 状态。--exact 仅适用于
可以承受完整 FCI 成本的小体系，普通后处理不要求计算完整 FCI。

siam.json 提供单杂质 Anderson 模型；impurity.json、four_impurity.json
提供单杂质与四杂质的自然轨道算例；h2.json、lih.json 提供分子 SqDRIFT 算例。
rotate --input <第一轮目录> --output <第二轮目录> 用第一轮 RDM 变基，
第二轮必须重新采样，不能复用旧基底的 bitstring。
estimate --config <配置> 仅做静态成本估算，不代表已完成对应规模的实验。
各子命令的参数可通过 <子命令> --help 查看。"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from tyxonq.applications.chem.algorithms.skqd import (
    SKQDEnsemble, build_local_natural_orbitals, build_sqdrift_circuits,
    build_trotter_krylov_circuits, correlations_from_rdms, fit_gaussian_evolution,
    postprocess_skqd, prepare_slater_reference, rdms_to_original_basis,
    rotate_integrals, sample_statevector_ensemble,
)
from tyxonq.applications.chem.algorithms.sqd import solve_expanded_subspace

try:
    from .models import build_problem, near_fermi_reference
except ImportError:
    from models import build_problem, near_fermi_reference


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    """数组仅在示例结果文件转成列表，核心算法仍保留 numpy 数据。"""
    def encode(value):
        if isinstance(value, np.ndarray):
            return np.real_if_close(value).tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"cannot serialize {type(value).__name__}")
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=encode) + "\n", encoding="utf-8")


def missing_inputs(config):
    """列出配置中尚未填写的必要输入。"""
    missing = []
    for section, key in (("model", "path"), ("model", "bath_energies"),
                         ("evolution", "time_step"), ("evolution", "krylov_indices")):
        if key in config[section] and config[section][key] is None:
            missing.append(f"{section}.{key}")
    return missing


def estimate(config):
    """只读取参数，不构造积分、线路或设备连接；CI 维度是完整扇区上界。"""
    n, (a, b) = config["model"]["norb"], config["model"]["nelec"]
    evolution = config["evolution"]
    times = evolution.get("krylov_indices")
    repeats = evolution.get("num_randomizations", 1)
    count = None if times is None else len(times) * repeats
    return {"name": config["name"], "norb": n, "qubits": 2 * n,
            "full_ci_dimension": comb(n, a) * comb(n, b),
            "dense_eri_bytes": 8 * n ** 4, "statevector_bytes": 16 * 2 ** (2 * n),
            "circuits": count, "shots": None if count is None else count * config["sampling"]["shots"],
            "sampled_groups_per_circuit": evolution.get("num_sampled_groups"),
            "missing_inputs": missing_inputs(config),
            "note": "线路数不等于编译后资源；完整 CI 和 statevector 数值仅作成本说明。"}


def _ensemble(problem, config, basis_id):
    """所有数学调用 src 公开入口；这里只选择实验参数和参考态。"""
    n, nelec = int(problem["norb"]), tuple(int(x) for x in problem["nelec"])
    reference = (near_fermi_reference(n, nelec) if config.get("reference") == "near_fermi"
                 else prepare_slater_reference(n, nelec))
    options = deepcopy(config["evolution"])
    method = options.pop("method")
    depth = options.pop("fit_depth", None)
    fit = None
    if method == "trotter":
        if depth is not None:
            target = expm(-1j * options["time_step"] / options.get("trotter_substeps", 1) * problem["h1e"])
            fit = fit_gaussian_evolution(target, depth=depth, seed=config["sampling"]["seed"])
        ensemble = build_trotter_krylov_circuits(
            problem["h1e"], problem["eri"], n, nelec, initial_state=reference,
            interaction_rotation=problem["interaction_rotation"], gaussian_fit=fit,
            basis_id=basis_id, **options)
    elif method == "sqdrift":
        ensemble = build_sqdrift_circuits(problem["h1e"], problem["eri"], n, nelec,
                                        initial_state=reference, basis_id=basis_id, **options)
    else:
        raise ValueError(f"unknown evolution method: {method}")
    fit_info = None if fit is None else {"depth": fit.depth, "error": fit.error,
                "initial_error": fit.initial_error, "converged": fit.converged,
                "iterations": fit.iterations, "message": fit.message}
    return ensemble, fit_info


def _save_prepared(problem, config, output, blocks, basis_id):
    output = Path(output)
    if (output / "ensemble.json").exists():
        raise ValueError("output already contains an ensemble; use a new output directory")
    ensemble, fit = _ensemble(problem, config, basis_id)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "problem.npz", **problem)
    ensemble.save(output / "ensemble.json")
    write_json(output / "config.json", config)
    gate_counts = [Counter(op[0] for op in circuit.ops) for circuit in ensemble.circuits]
    metadata = {"basis_id": basis_id, "natural_orbital_blocks": blocks,
                "fit": fit, "resources": estimate(config), "gate_counts": gate_counts}
    write_json(output / "metadata.json", metadata)
    return metadata


def prepare(config_path, output):
    config = read_json(config_path)
    missing = missing_inputs(config)
    if missing:
        raise ValueError("complete these config fields first: " + ", ".join(missing))
    if config["model"]["kind"] == "fcidump":
        config["model"]["path"] = str((Path(config_path).resolve().parent / config["model"]["path"]).resolve())
    problem, blocks = build_problem(config)
    return _save_prepared(problem, config, output, blocks, "round1")


def sample(input_dir, *, shots=None, seed=None):
    root = Path(input_dir)
    ensemble = SKQDEnsemble.load(root / "ensemble.json")
    options = read_json(root / "config.json")["sampling"]
    if shots is not None:
        options["shots"] = shots
    if seed is not None:
        options["seed"] = seed
    counts = sample_statevector_ensemble(ensemble, **options)
    write_json(root / "counts.json", counts)
    write_json(root / "sampling.json", options)
    return {"circuits": len(counts), "shots": sum(sum(c.values()) for c in counts.values()),
            "seed": options["seed"]}


def postprocess(input_dir, *, exact=False, strict_rank=None, max_determinants=100000):
    root = Path(input_dir)
    problem = dict(np.load(root / "problem.npz", allow_pickle=False))
    h, g, n = problem["h1e"], problem["eri"], int(problem["norb"])
    nelec = tuple(int(x) for x in problem["nelec"])
    config = read_json(root / "config.json")
    ensemble = SKQDEnsemble.load(root / "ensemble.json")
    result = postprocess_skqd(h, g, read_json(root / "counts.json"), ensemble,
                             energy_offset=float(problem["energy_offset"]), **config["sqd"])
    sci = result.sci_result
    one = sci.rdm1s if sci.rdm1s is not None else sci.sci_state.rdm(1)
    two = sci.rdm2s if sci.rdm2s is not None else sci.sci_state.rdm(2)
    summed_two = two[0] + two[1] + two[1].transpose(2, 3, 0, 1) + two[2]
    rdm_energy = np.einsum("pq,qp->", h, one[0] + one[1]) + 0.5 * np.einsum("pqrs,pqrs->", g, summed_two)
    original_one, original_two = rdms_to_original_basis(one, two, problem["original_rotation"])
    correlation = correlations_from_rdms(original_one, original_two)
    np.savez_compressed(root / "rdms.npz", rdm1s=one, rdm2s=two,
                        original_rdm1s=original_one, original_rdm2s=original_two)
    sci.sci_state.save(str(root / "sci_state.npz"))
    summary = {"name": config["name"], "energy_active": result.energy,
               "energy_offset": result.energy_offset, "energy_total": result.total_energy,
               "energy_from_rdms": float(np.real(rdm_energy)),
               "rdm_energy_error": float(abs(rdm_energy - result.energy)),
               "occupancies": np.diagonal(np.asarray(one), axis1=1, axis2=2),
               "spin_square": sci.spin_square(), "statistics": result.statistics,
               "history": result.history, "correlations_original_basis": correlation}
    if exact:
        # 这是显式启用的小体系验证，绝不在大体系默认运行完整 CI。
        dimension = comb(n, nelec[0]) * comb(n, nelec[1])
        if dimension > 100000:
            raise ValueError("exact validation is limited to 100000 determinants")
        from pyscf.fci import direct_spin1
        energy, vector = direct_spin1.kernel(h, g, n, nelec, tol=1e-12, max_cycle=200)
        exact_one, exact_two = direct_spin1.make_rdm12s(vector, n, nelec)
        old_one, old_two = rdms_to_original_basis(exact_one, exact_two, problem["original_rotation"])
        reference_correlations = correlations_from_rdms(old_one, old_two)
        summary.update(exact_energy_active=float(energy), exact_energy_total=float(energy + result.energy_offset),
                       energy_error=float(result.energy - energy),
                       correlation_max_errors={key: float(np.max(np.abs(value - reference_correlations[key])))
                                               for key, value in correlation.items()})
    if strict_rank is not None:
        # 显式选择整个旧 SCI 基底作为 seeds，保证扩展子空间包含基线。
        state = sci.sci_state
        if state.amplitudes.size > max_determinants:
            raise ValueError("baseline SCI dimension exceeds max_determinants")
        seeds = [int(a) | (int(b) << n) for a in state.ci_strs_a for b in state.ci_strs_b]
        extended = solve_expanded_subspace(h, g, seeds, n, nelec, max_rank=strict_rank,
                    max_determinants=max_determinants, energy_offset=result.energy_offset)
        write_json(root / "strict_determinants.json", list(extended.determinants))
        np.savez_compressed(root / "strict_state.npz", amplitudes=extended.amplitudes,
                            rdm1s=extended.rdm1s, rdm2s=extended.rdm2s)
        summary["strict_extension"] = {"max_rank": strict_rank, "dimension": len(extended.determinants),
                        "energy_active": extended.energy, "energy_total": extended.total_energy,
                        "residual_norm": extended.residual_norm}
    write_json(root / "result.json", summary)
    return summary


def rotate(input_dir, output, *, fit_depth=None):
    """由上一轮实际 RDM 变基，再生成待独立采样的新线路，绝不复用旧 counts。"""
    root = Path(input_dir)
    problem = dict(np.load(root / "problem.npz", allow_pickle=False))
    density = np.load(root / "rdms.npz", allow_pickle=False)["rdm1s"].sum(axis=0)
    metadata, config = read_json(root / "metadata.json"), read_json(root / "config.json")
    if config["evolution"]["method"] != "trotter":
        raise ValueError("this two-round impurity example uses Trotter evolution")
    rotation = build_local_natural_orbitals(density, metadata["natural_orbital_blocks"])
    problem["h1e"], problem["eri"] = rotate_integrals(problem["h1e"], problem["eri"], rotation)
    for key in ("interaction_rotation", "original_rotation"):
        problem[key] = problem[key] @ rotation
    config["evolution"]["fit_depth"] = fit_depth
    # 与旧轮区分采样种子；不把不同轮的随机能量要求为逐点单调。
    config["sampling"]["seed"] += 1
    return _save_prepared(problem, config, output, metadata["natural_orbital_blocks"], metadata["basis_id"] + "-local-no")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "estimate"):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
        if command == "prepare":
            item.add_argument("--output", required=True)
    for command in ("sample", "postprocess", "rotate"):
        item = sub.add_parser(command)
        item.add_argument("--input", required=True)
        if command == "sample":
            item.add_argument("--shots", type=int)
            item.add_argument("--seed", type=int)
        elif command == "postprocess":
            item.add_argument("--exact", action="store_true")
            item.add_argument("--strict-rank", type=int, choices=(1, 2, 3))
            item.add_argument("--max-determinants", type=int, default=100000)
        else:
            item.add_argument("--output", required=True)
            item.add_argument("--fit-depth", type=int)
    args = parser.parse_args()
    if args.command == "estimate":
        output = estimate(read_json(args.config))
    elif args.command == "prepare":
        output = prepare(args.config, args.output)
    elif args.command == "sample":
        output = sample(args.input, shots=args.shots, seed=args.seed)
    elif args.command == "rotate":
        output = rotate(args.input, args.output, fit_depth=args.fit_depth)
    else:
        result = postprocess(args.input, exact=args.exact, strict_rank=args.strict_rank,
                             max_determinants=args.max_determinants)
        output = {k: result[k] for k in ("energy_total", "rdm_energy_error", "statistics")}
    if "gate_counts" in output:
        output = {key: value for key, value in output.items() if key != "gate_counts"}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
