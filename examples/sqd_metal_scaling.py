"""Three-way Metal, PyTorch CPU and PyTorch MPS selected-CI benchmarks.

Timed regions exclude setup/JIT and independent FP64 golden checks. Metal, CPU
FP32 and MPS use identical fixed solver workloads for break-even comparisons;
normal convergence is measured separately. PyTorch MPS is only a benchmark
reference, not an SQD fallback.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
from pyscf.fci import cistring, direct_spin1, selected_ci

from tyxonq.numerics.sbd.metal import (
    MetalSelectedCIContractor, absorb_h1e, davidson_lowest, make_hdiag,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"


def _configure_metal_matmul(backend: str) -> None:
    if backend == "custom":
        return
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise RuntimeError("mlx-full-fp32 requires MLX_ENABLE_TF32=0 before process startup")
    import mlx.core as mx
    from tyxonq.numerics.sbd.metal import solver

    # Process-local benchmark A/B only; the production SBD default is unchanged.
    solver._metal_matmul = solver.metal_gpu_execution(mx.matmul)


def _timing(values: list[float]) -> dict[str, float | list[float]]:
    return {"median_seconds": statistics.median(values), "min_seconds": min(values),
            "max_seconds": max(values), "samples_seconds": values}


def _problem(norb: int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(20261118 + norb)
    h1e = rng.normal(size=(norb, norb))
    h1e = (h1e + h1e.T) / 2
    factors = rng.normal(size=(max(3, norb // 2), norb, norb))
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    eri = np.einsum("lpq,lrs->pqrs", factors, factors)
    strings = (
        np.asarray(cistring.make_strings(range(norb), 2), dtype=np.int32),
        np.asarray(cistring.make_strings(range(norb), 2), dtype=np.int32),
    )
    coefficients = rng.normal(size=tuple(map(len, strings))).astype(np.float32)
    coefficients /= np.linalg.norm(coefficients)
    return h1e, eri, strings, coefficients


def _golden(h1e: np.ndarray, eri: np.ndarray, strings: tuple[np.ndarray, np.ndarray],
            coefficients: np.ndarray) -> tuple[np.ndarray, float]:
    vector = selected_ci._as_SCIvector(np.asarray(coefficients, dtype=np.float64), strings)
    expected = selected_ci.contract_2e(
        direct_spin1.absorb_h1e(
            h1e, eri, h1e.shape[0], (2, 2), 0.5,  # pyright: ignore[reportArgumentType]
        ), vector, h1e.shape[0], (2, 2))
    return expected, float(np.linalg.norm(expected))


def _solver_fields(result) -> dict[str, float | int | bool]:
    return {"iterations": result.iterations, "h_matvecs": result.h_matvecs,
            "residual_norm": result.residual_norm, "converged": result.converged,
            "energy": result.energy}


def _normalized_host(coefficients) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    return values / np.linalg.norm(values)


def run_case(norb: int, args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    import torch
    from tyxonq.applications.chem.algorithms.sqd.gpu.davidson import (
        davidson_lowest as torch_davidson,
    )
    from tyxonq.applications.chem.algorithms.sqd.gpu.selected_ci import (
        TorchSelectedCIContractor, absorb_h1e_torch, make_hdiag_torch,
    )

    if not torch.mps.is_available():
        raise RuntimeError("PyTorch MPS is required for the three-way benchmark")
    torch.set_num_threads(args.threads)
    h1e, eri, strings, coefficients = _problem(norb)
    links = selected_ci._all_linkstr_index(strings, norb, (2, 2))
    shape = tuple(map(len, strings))
    report = {
        "norb": norb, "shape": shape, "determinants": int(np.prod(shape)),
        "input_seed": 20261118 + norb,
        "precision": {"metal": "float32", "mps": "float32", "cpu_hamiltonian": "float32",
                      "cpu32_solver": "float32", "cpu64_solver": "float64", "golden": "float64"},
        "fixed_solver_tolerance": args.fixed_tolerance,
        "workload": {"max_cycles": args.max_cycles, "max_space": args.max_space,
                     "chunk_size": args.chunk_size, "repeats": args.repeats},
        "estimated_solver_vector_bytes": int(np.prod(shape)) * args.max_space * 4,
    }

    with mx.stream(mx.gpu):
        metal = MetalSelectedCIContractor(absorb_h1e(h1e, eri, (2, 2)), links, (2, 2),
                                          chunk_size=args.chunk_size)
        diagonal_metal = make_hdiag(mx.array(h1e.astype(np.float32)),
                                    mx.array(eri.astype(np.float32)), strings)
        coefficients_metal = mx.array(coefficients)
        metal.contract(coefficients_metal)
        metal_samples = []
        metal_image = None
        for _ in range(args.repeats):
            started = time.perf_counter()
            metal_image = metal.contract(coefficients_metal)
            mx.eval(metal_image)
            metal_samples.append(time.perf_counter() - started)
        assert metal_image is not None
        metal_image_np = np.asarray(metal_image, dtype=np.float64)
        report["metal_hamiltonian"] = _timing(metal_samples)
        report["metal_execution"] = dict(metal.execution)

        initial_metal = mx.zeros((int(diagonal_metal.size),), dtype=mx.float32)
        initial_index = int(np.asarray(mx.argmin(diagonal_metal)).item())
        initial_metal = initial_metal.at[initial_index].add(1.0)

        def metal_matvec(vector):
            return metal.contract(vector.reshape(shape)).reshape(-1)

        def solve_metal(tolerance):
            return davidson_lowest(
                metal_matvec, diagonal_metal.reshape(-1), initial_vector=initial_metal,
                tolerance=tolerance, max_cycles=args.max_cycles, max_space=args.max_space)

        solve_metal(args.fixed_tolerance)
        metal_solver_samples = []
        metal_solved = None
        for _ in range(args.repeats):
            started = time.perf_counter()
            metal_solved = solve_metal(args.fixed_tolerance)
            mx.eval(metal_solved.coefficients)
            metal_solver_samples.append(time.perf_counter() - started)
        assert metal_solved is not None
        report["metal_solver"] = {**_timing(metal_solver_samples), **_solver_fields(metal_solved)}
        started = time.perf_counter()
        normal_metal = solve_metal(args.tolerance)
        mx.eval(normal_metal.coefficients)
        normal_metal_seconds = time.perf_counter() - started
        normal_metal_coefficients = _normalized_host(normal_metal.coefficients)

    h1e32 = torch.from_numpy(h1e.astype(np.float32))
    eri32 = torch.from_numpy(eri.astype(np.float32))
    cpu32 = TorchSelectedCIContractor(
        absorb_h1e_torch(h1e32, eri32, (2, 2)), links, (2, 2), chunk_size=args.chunk_size)
    coefficients_cpu32 = torch.from_numpy(coefficients.copy())
    with torch.inference_mode():
        cpu32.contract(coefficients_cpu32)
        cpu32_samples = []
        cpu32_image = None
        for _ in range(args.repeats):
            started = time.perf_counter()
            cpu32_image = cpu32.contract(coefficients_cpu32)
            cpu32_samples.append(time.perf_counter() - started)
    assert cpu32_image is not None
    report["cpu32_hamiltonian"] = _timing(cpu32_samples)
    report["hamiltonian_speedup_cpu32_over_metal"] = (
        statistics.median(cpu32_samples) / statistics.median(metal_samples))
    report["cpu32_hamiltonian_max_abs_error"] = float(np.max(np.abs(
        cpu32_image.numpy().astype(np.float64) - metal_image_np)))

    diagonal_cpu32 = make_hdiag_torch(h1e32, eri32, strings).reshape(-1)
    initial_cpu32 = torch.zeros_like(diagonal_cpu32)
    initial_cpu32[int(torch.argmin(diagonal_cpu32).item())] = 1.0

    def cpu32_matvec(vector):
        return cpu32.contract(vector.reshape(shape)).reshape(-1)

    def solve_cpu32(tolerance):
        return torch_davidson(
            cpu32_matvec, diagonal_cpu32, initial_vector=initial_cpu32,
            tolerance=tolerance, max_cycles=args.max_cycles, max_space=args.max_space)

    solve_cpu32(args.fixed_tolerance)
    cpu32_solver_samples = []
    cpu32_solved = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        cpu32_solved = solve_cpu32(args.fixed_tolerance)
        cpu32_solver_samples.append(time.perf_counter() - started)
    assert cpu32_solved is not None
    report["cpu32_solver"] = {**_timing(cpu32_solver_samples), **_solver_fields(cpu32_solved)}

    mps_device = torch.device("mps")
    h1e_mps = torch.from_numpy(h1e.astype(np.float32)).to(mps_device)
    eri_mps = torch.from_numpy(eri.astype(np.float32)).to(mps_device)
    mps = TorchSelectedCIContractor(absorb_h1e_torch(h1e_mps, eri_mps, (2, 2)), links,
                                    (2, 2), chunk_size=args.chunk_size)
    coefficients_mps = torch.from_numpy(coefficients.copy()).to(mps_device)
    with torch.inference_mode():
        mps.contract(coefficients_mps)
        torch.mps.synchronize()
        mps_samples = []
        mps_image = None
        for _ in range(args.repeats):
            torch.mps.synchronize()
            started = time.perf_counter()
            mps_image = mps.contract(coefficients_mps)
            torch.mps.synchronize()
            mps_samples.append(time.perf_counter() - started)
    assert mps_image is not None
    mps_image_np = mps_image.cpu().numpy().astype(np.float64)
    report["mps_hamiltonian"] = {**_timing(mps_samples), "device": str(mps_image.device),
                                 "cpu_fallback_enabled": False}

    diagonal_mps = make_hdiag_torch(h1e_mps, eri_mps, strings).reshape(-1)
    initial_mps = torch.zeros_like(diagonal_mps)
    initial_mps[int(torch.argmin(diagonal_mps).item())] = 1.0

    def mps_matvec(vector):
        return mps.contract(vector.reshape(shape)).reshape(-1)

    def solve_mps(tolerance):
        return torch_davidson(mps_matvec, diagonal_mps, initial_vector=initial_mps,
                              tolerance=tolerance, max_cycles=args.max_cycles,
                              max_space=args.max_space)

    solve_mps(args.fixed_tolerance)
    torch.mps.synchronize()
    mps_solver_samples = []
    mps_solved = None
    for _ in range(args.repeats):
        torch.mps.synchronize()
        started = time.perf_counter()
        mps_solved = solve_mps(args.fixed_tolerance)
        torch.mps.synchronize()
        mps_solver_samples.append(time.perf_counter() - started)
    assert mps_solved is not None
    report["mps_solver"] = {**_timing(mps_solver_samples), **_solver_fields(mps_solved),
                            "device": str(mps_solved.coefficients.device)}
    fixed_solutions = (metal_solved, cpu32_solved, mps_solved)
    if any(solved.iterations != args.max_cycles for solved in fixed_solutions):
        raise RuntimeError("Fixed-workload solver stopped before max_cycles")
    if len({solved.h_matvecs for solved in fixed_solutions}) != 1:
        raise RuntimeError("Metal/CPU32/MPS fixed-workload H@c counts differ")

    torch.mps.synchronize()
    started = time.perf_counter()
    normal_mps = solve_mps(args.tolerance)
    torch.mps.synchronize()
    normal_mps_seconds = time.perf_counter() - started
    normal_mps_coefficients = _normalized_host(normal_mps.coefficients.cpu().numpy())
    report["normal_convergence"] = {
        "metal": {**_solver_fields(normal_metal), "seconds": normal_metal_seconds},
        "mps": {**_solver_fields(normal_mps), "seconds": normal_mps_seconds},
    }

    h1e64, eri64 = torch.from_numpy(h1e), torch.from_numpy(eri)
    cpu64 = TorchSelectedCIContractor(absorb_h1e_torch(h1e64, eri64, (2, 2)), links, (2, 2),
                                      chunk_size=args.chunk_size)
    diagonal_cpu = make_hdiag_torch(h1e64, eri64, strings).reshape(-1)
    initial_cpu = torch.zeros_like(diagonal_cpu)
    initial_cpu[int(torch.argmin(diagonal_cpu).item())] = 1.0

    def cpu_matvec(vector):
        return cpu64.contract(vector.reshape(shape)).reshape(-1)

    cpu_solved = torch_davidson(cpu_matvec, diagonal_cpu, initial_vector=initial_cpu,
                                tolerance=args.tolerance, max_cycles=args.max_cycles,
                                max_space=args.max_space)
    cpu_solver_samples = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        cpu_solved = torch_davidson(cpu_matvec, diagonal_cpu, initial_vector=initial_cpu,
                                    tolerance=args.tolerance, max_cycles=args.max_cycles,
                                    max_space=args.max_space)
        cpu_solver_samples.append(time.perf_counter() - started)
    report["cpu64_solver"] = {**_timing(cpu_solver_samples), **_solver_fields(cpu_solved)}

    metal_h_median = statistics.median(metal_samples)
    cpu32_h_median = statistics.median(cpu32_samples)
    mps_h_median = statistics.median(mps_samples)
    metal_solver_median = statistics.median(metal_solver_samples)
    cpu32_solver_median = statistics.median(cpu32_solver_samples)
    mps_solver_median = statistics.median(mps_solver_samples)
    report["metal_to_mps_hamiltonian_ratio"] = metal_h_median / mps_h_median
    report["metal_to_mps_solver_ratio"] = metal_solver_median / mps_solver_median
    report["mps_over_metal_hamiltonian"] = mps_h_median / metal_h_median
    report["mps_over_metal_solver"] = mps_solver_median / metal_solver_median
    report["metal_to_cpu32_hamiltonian_ratio"] = metal_h_median / cpu32_h_median
    report["metal_to_cpu32_solver_ratio"] = metal_solver_median / cpu32_solver_median
    report["cpu32_over_metal_hamiltonian_speedup"] = cpu32_h_median / metal_h_median
    report["cpu32_over_metal_solver_speedup"] = cpu32_solver_median / metal_solver_median
    report["break_even_target"] = 1.0
    report["hamiltonian_break_even_reached"] = report["metal_to_cpu32_hamiltonian_ratio"] <= 1.0
    report["solver_break_even_reached"] = report["metal_to_cpu32_solver_ratio"] <= 1.0
    report["mps_parity_target"] = 1.10
    report["mps_parity_reached"] = (
        report["metal_to_mps_hamiltonian_ratio"] <= 1.10 and
        report["metal_to_mps_solver_ratio"] <= 1.10)

    expected, expected_norm = _golden(h1e, eri, strings, coefficients)
    golden: dict[str, float | bool] = {"tolerance": args.golden_tolerance}
    report["golden"] = golden
    for name, actual in (("metal", metal_image_np), ("mps", mps_image_np)):
        golden[f"{name}_hamiltonian_max_abs_error"] = float(np.max(np.abs(actual - expected)))
        relative = float(np.linalg.norm(actual - expected) / max(expected_norm, 1e-30))
        golden[f"{name}_hamiltonian_relative_l2_error"] = relative
        if relative > args.golden_tolerance:
            raise RuntimeError(f"{name} H@c failed FP64 golden: {golden}")

    for name, solved, coeffs in (
            ("metal", metal_solved, _normalized_host(metal_solved.coefficients)),
            ("cpu32", cpu32_solved, _normalized_host(cpu32_solved.coefficients.numpy())),
            ("mps", mps_solved, _normalized_host(mps_solved.coefficients.cpu().numpy()))):
        final, _ = _golden(h1e, eri, strings, coeffs)
        golden[f"{name}_fixed_solver_independent_residual"] = float(np.linalg.norm(
            final - solved.energy * coeffs))
    for name, solved, coeffs in (("metal", normal_metal, normal_metal_coefficients),
                                 ("mps", normal_mps, normal_mps_coefficients)):
        final, _ = _golden(h1e, eri, strings, coeffs)
        residual = float(np.linalg.norm(final - solved.energy * coeffs))
        golden[f"{name}_normal_solver_independent_residual"] = residual
        golden[f"{name}_normal_external_residual_pass"] = residual <= args.operator_residual_tolerance
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norb", nargs="+", type=int, default=[4, 6, 8, 10])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--metal-matmul", choices=("custom", "mlx-full-fp32"), default="custom")
    parser.add_argument("--tolerance", type=float, default=2e-5)
    parser.add_argument("--fixed-tolerance", type=float, default=1e-12)
    parser.add_argument("--max-cycles", type=int, default=40)
    parser.add_argument("--max-space", type=int, default=12)
    parser.add_argument("--golden-tolerance", type=float, default=2e-5)
    parser.add_argument("--operator-residual-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.repeats, args.threads, args.chunk_size, args.max_cycles, args.max_space) < 1:
        parser.error("repeat and workload limits must be positive")
    if min(args.tolerance, args.fixed_tolerance, args.golden_tolerance,
           args.operator_residual_tolerance) <= 0:
        parser.error("tolerances must be positive")
    if min(args.norb) < 2:
        parser.error("norb must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    _configure_metal_matmul(args.metal_matmul)
    report = {
        "protocol": "Warmup; mx.eval/torch.mps.synchronize per sample; fixed Metal/CPU32/MPS workload; external FP64 golden",
        "repeats": args.repeats, "threads": args.threads,
        "metal_matmul_backend": args.metal_matmul,
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32"),
        "mps_cpu_fallback_enabled": False,
        "packages": {name: importlib.metadata.version(name) for name in ("mlx", "numpy", "torch", "pyscf")},
        "cases": [run_case(norb, args) for norb in args.norb],
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
