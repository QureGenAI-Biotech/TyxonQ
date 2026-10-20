"""CUDA 求解器共享的输入、初猜、密度和独立 CPU 验收。"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

from ..fermion import SCIResult, SCIState


@contextmanager
def validation_failure_snapshot(path, ci_strings, h1e, eri, norb, nelec, **metadata):
    """验收失败时原子写出可恢复输入；未指定路径时不写文件。

    调用方只登记当前数组引用，成功路径不复制设备数据。旧诊断键始终保留；
    尚未生成的算符像用空数组表示，其余恢复信息放入 metadata_json。
    """
    snapshot = {}
    try:
        yield snapshot
    except Exception as error:
        if path is not None:
            try:
                arrays = {}
                unavailable = {}
                for key, value in snapshot.items():
                    try:
                        if hasattr(value, "detach"):
                            value = value.detach().cpu().numpy()
                        arrays[key] = np.asarray(value)
                    except Exception as transfer_error:
                        # CUDA 错误后可能无法读显存，仍保存输入和其余可读数组。
                        unavailable[key] = str(transfer_error)
                for key in ("coefficients", "gpu_image", "cpu_image"):
                    arrays.setdefault(key, np.empty(0, dtype=np.float64))
                norm = float(np.linalg.norm(arrays["coefficients"]))
                # 同步缩放算符像，确保恢复后的 c 与 Hc 对应同一个归一化状态。
                if np.isfinite(norm) and norm > 0:
                    for key in ("coefficients", "gpu_image", "cpu_image", "gpu_bare_image"):
                        if key in arrays:
                            arrays[key] = arrays[key] / norm
                metadata = {**metadata, "norb": int(norb), "nelec": list(nelec),
                            "coefficients_role": str(arrays.pop("coefficients_role", "solver_result")),
                            "unavailable_arrays": unavailable,
                            "coefficient_norm_before_snapshot": norm,
                            "coefficients_normalized": bool(np.isfinite(norm) and norm > 0),
                            "failure_type": type(error).__name__, "failure_message": str(error)}
                arrays.update(ci_strs_a=ci_strings[0], ci_strs_b=ci_strings[1],
                              h1e=h1e, eri=eri, metadata_json=np.asarray(json.dumps(metadata)))
                debug_path = Path(path)
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = debug_path.with_suffix(debug_path.suffix + ".tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, **arrays)
                temporary.replace(debug_path)
            except Exception as save_error:
                # 写诊断失败不能覆盖真正的数值/CUDA 异常。
                if hasattr(error, "add_note"):
                    error.add_note(f"Unable to save failure snapshot: {save_error}")
        raise


def normalize_problem(ci_strings, h1e, eri, norb, nelec):
    """检查 restricted 实积分，并按数值顺序去重行列式标签。"""
    if not isinstance(norb, (int, np.integer)) or not 1 <= norb <= 63:
        raise ValueError("norb must be an integer in [1, 63] for int64 determinants.")
    if len(nelec) != 2 or any(int(n) != n or not 0 <= n <= norb for n in nelec):
        raise ValueError("nelec must contain two valid electron counts.")
    nelec = tuple(int(n) for n in nelec)
    if sum(nelec) == 0:
        raise ValueError("At least one electron is required.")
    arrays = []
    for array, shape in ((h1e, (norb, norb)), (eri, (norb,) * 4)):
        array = np.asarray(array)
        if np.iscomplexobj(array):
            raise NotImplementedError("Only real restricted integrals are supported.")
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Integrals must be finite with shape {shape}.")
        arrays.append(np.ascontiguousarray(array, dtype=np.float64))
    # SBD 会压缩为 restricted 的八重对称积分；不允许压缩过程静默丢掉
    # 非对称分量。这里仅检查，不修改用户 Hamiltonian。
    if not np.allclose(arrays[0], arrays[0].T, atol=1e-12, rtol=1e-12):
        raise ValueError("Restricted h1e must be real symmetric.")
    for permutation in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        if not np.allclose(arrays[1], arrays[1].transpose(permutation), atol=1e-12, rtol=1e-12):
            raise ValueError("Restricted eri must obey chemist-order eightfold symmetry.")
    if len(ci_strings) != 2:
        raise ValueError("ci_strings must contain alpha and beta labels.")
    labels = []
    for values, count in zip(ci_strings, nelec):
        values = np.asarray(values)
        if values.ndim != 1 or not values.size or values.dtype.kind not in "iu":
            raise ValueError("Determinant labels must be nonempty integer vectors.")
        if any(int(x) < 0 or int(x) >= 1 << norb or int(x).bit_count() != count for x in values):
            raise ValueError("Determinant labels disagree with norb/nelec.")
        labels.append(np.unique(values.astype(np.int64)))
    return tuple(labels), arrays[0], arrays[1], int(norb), nelec


def project_initial_state(initial_state, ci_strings, norb, nelec):
    """按标签投影显式初猜；与新空间无交集时沿用求解器默认初猜。"""
    if initial_state is None:
        return None
    if not isinstance(initial_state, SCIState):
        raise TypeError("initial_state must be an SCIState or None.")
    if initial_state.norb != norb or tuple(initial_state.nelec) != tuple(nelec):
        raise ValueError("initial_state has incompatible norb/nelec.")
    source = np.asarray(initial_state.amplitudes)
    if np.iscomplexobj(source) or not np.isfinite(source).all():
        raise ValueError("initial_state amplitudes must be finite and real.")
    old_labels = (initial_state.ci_strs_a, initial_state.ci_strs_b)
    if any(len(np.unique(x)) != len(x) for x in old_labels):
        raise ValueError("initial_state determinant labels must be unique.")
    projected = np.zeros(tuple(map(len, ci_strings)), dtype=np.float64)
    matches = [np.intersect1d(new, old, return_indices=True)[1:]
               for new, old in zip(ci_strings, old_labels)]
    projected[np.ix_(matches[0][0], matches[1][0])] = source[np.ix_(matches[0][1], matches[1][1])]
    norm = np.linalg.norm(projected)
    return None if norm == 0 else projected / norm


def independent_cpu_validation(coefficients, ci_strings, h1e, eri, norb, nelec,
                               *, spin_sq, spin_shift, threads, include_operator=True):
    """从干净的无 CUDA 进程调用 PySCF；不复用 GPU 算符或 RDM。"""
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="tyxonq_sci_validation_") as directory:
        path = Path(directory)
        np.savez(path / "input.npz", coefficients=np.asarray(coefficients).reshape(-1),
                 ci_strs_a=ci_strings[0], ci_strs_b=ci_strings[1], h1e=h1e, eri=eri,
                 metadata_json=np.asarray(json.dumps({"norb": norb, "nelec": list(nelec),
                                                       "spin_sq": spin_sq, "spin_shift": spin_shift,
                                                       "include_operator": include_operator})))
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(threads),
                       "OPENBLAS_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
        read_fd, write_fd = os.pipe()
        try:
            # 写端仅属于当前 worker；terminate/kill 后由操作系统关闭。
            # 传入现成读端，避免 communicate() 提前关闭 stdin 触发假退出。
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("cpu_validate_sci.py")),
                 "--input", str(path / "input.npz"), "--output", str(path / "output.npz"),
                 "--threads", str(threads), "--watch-parent-stdin"],
                stdin=read_fd, close_fds=True, env=environment, check=False,
                capture_output=True, text=True, timeout=600)
        finally:
            os.close(read_fd)
            os.close(write_fd)
        if completed.returncode:
            raise RuntimeError(f"Independent PySCF validation failed: {completed.stderr[-2000:]}")
        with np.load(path / "output.npz", allow_pickle=False) as archive:
            result = {key: np.asarray(archive[key]).copy() for key in archive.files}
    result["seconds"] = perf_counter() - started
    return result


def rdms_from_cpu(data):
    """从进程间结果读取 PySCF 自旋分辨 RDM。"""
    return ((data["rdm1_alpha"], data["rdm1_beta"]),
            (data["rdm2_aa"], data["rdm2_ab"], data["rdm2_bb"]))


def assemble_result(coefficients, ci_strings, h1e, eri, norb, nelec, rdm1s, rdm2s):
    """返回物理电子能；软自旋惩罚不计入 SCIResult.energy。"""
    rdm1 = rdm1s[0] + rdm1s[1]
    rdm2 = rdm2s[0] + rdm2s[2] + rdm2s[1] + rdm2s[1].transpose(2, 3, 0, 1)
    energy = float(np.einsum("pq,pq->", h1e, rdm1) + 0.5 * np.einsum("pqrs,pqrs->", eri, rdm2))
    return SCIResult(energy=energy, sci_state=SCIState(
        np.asarray(coefficients).reshape(tuple(map(len, ci_strings))), *ci_strings, norb, nelec),
        orbital_occupancies=tuple(np.diag(x).copy() for x in rdm1s),
        rdm1=rdm1, rdm2=rdm2, rdm1s=rdm1s, rdm2s=rdm2s, converged=True)


def validate_rdms(result, bare_rayleigh, *, energy_tolerance, rdm_tolerance=1e-6):
    """验收归一化、密度厄米性、粒子数及自旋分块收缩关系。"""
    a, b = result.rdm1s
    aa, ab, bb = result.rdm2s
    na, nb = result.sci_state.nelec
    blocks = (a, b, aa, ab, bb)
    if (any(not np.isfinite(block).all() for block in blocks)
            or not np.isfinite(result.sci_state.amplitudes).all()
            or not np.isfinite(result.energy) or not np.isfinite(bare_rayleigh)):
        raise RuntimeError("Wavefunction, energy or RDM contains nonfinite values.")
    errors = {
        "normalization_abs_error": abs(np.linalg.norm(result.sci_state.amplitudes) - 1.0),
        "rdm1_trace_abs_error": max(abs(np.trace(a) - na), abs(np.trace(b) - nb)),
        "rdm_hermiticity_max_abs_error": max(
            np.max(abs(a - a.T)), np.max(abs(b - b.T)),
            *(np.max(abs(x - x.transpose(1, 0, 3, 2))) for x in (aa, ab, bb))),
        "rdm_contraction_max_abs_error": max(
            np.max(abs(np.einsum("pqrr->pq", aa) - (na - 1) * a)),
            np.max(abs(np.einsum("pqrr->pq", bb) - (nb - 1) * b)),
            np.max(abs(np.einsum("pqrr->pq", ab) - nb * a)),
            np.max(abs(np.einsum("pprs->rs", ab) - na * b))),
        "rdm_energy_abs_error_vs_rayleigh": abs(result.energy - bare_rayleigh),
    }
    if errors["normalization_abs_error"] > 1e-8:
        raise RuntimeError(f"CI normalization validation failed: {errors}")
    if any(errors[key] > rdm_tolerance for key in
           ("rdm1_trace_abs_error", "rdm_hermiticity_max_abs_error", "rdm_contraction_max_abs_error")):
        raise RuntimeError(f"RDM validation failed: {errors}")
    if errors["rdm_energy_abs_error_vs_rayleigh"] > energy_tolerance:
        raise RuntimeError(f"RDM energy validation failed: {errors}")
    errors["final_spin_square_diagnostic"] = result.spin_square()
    errors["rdm1_spin_traces"] = [float(np.trace(a)), float(np.trace(b))]
    return {key: float(value) if np.isscalar(value) else value for key, value in errors.items()}


def compare_cpu(result, cpu, operator_image, *, residual_tolerance, energy_tolerance,
                rdm_tolerance=1e-6):
    """逐元素比较独立算符像和每个自旋 RDM，不只比较总能量。"""
    coefficients = result.sci_state.amplitudes.reshape(-1)
    image = cpu["cpu_image"].reshape(-1)
    if (not np.isfinite(image).all() or not np.isfinite(operator_image).all()
            or not np.isfinite(cpu["physical_energy"])):
        raise RuntimeError("Independent PySCF validation received nonfinite data.")
    rayleigh = float(np.dot(coefficients, image))
    difference = image - np.asarray(operator_image).reshape(-1)
    cpu_rdms = rdms_from_cpu(cpu)
    errors = {
        "cpu_reference_uses_spectator": bool(cpu.get("operator_uses_spectator", False)),
        "pyscf_operator_rayleigh_energy_hartree": rayleigh,
        "pyscf_bare_hamiltonian_rayleigh_energy_hartree": float(cpu["bare_cpu_rayleigh"]),
        "pyscf_operator_residual_norm": float(np.linalg.norm(image - rayleigh * coefficients)),
        "operator_image_max_abs_error": float(np.max(abs(difference))),
        "operator_image_relative_l2_error": float(np.linalg.norm(difference) / max(np.linalg.norm(image), 1e-15)),
        "rdm_cpu_max_abs_error": max(float(np.max(abs(x - y))) for x, y in
                                     zip((*result.rdm1s, *result.rdm2s), (*cpu_rdms[0], *cpu_rdms[1]))),
        "cpu_energy_abs_error": abs(result.energy - float(cpu["physical_energy"])),
    }
    if (errors["pyscf_operator_residual_norm"] > residual_tolerance or
        (errors["operator_image_relative_l2_error"] > 1e-10 and errors["operator_image_max_abs_error"] > 1e-10) or
        errors["rdm_cpu_max_abs_error"] > rdm_tolerance or errors["cpu_energy_abs_error"] > energy_tolerance):
        raise RuntimeError(f"Independent PySCF validation failed: {errors}")
    return errors


def validate_spin_square(coefficients, result, *, contractor=None, tolerance=1e-6,
                         backend="torch", max_memory_fraction=.8):
    """将 RDM 自旋期望与设备上的独立 S²c 收缩比较，不要求目标自旋纯化。"""
    import torch
    from .torch_spin_square import TorchSelectedS2Contractor

    state = result.sci_state
    if contractor is None:
        contractor = TorchSelectedS2Contractor(
            (state.ci_strs_a, state.ci_strs_b), state.norb, state.nelec,
            device=coefficients.device, dtype=coefficients.dtype,
            backend=backend, max_memory_fraction=max_memory_fraction)
    matrix = coefficients.reshape(state.amplitudes.shape)
    spin_image = contractor.contract(matrix)
    expectation = float(torch.sum(matrix * spin_image))
    error = abs(expectation - result.spin_square())
    if not np.isfinite(expectation) or not np.isfinite(error) or error > tolerance:
        raise RuntimeError(f"RDM and device S^2 expectation disagree: error={error:.3e}")
    return {"operator_spin_square_expectation": expectation,
            "spin_square_rdm_abs_error": error,
            "spin_context_memory_bytes": (int(contractor.external_memory_bytes())
                                           if hasattr(contractor, "external_memory_bytes") else 0),
            "spin_validation": "RDM_vs_device_S2"}
