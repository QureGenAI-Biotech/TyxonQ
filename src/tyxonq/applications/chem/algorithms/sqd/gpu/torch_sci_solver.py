"""使用本方 Davidson 的 A100 双精度 selected-CI 求解入口。

支持实数 restricted 积分、可选 S² 自旋惩罚，以及 Torch/SBD Hamiltonian 后端；
RDM 可由 CPU 或 SBD 计算，结果按所选模式执行独立 CPU 复核或密度与残差检查。"""

from __future__ import annotations

from collections.abc import Callable
import gc
import json
import os
from time import perf_counter, sleep
from typing import Any

import numpy as np
from pyscf import lib
from pyscf.fci import selected_ci
import torch

from ..fermion import SCIResult, SCIState
from .validation import (
    assemble_result, compare_cpu, independent_cpu_validation, normalize_problem,
    project_initial_state, rdms_from_cpu, validate_rdms, validate_spin_square,
    validation_failure_snapshot,
)

from .torch_davidson import davidson_lowest
from .torch_selected_ci import (
    TorchSelectedCIContractor,
    absorb_h1e_torch,
    make_hdiag_torch,
)
from .torch_spin_square import (
    TorchSelectedS2Contractor,
    make_spin_square_diagonal_torch,
)


def davidson_space_for_memory(requested, vector_bytes, available_bytes, *, max_memory_fraction=0.8):
    """预算常驻 V/AV、厚重启副本和长向量临时量，保留收缩工作区。"""
    dimension = max(1, int(vector_bytes) // 8)
    # 单行列式会在 Davidson 内收缩为一维，但参数 max_space 仍至少为 2。
    requested = min(requested, max(2, dimension))
    budget = max_memory_fraction * available_bytes - 8 * 1024**3
    capacity = requested
    while capacity > 0:
        retained = min(max(2, capacity // 4), 8, max(0, capacity - 1))
        if (2 * capacity + 2 * retained + 12) * vector_bytes <= budget:
            break
        capacity -= 1
    if capacity < min(8, requested):
        raise RuntimeError(
            "可用显存不足以安全保存 Davidson 辅助空间；未截断 CI 空间。"
            f" available_bytes={available_bytes}, vector_bytes={vector_bytes},"
            f" capacity={capacity}, requested={requested}"
        )
    return min(requested, capacity)


def wait_for_davidson_memory(device, requested, vector_bytes, progress_callback,
                             *, timeout_seconds=600.0, poll_seconds=5.0, max_memory_fraction=0.8):
    """显存短时不足时清理未使用的缓存并重测；超时保留完整诊断。"""
    started = perf_counter()
    while True:
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        available = int(free + max(0, reserved - allocated))
        snapshot = {
            "free_bytes": int(free), "total_bytes": int(total),
            "allocated_bytes": int(allocated), "reserved_bytes": int(reserved),
            "available_bytes": available, "vector_bytes": int(vector_bytes),
            "requested_max_space": requested,
            "waited_seconds": perf_counter() - started,
        }
        try:
            space = davidson_space_for_memory(requested, vector_bytes, available,
                                              max_memory_fraction=max_memory_fraction)
        except RuntimeError as exc:
            if progress_callback is not None:
                progress_callback({"stage": "gpu_memory_wait", "status": "waiting", **snapshot})
            if snapshot["waited_seconds"] >= timeout_seconds:
                raise RuntimeError(
                    f"GPU 显存等待超时；{exc}；snapshot={json.dumps(snapshot)}"
                ) from exc
            # 只回收本进程不再使用的对象和缓存；不释放有效 CI 向量。
            gc.collect()
            torch.cuda.empty_cache()
            sleep(min(poll_seconds, max(0.0, timeout_seconds - snapshot["waited_seconds"])))
            continue
        if progress_callback is not None:
            progress_callback({"stage": "gpu_memory_budget", "status": "complete",
                               "effective_max_space": space, **snapshot})
        return space, available


def solve_sci_cuda(
    ci_strings: tuple[np.ndarray, np.ndarray],
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    threads: int = 2,
    chunk_size: int = 256,
    tolerance: float = 1e-8,
    max_cycles: int = 100,
    max_space: int = 20,
    operator_residual_tolerance: float = 1e-7,
    rdm_energy_tolerance: float = 1e-8,
    spin_sq: float | None = None,
    spin_shift: float = 0.1,
    device_text: str = "cuda:0",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    failure_debug_path: str | None = None,
    hamiltonian_backend: str = "torch",
    rdm_backend: str = "cpu",
    validation: str = "full_cpu",
    initial_state: SCIState | None = None,
    max_memory_fraction: float = 0.8,
    sbd_context=None,
    spin_backend: str = "torch",
) -> tuple[SCIResult, dict[str, Any]]:
    """使用本方 Davidson；显式选择 Hamiltonian、RDM 和独立验收后端。"""

    if hamiltonian_backend not in ("torch", "sbd") or rdm_backend not in ("cpu", "sbd"):
        raise ValueError("Unsupported hamiltonian_backend or rdm_backend.")
    if spin_backend not in ("torch", "cuda_fused"):
        raise ValueError("spin_backend must be 'torch' or 'cuda_fused'.")
    if validation not in ("full_cpu", "gpu"):
        raise ValueError("validation must be 'full_cpu' or 'gpu'.")
    if sbd_context is not None and hamiltonian_backend != "sbd" and rdm_backend != "sbd":
        raise ValueError("sbd_context requires an SBD Hamiltonian or RDM backend.")
    if not 0 < max_memory_fraction <= 0.8:
        raise ValueError("max_memory_fraction must be in (0, 0.8].")
    ci_strings, h1e, eri, norb, nelec = normalize_problem(ci_strings, h1e, eri, norb, nelec)
    projected_initial = project_initial_state(initial_state, ci_strings, norb, nelec)
    if min(threads, chunk_size, max_cycles, max_space) < 1:
        raise ValueError("A100 solver 的线程、chunk、空间或迭代参数必须为正数。")
    if min(tolerance, operator_residual_tolerance, rdm_energy_tolerance) <= 0:
        raise ValueError("A100 solver 的误差阈值必须为正数。")
    if spin_sq is not None:
        if not np.isfinite(spin_sq) or spin_sq < 0.0:
            raise ValueError("spin_sq 必须为有限非负数或 None。")
        if not np.isfinite(spin_shift) or spin_shift <= 0.0:
            raise ValueError("spin_shift 必须为有限正数。")

    device = torch.device(device_text)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("请求了 A100 solver，但 CUDA 当前不可用。")
    device_name = torch.cuda.get_device_name(device)
    if "A100" not in device_name:
        raise RuntimeError(f"当前 CUDA 设备不是 A100：{device_name}")

    lib.num_threads(threads)
    torch.set_num_threads(threads)
    torch_hamiltonian_uses_spectator = hamiltonian_backend == "torch" and min(nelec) == 0
    h_strings, h_one, h_two, h_norb, h_nelec = ci_strings, h1e, eri, norb, nelec
    if torch_hamiltonian_uses_spectator:
        from .cpu_validate_sci import add_zero_coupling_spectator

        h_strings, h_one, h_two, h_norb, h_nelec = add_zero_coupling_spectator(
            ci_strings, h1e, eri, norb, nelec)
    link_started = perf_counter()
    link_index = (selected_ci._all_linkstr_index(h_strings, h_norb, h_nelec)
                  if hamiltonian_backend == "torch" else None)
    link_seconds = perf_counter() - link_started

    torch.cuda.reset_peak_memory_stats(device)
    initial_free, total_memory = torch.cuda.mem_get_info(device)
    setup_started = perf_counter()
    sbd_contractor = None
    if hamiltonian_backend == "sbd" or rdm_backend == "sbd":
        from .sbd_solver import SBDContractor

        if sbd_context is None:
            sbd_contractor = SBDContractor(h1e, eri, ci_strings, norb, nelec, device=device,
                                           memory_fraction=max_memory_fraction)
        else:
            if not isinstance(sbd_context, SBDContractor):
                raise TypeError("sbd_context must be an SBDContractor.")
            sbd_context.assert_compatible(ci_strings, norb, nelec, device,
                                          memory_fraction=max_memory_fraction)
            sbd_context.update_integrals(h1e, eri)
            sbd_contractor = sbd_context
    if hamiltonian_backend == "sbd":
        contractor = sbd_contractor
        hdiag_device = contractor.diagonal()
    else:
        h1e_device = torch.as_tensor(h1e, dtype=torch.float64, device=device)
        eri_device = torch.as_tensor(eri, dtype=torch.float64, device=device)
        hdiag_device = make_hdiag_torch(h1e_device, eri_device, ci_strings)
        # 仅 Hc 使用辅助电子；对角元、自旋、RDM 与返回状态均属于原始空间。
        if torch_hamiltonian_uses_spectator:
            h1e_device = torch.as_tensor(h_one, dtype=torch.float64, device=device)
            eri_device = torch.as_tensor(h_two, dtype=torch.float64, device=device)
        h2e_device = absorb_h1e_torch(h1e_device, eri_device, h_nelec)
        contractor = TorchSelectedCIContractor(h2e_device, link_index, h_nelec, chunk_size=chunk_size)
    spin_contractor = (
        None
        if spin_sq is None
        else TorchSelectedS2Contractor(
            ci_strings,
            norb,
            nelec,
            device=device,
            dtype=torch.float64,
            backend=spin_backend,
            max_memory_fraction=max_memory_fraction,
        )
    )
    davidson_diagonal = hdiag_device
    if spin_contractor is not None:
        minimum_spin = 0.5 * abs(nelec[0] - nelec[1])
        minimum_spin_square = minimum_spin * (minimum_spin + 1.0)
        if float(spin_sq) < minimum_spin_square + 0.1:
            spin_diagonal = make_spin_square_diagonal_torch(
                ci_strings,
                norb,
                nelec,
                device=device,
                dtype=torch.float64,
            )
            davidson_diagonal = hdiag_device + spin_shift * (
                spin_diagonal - float(spin_sq)
            )
    torch.cuda.synchronize(device)
    setup_seconds = perf_counter() - setup_started
    # 辅助迭代空间可随可用显存缩小；最终残差标准、完整 CI 池保持原值。
    # 已保留但未分配的缓存可由分配器复用，不能只看驱动层 free。
    effective_max_space, available_bytes = wait_for_davidson_memory(
        device, max_space, hdiag_device.numel() * hdiag_device.element_size(),
        progress_callback, max_memory_fraction=max_memory_fraction,
    )
    if progress_callback is not None:
        progress_callback(
            {"stage": "gpu_setup", "status": "complete", "seconds": setup_seconds,
             "requested_max_space": max_space, "effective_max_space": effective_max_space}
        )

    def gpu_matvec(vector: torch.Tensor) -> torch.Tensor:
        matrix = vector.reshape(hdiag_device.shape)
        image = contractor.contract(matrix)
        if spin_contractor is not None:
            spin_residual = spin_contractor.contract(matrix) - float(spin_sq) * matrix
            minimum_spin = 0.5 * abs(nelec[0] - nelec[1])
            minimum_spin_square = minimum_spin * (minimum_spin + 1.0)
            if float(spin_sq) < minimum_spin_square + 0.1:
                image = image + spin_shift * spin_residual
            else:
                image = image + spin_shift * (
                    spin_contractor.contract(spin_residual)
                    - float(spin_sq) * spin_residual
                )
        return image.reshape(-1)

    def report_cycle(record: dict[str, float | int]) -> None:
        if progress_callback is not None:
            progress_callback(
                {"stage": "gpu_davidson", "status": "running", **record}
            )

    solver_started = perf_counter()
    gpu_result = davidson_lowest(
        gpu_matvec,
        davidson_diagonal.reshape(-1),
        tolerance=tolerance,
        max_cycles=max_cycles,
        max_space=effective_max_space,
        initial_vector=(None if projected_initial is None else torch.as_tensor(
            projected_initial.reshape(-1), dtype=torch.float64, device=device)),
        callback=report_cycle,
    )
    torch.cuda.synchronize(device)
    solver_seconds = perf_counter() - solver_started
    peak_device_memory = int(torch.cuda.max_memory_allocated(device))
    with validation_failure_snapshot(
            failure_debug_path, ci_strings, h1e, eri, norb, nelec,
            spin_sq=spin_sq, spin_shift=spin_shift, validation=validation,
            hamiltonian_backend=hamiltonian_backend, rdm_backend=rdm_backend,
            operator_residual_tolerance=operator_residual_tolerance,
            rdm_energy_tolerance=rdm_energy_tolerance) as debug:
        debug["coefficients"] = gpu_result.coefficients
        if not gpu_result.converged:
            raise RuntimeError(
                "A100 Davidson 未收敛："
                f"cycles={gpu_result.iterations}, residual={gpu_result.residual_norm:.3e}。"
            )

        # Davidson 内部缓存 H@basis；返回前重新做一次独立 GPU H@c，防止缓存的
        # Ritz image 与最终归一化向量不一致却产生假收敛。
        fresh_coefficients = gpu_result.coefficients / torch.linalg.vector_norm(
            gpu_result.coefficients
        )
        debug["coefficients"] = fresh_coefficients
        fresh_gpu_image = gpu_matvec(fresh_coefficients)
        debug["gpu_image"] = fresh_gpu_image
        fresh_gpu_rayleigh = torch.dot(fresh_coefficients, fresh_gpu_image)
        fresh_gpu_residual = float(
            torch.linalg.vector_norm(
                fresh_gpu_image - fresh_gpu_rayleigh * fresh_coefficients
            ).item()
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "gpu_fresh_residual",
                    "status": "complete",
                    "cached_residual_norm": float(gpu_result.residual_norm),
                    "fresh_residual_norm": fresh_gpu_residual,
                }
            )

        if not np.isfinite(fresh_gpu_residual) or fresh_gpu_residual > operator_residual_tolerance:
            raise RuntimeError(
                "A100 Davidson 缓存残差出现假收敛："
                f"cached={gpu_result.residual_norm:.3e}, fresh={fresh_gpu_residual:.3e}。"
            )

        coefficients = fresh_coefficients.detach().cpu().numpy().astype(np.float64, copy=False)
        bare_image = (fresh_gpu_image if spin_contractor is None else
                      contractor.contract(fresh_coefficients.reshape(hdiag_device.shape)).reshape(-1))
        debug["gpu_bare_image"] = bare_image
        bare_rayleigh = float(torch.dot(fresh_coefficients, bare_image))
        fresh_gpu_image_cpu = fresh_gpu_image.detach().cpu().numpy()
        cpu = None
        validation_started = perf_counter()
        if validation == "full_cpu" or rdm_backend == "cpu":
            if progress_callback:
                progress_callback({"stage": "cpu_validation_rdm", "status": "running"})
            cpu = independent_cpu_validation(
                coefficients, ci_strings, h1e, eri, norb, nelec,
                spin_sq=spin_sq, spin_shift=spin_shift, threads=threads,
                include_operator=validation == "full_cpu")
            if "cpu_image" in cpu:
                debug["cpu_image"] = cpu["cpu_image"]
        rdm_started = perf_counter()
        rdms = (sbd_contractor.rdms(fresh_coefficients) if rdm_backend == "sbd"
                else rdms_from_cpu(cpu))
        rdm_seconds = (perf_counter() - rdm_started if rdm_backend == "sbd"
                       else float(cpu["rdm_seconds"]))
        host_check_started = perf_counter()
        physical = assemble_result(coefficients, ci_strings, h1e, eri, norb, nelec, *rdms)
        report = validate_rdms(physical, bare_rayleigh, energy_tolerance=rdm_energy_tolerance)
        if validation == "full_cpu":
            report.update(compare_cpu(
                physical, cpu, fresh_gpu_image_cpu,
                residual_tolerance=operator_residual_tolerance,
                energy_tolerance=rdm_energy_tolerance))
        host_check_seconds = perf_counter() - host_check_started
        if validation == "gpu":
            report.update(validate_spin_square(fresh_coefficients, physical, contractor=spin_contractor,
                                               backend=spin_backend, max_memory_fraction=max_memory_fraction))
        cpu_subprocess_seconds = 0.0 if cpu is None else float(cpu["seconds"])
        cpu_validation_seconds = 0.0 if cpu is None else cpu_subprocess_seconds + host_check_seconds
        if progress_callback and cpu is not None:
            progress_callback({"stage": "cpu_validation_rdm", "status": "complete",
                               "seconds": cpu_validation_seconds,
                               "include_operator": validation == "full_cpu"})
        validation_seconds = perf_counter() - validation_started
        if progress_callback:
            progress_callback({"stage": "solution_validation", "status": "complete",
                               "validation": validation, "seconds": validation_seconds})
    report.update({
        "solver_backend": ("a100_fp64_final_davidson_cpu_validation" if
                           hamiltonian_backend == "torch" and validation == "full_cpu" else
                           "sbd_hybrid" if hamiltonian_backend == "sbd" else "torch_cuda"),
        "hamiltonian_backend": hamiltonian_backend,
        "hamiltonian_uses_zero_coupling_spectator": torch_hamiltonian_uses_spectator,
        "rdm_backend": rdm_backend,
        "validation": validation,
        "independent_cpu_validation": validation == "full_cpu",
        "gpu_acceleration_scope": "hdiag,H@c,S2_constraint,final_Davidson" + (",RDM" if rdm_backend == "sbd" else ""),
        "cpu_scope": ("independent_operator_validation," if validation == "full_cpu" else "") +
                     ("RDM," if rdm_backend == "cpu" else "") + "link_tables,RDM_invariants,C-PCM",
        "spin_constraint_applied": spin_sq is not None,
        "spin_backend": spin_backend,
        "spin_backend_active": spin_sq is not None or validation == "gpu",
        "spin_context_memory_bytes": (int(spin_contractor.external_memory_bytes()) if spin_contractor is not None
                                      else report.get("spin_context_memory_bytes", 0)),
        "target_spin_square": spin_sq,
        "spin_shift": spin_shift if spin_sq is not None else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host_gpu_id": os.environ.get("TYXONQ_A100_HOST_GPU_ID"),
        "device": str(device), "device_name": device_name,
        "dtype": "float64", "threads": int(threads),
        "n_alpha_strings": len(ci_strings[0]), "n_beta_strings": len(ci_strings[1]),
        "selected_ci_dimension": int(coefficients.size),
        "link_setup_seconds_cpu": link_seconds,
        "gpu_setup_seconds": setup_seconds,
        "gpu_solver_seconds": solver_seconds,
        "gpu_iterations": int(gpu_result.iterations),
        "requested_max_space": max_space,
        "effective_max_space": effective_max_space,
        "available_device_bytes_at_setup": available_bytes,
        "gpu_h_matvecs": int(gpu_result.h_matvecs),
        "gpu_residual_norm_device": float(gpu_result.residual_norm),
        "gpu_fresh_residual_norm_device": fresh_gpu_residual,
        "gpu_penalized_eigenvalue_hartree": float(fresh_gpu_rayleigh),
        "physical_rayleigh_energy_hartree": bare_rayleigh,
        "spin_penalty_expectation_hartree": float(fresh_gpu_rayleigh) - bare_rayleigh,
        "cpu_validation_seconds": cpu_validation_seconds,
        "cpu_subprocess_seconds": cpu_subprocess_seconds,
        "validation_host_check_seconds": host_check_seconds,
        "cpu_validation_timing_scope": "CPU subprocess plus host validation; excludes GPU RDM and device S2",
        "cpu_validation_isolated_external_process": cpu is not None,
        "rdm_seconds": rdm_seconds,
        "peak_device_memory_bytes": max(peak_device_memory, int(torch.cuda.max_memory_allocated(device))),
        "peak_device_memory_scope": "PyTorch allocator only; excludes SBD C++ allocations",
        "sbd_context_memory_bytes": (0 if sbd_contractor is None else int(sbd_contractor.context.memory_bytes())),
        "device_used_increase_bytes_at_return": max(0, initial_free - torch.cuda.mem_get_info(device)[0]),
        "device_memory_total_bytes": int(total_memory),
        "max_memory_fraction": max_memory_fraction,
        "initial_state_projected": projected_initial is not None,
        "sbd_context_reused": sbd_context is not None,
        "gpu_history": list(gpu_result.history),
    })
    return physical, report
