"""可选 SBD TPB 后端；导入此模块不会加载 CUDA、MPI 或编译扩展。"""

from __future__ import annotations

import os
from time import perf_counter

import numpy as np

from tyxonq.numerics.sbd.cuda import (
    SBDContractor as _CUDAContractor,
    convert_raw_rdms,
    interleaved_phase,
    load_extension as _load_extension,  # 兼容按旧路径导入的基准脚本。
    solve_selected_basis,
)
from tyxonq.numerics.sbd.metal import metal_gpu_execution
from .validation import (
    assemble_result, compare_cpu, independent_cpu_validation, normalize_problem,
    project_initial_state, rdms_from_cpu, validate_rdms, validate_spin_square,
    validation_failure_snapshot,
)


class SBDContractor(_CUDAContractor):
    """在通用 CUDA SBD 算符外检查 SQD 的积分对称性和已排序组态。"""

    def __init__(self, h1e, eri, ci_strings, norb, nelec, *, device, memory_fraction=0.8):
        normalized, h1e, eri, norb, nelec = normalize_problem(ci_strings, h1e, eri, norb, nelec)
        if any(not np.array_equal(original, clean) for original, clean in zip(ci_strings, normalized)):
            raise ValueError("SBDContractor requires sorted unique determinant labels.")
        super().__init__(h1e, eri, normalized, norb, nelec, device=device,
                         memory_fraction=memory_fraction)

    def update_integrals(self, h1e, eri):
        """先检查新积分，再刷新原生对角元和连接权重。"""
        _, h1e, eri, _, _ = normalize_problem(self.ci_strings, h1e, eri, self.norb, self.nelec)
        self.context.update_integrals(h1e, eri)


def solve_sci_sbd(ci_strings, h1e, eri, norb, nelec, *, threads=2, chunk_size=256,
                  tolerance=1e-8, max_cycles=100, max_space=20,
                  operator_residual_tolerance=1e-7, rdm_energy_tolerance=1e-8,
                  spin_sq=None, spin_shift=0.1, device_text="cuda:0",
                  validation="full_cpu", rdm_backend="sbd", initial_state=None,
                  progress_callback=None, failure_debug_path=None, max_memory_fraction=0.8,
                  sbd_context=None, spin_backend="torch"):
    """单卡原生 TPB 求解；MPI 多卡仅由独立验收入口直接调用扩展。"""
    if spin_sq is not None:
        raise ValueError("Native SBD does not implement spin_sq; use the hybrid backend.")
    if validation not in ("full_cpu", "gpu") or rdm_backend not in ("cpu", "sbd"):
        raise ValueError("Unsupported validation or rdm_backend.")
    if spin_backend not in ("torch", "cuda_fused"):
        raise ValueError("spin_backend must be 'torch' or 'cuda_fused'.")
    if min(threads, max_cycles, max_space) < 1 or min(tolerance, operator_residual_tolerance, rdm_energy_tolerance) <= 0:
        raise ValueError("Solver limits and tolerances must be positive.")
    if not 0 < max_memory_fraction <= 0.8:
        raise ValueError("max_memory_fraction must be in (0, 0.8].")
    ci_strings, h1e, eri, norb, nelec = normalize_problem(ci_strings, h1e, eri, norb, nelec)
    initial = project_initial_state(initial_state, ci_strings, norb, nelec)
    import torch

    device = torch.device(device_text)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SBD requires an available CUDA device.")
    # is_available 只探测驱动；新进程必须先初始化并选择实际 CUDA 设备。
    device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    torch.cuda.set_device(device)
    torch.set_num_threads(threads)
    torch.cuda.reset_peak_memory_stats(device)
    initial_free, total_memory = torch.cuda.mem_get_info(device)
    started = perf_counter()
    if sbd_context is None:
        contractor = SBDContractor(h1e, eri, ci_strings, norb, nelec, device=device,
                                   memory_fraction=max_memory_fraction)
    else:
        if not isinstance(sbd_context, SBDContractor):
            raise TypeError("sbd_context must be an SBDContractor.")
        sbd_context.assert_compatible(ci_strings, norb, nelec, device, memory_fraction=max_memory_fraction)
        sbd_context.update_integrals(h1e, eri)
        contractor = sbd_context
    setup_seconds = perf_counter() - started
    from .solver import wait_for_davidson_memory

    effective_space, available_bytes = wait_for_davidson_memory(
        device, max_space, int(np.prod(contractor.shape)) * 8, progress_callback,
        max_memory_fraction=max_memory_fraction)
    if initial is None:
        diagonal = contractor.diagonal()
        initial_tensor = torch.zeros_like(diagonal).reshape(-1)
        initial_tensor[torch.argmin(diagonal)] = 1.0
        initial_tensor = initial_tensor.reshape(contractor.shape)
    else:
        initial_tensor = torch.as_tensor(initial, dtype=torch.float64, device=device)
    raw_initial = contractor._raw(initial_tensor).cpu().numpy()
    if progress_callback:
        progress_callback({"stage": "sbd_native", "status": "running"})
    with validation_failure_snapshot(
            failure_debug_path, ci_strings, h1e, eri, norb, nelec,
            spin_sq=None, spin_shift=spin_shift, validation=validation,
            hamiltonian_backend="sbd", rdm_backend=rdm_backend,
            operator_residual_tolerance=operator_residual_tolerance,
            rdm_energy_tolerance=rdm_energy_tolerance) as debug:
        # C++ 内同时完成求解/RDM；内部异常时最终态尚未返回，只能保存初猜。
        debug["coefficients"] = initial if initial is not None else np.abs(raw_initial)
        debug["coefficients_role"] = "initial_guess"
        raw, native_total_seconds = solve_selected_basis(
            contractor, raw_initial, tolerance=tolerance, max_cycles=max_cycles,
            max_space=effective_space, compute_rdms=rdm_backend == "sbd")
        native_timings = raw["timings"]
        solver_seconds = float(native_timings["davidson_seconds"])
        coefficients = torch.as_tensor(np.asarray(raw["coefficients"]).reshape(contractor.shape),
                                       dtype=torch.float64, device=device) * contractor.phase
        debug["coefficients_role"] = "solver_result"
        debug["coefficients"] = coefficients
        norm = torch.linalg.vector_norm(coefficients)
        if not torch.isfinite(norm) or abs(float(norm) - 1.0) > 1e-8:
            raise RuntimeError("SBD returned an invalid normalized wavefunction.")
        coefficients = coefficients / norm
        debug["coefficients"] = coefficients
        if "hamiltonian_image" in raw:
            # C++ 已在 Davidson 之外重新计算一次 Hc；相位和归一化需同时作用于 c/Hc。
            image = torch.as_tensor(np.asarray(raw["hamiltonian_image"]).reshape(contractor.shape),
                                    dtype=torch.float64, device=device) * contractor.phase / norm
        else:
            image = contractor.contract(coefficients)
        debug["gpu_image"] = image
        rayleigh = float(torch.sum(coefficients * image))
        residual = float(torch.linalg.vector_norm(image - rayleigh * coefficients))
        if not raw.get("converged", True) or not np.isfinite(residual) or residual > operator_residual_tolerance:
            raise RuntimeError(f"SBD Davidson did not converge: residual={residual:.3e}, iterations={raw.get('iterations')}")
        coeff_cpu = coefficients.cpu().numpy()
        debug["coefficients"] = coeff_cpu
        cpu = None
        if validation == "full_cpu" or rdm_backend == "cpu":
            cpu = independent_cpu_validation(coeff_cpu, ci_strings, h1e, eri, norb, nelec,
                                            spin_sq=None, spin_shift=spin_shift, threads=threads,
                                            include_operator=validation == "full_cpu")
            if "cpu_image" in cpu:
                debug["cpu_image"] = cpu["cpu_image"]
        # 原生求解已生成一次 GPU RDM；这里仅变换约定，不再次调用 RDM 内核。
        rdms = convert_raw_rdms(raw, norb) if rdm_backend == "sbd" else rdms_from_cpu(cpu)
        rdm_seconds = float(native_timings["rdm_seconds"]) if rdm_backend == "sbd" else float(cpu["rdm_seconds"])
        host_check_started = perf_counter()
        physical = assemble_result(coeff_cpu, ci_strings, h1e, eri, norb, nelec, *rdms)
        report = validate_rdms(physical, rayleigh, energy_tolerance=rdm_energy_tolerance)
        if validation == "full_cpu":
            report.update(compare_cpu(physical, cpu, image.cpu().numpy(),
                                      residual_tolerance=operator_residual_tolerance,
                                      energy_tolerance=rdm_energy_tolerance))
        host_check_seconds = perf_counter() - host_check_started
        if validation == "gpu":
            report.update(validate_spin_square(coefficients, physical, backend=spin_backend,
                                               max_memory_fraction=max_memory_fraction))
        cpu_subprocess_seconds = 0.0 if cpu is None else float(cpu["seconds"])
        cpu_validation_seconds = 0.0 if cpu is None else cpu_subprocess_seconds + host_check_seconds
    report.update({
        "solver_backend": "sbd_native", "hamiltonian_backend": "sbd", "rdm_backend": rdm_backend,
        "validation": validation, "independent_cpu_validation": validation == "full_cpu",
        "gpu_setup_seconds": setup_seconds, "gpu_solver_seconds": solver_seconds,
        "native_total_seconds": native_total_seconds,
        "native_fresh_image_reused": "hamiltonian_image" in raw,
        "solver_timing_scope": "Davidson", "native_timings": dict(native_timings),
        "rdm_seconds": rdm_seconds, "cpu_validation_seconds": cpu_validation_seconds,
        "cpu_subprocess_seconds": cpu_subprocess_seconds,
        "validation_host_check_seconds": host_check_seconds,
        "cpu_validation_timing_scope": "CPU subprocess plus host validation; excludes GPU RDM and device S2",
        "gpu_iterations": int(raw.get("iterations", -1)), "gpu_h_matvecs": int(raw.get("h_matvecs", -1)),
        "gpu_fresh_residual_norm_device": residual, "gpu_penalized_eigenvalue_hartree": rayleigh,
        "spin_penalty_expectation_hartree": 0.0, "spin_constraint_applied": False,
        "spin_backend": spin_backend, "spin_backend_active": validation == "gpu",
        "selected_ci_dimension": int(coefficients.numel()), "n_alpha_strings": len(ci_strings[0]),
        "n_beta_strings": len(ci_strings[1]), "device": str(device), "device_name": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "dtype": "float64", "threads": threads,
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_device_memory_scope": "PyTorch allocator only; excludes SBD C++ allocations",
        "sbd_context_memory_bytes": int(contractor.context.memory_bytes()),
        "sbd_hc_lanes": int(getattr(contractor.context, "hc_lanes", 0)),
        "device_used_increase_bytes_at_return": max(0, initial_free - torch.cuda.mem_get_info(device)[0]),
        "device_memory_total_bytes": int(total_memory),
        "available_device_bytes_at_setup": int(available_bytes),
        "max_memory_fraction": max_memory_fraction,
        "requested_max_space": max_space, "effective_max_space": effective_space,
        "initial_state_projected": initial is not None,
        "sbd_context_reused": sbd_context is not None,
        "gpu_acceleration_scope": "SBD hdiag,H@c,Davidson" + (",RDM" if rdm_backend == "sbd" else ""),
    })
    if progress_callback:
        progress_callback({"stage": "sbd_native", "status": "complete", "residual_norm": residual})
    return physical, report


def _determinant_energy_offset(h1e, eri, ci_strings):
    """用第一个组态的 FP64 Slater–Condon 对角能选择数值零点。"""
    n = len(h1e)
    occ = [np.array([(int(labels[0]) >> i) & 1 for i in range(n)], dtype=float)
           for labels in ci_strings]
    charge = occ[0] + occ[1]
    coulomb = np.einsum("iijj->ij", eri)
    exchange = np.einsum("ijji->ij", eri)
    return float(charge @ np.diag(h1e) + 0.5 * (charge @ coulomb @ charge
                 - sum(s @ exchange @ s for s in occ)))


@metal_gpu_execution
def solve_sci_metal(
    ci_strings, h1e, eri, norb, nelec, *, threads=2, chunk_size=64,
    tolerance=1e-5, max_cycles=100, max_space=20,
    operator_residual_tolerance=5e-5, rdm_energy_tolerance=5e-5,
    spin_sq=None, spin_shift=0.1, device_text="metal:0",
    validation="full_cpu", rdm_backend="cpu", initial_state=None,
    progress_callback=None, failure_debug_path=None, max_memory_fraction=0.8,
    sbd_context=None, spin_backend="metal", spin_tolerance=1e-4,
):
    """用原生 Metal 求解 restricted selected-CI，可选在 GPU 上计算异自旋 2-RDM。

    rdm_backend='cpu' 为默认设置。'metal' 模式的 1-RDM 和同自旋块在 CPU
    以 FP64 计算。full_cpu 另外逐元素核验 Metal 的异自旋块及 Hamiltonian 算符像。
    """
    if device_text not in ("metal", "metal:0", "gpu"):
        raise ValueError("Metal SBD requires device_text='metal:0'.")
    if validation not in ("full_cpu", "gpu") or rdm_backend not in ("cpu", "metal"):
        raise ValueError("Metal SBD requires validation='full_cpu' or 'gpu' and rdm_backend='cpu' or 'metal'.")
    if spin_backend != "metal":
        raise ValueError("Metal SBD uses spin_backend='metal'.")
    if sbd_context is not None:
        raise ValueError("A CUDA SBD context cannot be used by Metal SBD.")
    if min(threads, chunk_size, max_cycles, max_space) < 1:
        raise ValueError("Metal solver limits must be positive.")
    limits = (tolerance, operator_residual_tolerance, rdm_energy_tolerance)
    if any(not np.isfinite(value) or value <= 0 for value in limits):
        raise ValueError("Metal solver tolerances must be finite and positive.")
    ci_strings, h1e, eri, norb, nelec = normalize_problem(ci_strings, h1e, eri, norb, nelec)
    nelec = (int(nelec[0]), int(nelec[1]))
    if rdm_backend == "metal":
        from tyxonq.numerics.sbd.metal.rdm import check_mixed_rdm_size, mixed_rdm_q60
        check_mixed_rdm_size(tuple(map(len, ci_strings)))
    initial = project_initial_state(initial_state, ci_strings, norb, nelec)
    if spin_sq is not None and (not np.isfinite(spin_sq) or spin_sq < 0):
        raise ValueError("spin_sq must be finite and nonnegative.")
    if spin_sq is not None and (not np.isfinite(spin_shift) or spin_shift <= 0):
        raise ValueError("spin_shift must be finite and positive.")

    if spin_sq is not None and (not np.isfinite(spin_tolerance) or spin_tolerance <= 0):
        raise ValueError("spin_tolerance must be finite and positive.")

    import mlx.core as mx
    from pyscf.fci import selected_ci
    from tyxonq.numerics.sbd.metal import (
        MetalSelectedCIContractor, MetalSelectedS2Contractor, absorb_h1e,
        davidson_lowest, make_hdiag, make_spin_square_diagonal,
    )

    started = perf_counter()
    # 固定电子数 N 时，给一个电子的对角积分减去 C/N 使整个 H 精确平移 −C I。
    # 要在吸收一体项和转换 FP32 前进行，避免缩小残差时丢掉共有的大能量常数。
    energy_offset = _determinant_energy_offset(h1e, eri, ci_strings)
    shifted_h1 = h1e.copy()
    shifted_h1.flat[::norb + 1] -= energy_offset / sum(nelec)
    h_strings, h_one, h_two, h_norb, h_nelec = ci_strings, shifted_h1, eri, norb, nelec
    uses_spectator = min(nelec) == 0
    if uses_spectator:
        from .cpu_validate_sci import add_zero_coupling_spectator
        h_strings, h_one, h_two, h_norb, h_nelec = add_zero_coupling_spectator(
            ci_strings, shifted_h1, eri, norb, nelec)
    h_nelec = (int(h_nelec[0]), int(h_nelec[1]))
    link_started = perf_counter()
    links = selected_ci._all_linkstr_index(h_strings, h_norb, h_nelec)
    link_seconds = perf_counter() - link_started
    h1_mx = mx.array(np.asarray(shifted_h1, dtype=np.float32))
    eri_mx = mx.array(np.asarray(eri, dtype=np.float32))
    hdiag = make_hdiag(h1_mx, eri_mx, ci_strings)
    contractor = MetalSelectedCIContractor(
        absorb_h1e(h_one, h_two, h_nelec), links, h_nelec, chunk_size=chunk_size)
    spin_contractor = MetalSelectedS2Contractor(ci_strings, norb, nelec)
    diagonal = hdiag
    if spin_sq is not None:
        minimum_spin = 0.5 * abs(nelec[0] - nelec[1])
        minimum_spin_square = minimum_spin * (minimum_spin + 1.0)
        if float(spin_sq) < minimum_spin_square + 0.1:
            diagonal = diagonal + spin_shift * (
                make_spin_square_diagonal(ci_strings, norb, nelec) - float(spin_sq))
    mx.eval(diagonal)
    setup_seconds = perf_counter() - started

    def scalar(value):
        mx.eval(value)
        return float(value.item())

    def matvec(vector):
        matrix = vector.reshape(hdiag.shape)
        image = contractor.contract(matrix)
        if spin_sq is not None:
            spin_residual = spin_contractor.contract(matrix) - float(spin_sq) * matrix
            minimum_spin = 0.5 * abs(nelec[0] - nelec[1])
            minimum_spin_square = minimum_spin * (minimum_spin + 1.0)
            if float(spin_sq) < minimum_spin_square + 0.1:
                image = image + spin_shift * spin_residual
            else:
                image = image + spin_shift * (
                    spin_contractor.contract(spin_residual) - float(spin_sq) * spin_residual)
        return image.reshape(-1)

    def report_cycle(record):
        if progress_callback is not None:
            progress_callback({"stage": "metal_davidson", "status": "running", **record,
                               "energy": record["energy"] + energy_offset})

    solver_started = perf_counter()
    initial_mx = None if initial is None else mx.array(initial.reshape(-1).astype(np.float32))
    def solve(operator, guess):
        return davidson_lowest(
            operator, diagonal.reshape(-1), initial_vector=guess, tolerance=tolerance,
            max_cycles=max_cycles, max_space=max_space, callback=report_cycle)

    target_report = {}
    if spin_sq is None:
        solved = solve(matvec, initial_mx)
    else:
        from .spin_target import solve_target_spin

        solved, target_report = solve_target_spin(
            matvec, lambda v: contractor.contract(v.reshape(hdiag.shape)).reshape(-1),
            lambda v: spin_contractor.contract(v.reshape(hdiag.shape)).reshape(-1),
            diagonal, ci_strings, norb, nelec, spin_sq, spin_tolerance, initial_mx,
            solve, rdm_energy_tolerance)
    def fresh_candidate(solution):
        if not solution.converged:
            raise RuntimeError(
                f"Metal Davidson did not converge: cycles={solution.iterations}, "
                f"residual={solution.residual_norm:.3e}")
        c = solution.coefficients / mx.linalg.norm(solution.coefficients)
        image = matvec(c)
        return c, image, scalar(mx.sum(c * image) / mx.sum(c**2))

    coefficients_mx, fresh_image, shifted_rayleigh = fresh_candidate(solved)
    if spin_sq is None:
        # 任一单行列式的期望都是无自旋约束最低能量的变分上界。重算所选对角元的
        # FP64 值，可避免设备 FP32 对角元的舍入误差被当成判据。通过此检查仍可能是激发态。
        index = int(mx.argmin(hdiag.reshape(-1)).item())
        i, j = divmod(index, len(ci_strings[1]))
        trial = ci_strings[0][i:i+1], ci_strings[1][j:j+1]
        bound = _determinant_energy_offset(shifted_h1, eri, trial)
        slack = max(tolerance, 16 * np.finfo(np.float32).eps * max(1., abs(bound)))
        target_report = dict(
            determinant_upper_bound_hartree=bound + energy_offset,
            determinant_bound_labels=[int(trial[0][0]), int(trial[1][0])],
            determinant_bound_tolerance_hartree=float(slack),
            initial_guess_kind="projected_explicit" if initial is not None else "minimum_diagonal",
            first_candidate_rayleigh_hartree=shifted_rayleigh + energy_offset,
            first_davidson_h_matvecs=int(solved.h_matvecs), first_davidson_iterations=int(solved.iterations),
            determinant_bound_retries=0, retry_seconds=0., retry_davidson_h_matvecs=0,
            retry_davidson_iterations=0)
        if shifted_rayleigh > bound + slack and initial is not None:
            # 残差即使为零，显式初猜也可能卡在断开的高能块。只对这种可证伪的候选额外求解一次。
            start = perf_counter()
            ia, ib = np.unravel_index(np.argmax(np.abs(initial)), initial.shape)
            target_report["initial_guess_dominant_labels"] = [int(ci_strings[0][ia]), int(ci_strings[1][ib])]
            target_report["initial_guess_dominant_weight"] = float(initial[ia, ib]**2)
            retry_guess = mx.zeros_like(hdiag.reshape(-1)).at[index].add(1.)
            solved = solve(matvec, retry_guess)
            coefficients_mx, fresh_image, shifted_rayleigh = fresh_candidate(solved)
            target_report.update(determinant_bound_retries=1, retry_seconds=perf_counter() - start,
                                 retry_davidson_h_matvecs=int(solved.h_matvecs),
                                 retry_davidson_iterations=int(solved.iterations))
        if shifted_rayleigh > bound + slack:
            raise RuntimeError(
                f"Metal ground-state Rayleigh energy exceeds determinant variational upper bound: "
                f"{shifted_rayleigh + energy_offset:.12g} > {bound + energy_offset:.12g} Eh, "
                f"tolerance={slack:.3e}, retries={target_report['determinant_bound_retries']}")
    residual = scalar(mx.linalg.norm(fresh_image - shifted_rayleigh * coefficients_mx))
    if not np.isfinite(residual) or residual > operator_residual_tolerance:
        raise RuntimeError(f"Metal final residual is too large: {residual:.3e}")
    solver_seconds = perf_counter() - solver_started
    bare_image = (fresh_image if spin_sq is None else
                  contractor.contract(coefficients_mx.reshape(hdiag.shape)).reshape(-1))
    # 恢复原始物理 Hc 及 Rayleigh 能量时，系数与算符像要使用同一归一化因子。
    coefficients = np.asarray(coefficients_mx, dtype=np.float64)
    norm = np.linalg.norm(coefficients)
    coefficients /= norm
    fresh_image_cpu = np.asarray(fresh_image, dtype=np.float64) / norm
    rayleigh = float(coefficients @ fresh_image_cpu) + energy_offset
    bare_rayleigh = (rayleigh if spin_sq is None else
                     float(coefficients @ np.asarray(bare_image, dtype=np.float64)) / norm + energy_offset)
    fresh_image_cpu += energy_offset * coefficients

    with validation_failure_snapshot(
            failure_debug_path, ci_strings, h1e, eri, norb, nelec,
            spin_sq=spin_sq, spin_shift=spin_shift, validation=validation,
            hamiltonian_backend="metal", rdm_backend=rdm_backend,
            operator_residual_tolerance=operator_residual_tolerance,
            rdm_energy_tolerance=rdm_energy_tolerance) as debug:
        debug["coefficients"] = coefficients
        debug["gpu_image"] = fresh_image_cpu
        validation_started = perf_counter()
        cpu = None
        if validation == "full_cpu" or rdm_backend == "cpu":
            cpu = independent_cpu_validation(
                coefficients, ci_strings, h1e, eri, norb, nelec,
                spin_sq=spin_sq, spin_shift=spin_shift, threads=threads,
                include_operator=validation == "full_cpu")
            debug["cpu_image"] = cpu["cpu_image"]

        rdm_metrics = {}
        if rdm_backend == "cpu":
            rdms = rdms_from_cpu(cpu)
            rdm_seconds = float(cpu["rdm_seconds"])
            rdm_algorithm = str(cpu["rdm_algorithm"])
        else:
            mixed_start = perf_counter()
            tables = tuple(selected_ci.cre_des_linkstr(labels, norb, count)
                           for labels, count in zip(ci_strings, nelec))
            c = coefficients.reshape(tuple(map(len, ci_strings)))
            ab = mixed_rdm_q60(c, norb, nelec, tables)
            mixed_seconds = perf_counter() - mixed_start
            if cpu is None:
                from .rdm_factorized import same_spin_rdms
                same_start = perf_counter()
                one, (aa, bb) = same_spin_rdms(c, ci_strings, norb, nelec, tables)
                cpu_seconds = perf_counter() - same_start
            else:
                # 完整独立验证仍构造所有 CPU 分量；最终返回的是 Metal 的 ab 块。
                one, (aa, cpu_ab, bb) = rdms_from_cpu(cpu)
                error = float(np.max(np.abs(ab - cpu_ab)))
                if not np.isfinite(error) or error > 1e-11:
                    raise RuntimeError(f"Metal mixed RDM disagrees with independent CPU: {error:.3e}")
                rdm_metrics["rdm2_ab_cpu_max_abs_error"] = error
                cpu_seconds = float(cpu["rdm_seconds"])
            rdms = one, (aa, ab, bb)
            rdm_algorithm = "cpu_fp64_same_spin+metal_q60_ab" if min(nelec) else "cpu_fp64_same_spin+zero_ab"
            rdm_seconds = mixed_seconds + cpu_seconds
            rdm_metrics.update(rdm_ab_backend="metal_q60" if min(nelec) else "analytic_zero",
                               rdm_ab_kernel_calls=int(min(nelec) > 0),
                               rdm_ab_cpu_reference=cpu is not None,
                               rdm_ab_quantization_bound=c.size * 2.**-60 if min(nelec) else 0.,
                               rdm_ab_seconds=mixed_seconds, rdm_cpu_block_seconds=cpu_seconds)
        subprocess_seconds = float(cpu["seconds"]) if cpu is not None else 0.
        host_started = perf_counter()
        physical = assemble_result(coefficients, ci_strings, h1e, eri, norb, nelec, *rdms)
        report = validate_rdms(physical, bare_rayleigh, energy_tolerance=rdm_energy_tolerance)
        if validation == "full_cpu":
            report.update(compare_cpu(
                physical, cpu, fresh_image_cpu,
                residual_tolerance=operator_residual_tolerance,
                energy_tolerance=rdm_energy_tolerance,
                operator_tolerance=operator_residual_tolerance))
        spin_image = spin_contractor.contract(coefficients_mx.reshape(hdiag.shape))
        spin_expectation = scalar(mx.sum(coefficients_mx.reshape(hdiag.shape) * spin_image))
        spin_error = abs(spin_expectation - physical.spin_square())
        if not np.isfinite(spin_error) or spin_error > 2e-5:
            raise RuntimeError(f"Metal S² and RDM disagree: error={spin_error:.3e}")
        report.update(operator_spin_square_expectation=spin_expectation,
                      spin_square_rdm_abs_error=spin_error,
                      spin_validation="RDM_vs_Metal_S2")
        host_seconds = perf_counter() - host_started
        validation_seconds = perf_counter() - validation_started

    report.update({
        **target_report, **rdm_metrics,
        "solver_backend": "metal_native_sbd", "hamiltonian_backend": "metal",
        "hamiltonian_uses_zero_coupling_spectator": uses_spectator,
        "rdm_backend": rdm_backend, "rdm_algorithm": rdm_algorithm, "validation": validation,
        "independent_cpu_validation": validation == "full_cpu",
        "gpu_acceleration_scope": "hdiag,H@c,Davidson,S2" + (",RDM2ab" if rdm_backend == "metal" and min(nelec) else ""),
        "cpu_scope": ("link_tables,small_eigh," + ("RDM" if rdm_backend == "cpu" else "RDM1a/b,RDM2aa/bb")
                      + (",independent_validation" if validation == "full_cpu" else "")),
        "spin_constraint_applied": spin_sq is not None,
        "spin_backend": "metal", "spin_backend_active": True,
        "target_spin_square": spin_sq, "spin_shift": spin_shift if spin_sq is not None else None,
        "backend": "metal", "device_kind": "metal", "device": "metal:0",
        "device_name": "Apple Metal GPU", "dtype": "float32", "precision_mode": "mixed",
        "threads": int(threads), "selected_ci_dimension": int(coefficients.size),
        "n_alpha_strings": len(ci_strings[0]), "n_beta_strings": len(ci_strings[1]),
        "link_setup_seconds_cpu": link_seconds, "gpu_setup_seconds": setup_seconds,
        "gpu_solver_seconds": solver_seconds, "gpu_iterations": int(solved.iterations),
        "requested_max_space": max_space, "effective_max_space": min(max_space, coefficients.size),
        "gpu_h_matvecs": int(solved.h_matvecs),
        "gpu_residual_norm_device": float(solved.residual_norm),
        "gpu_fresh_residual_norm_device": residual,
        "hamiltonian_energy_offset_hartree": energy_offset,
        "gpu_penalized_eigenvalue_hartree": rayleigh,
        "physical_rayleigh_energy_hartree": bare_rayleigh,
        "spin_penalty_expectation_hartree": rayleigh - bare_rayleigh,
        "cpu_validation_seconds": subprocess_seconds + host_seconds,
        "cpu_subprocess_seconds": subprocess_seconds,
        "validation_host_check_seconds": host_seconds, "rdm_seconds": rdm_seconds,
        "validation_seconds": validation_seconds,
        "native_metal_calls": {**contractor.native_calls, **{
            f"spin_{key}": value for key, value in spin_contractor.native_calls.items()},
            **({"metal_mixed_rdm_q60": rdm_metrics["rdm_ab_kernel_calls"]} if rdm_backend == "metal" else {})},
        "native_metal_execution": {
            "hamiltonian": dict(contractor.execution), "spin": dict(spin_contractor.execution)},
        "max_memory_fraction": max_memory_fraction,
        "initial_state_projected": initial is not None, "sbd_context_reused": False,
        "gpu_history": [{**step, "energy": step["energy"] + energy_offset} for step in solved.history],
    })
    if progress_callback is not None:
        progress_callback({"stage": "metal_native", "status": "complete", "residual_norm": residual})
    return physical, report
