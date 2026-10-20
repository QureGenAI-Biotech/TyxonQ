"""可选 SBD TPB 后端；导入此模块不会加载 CUDA、MPI 或编译扩展。"""

from __future__ import annotations

import importlib
import os
from time import perf_counter

import numpy as np

from .validation import (
    assemble_result, compare_cpu, independent_cpu_validation, normalize_problem,
    project_initial_state, rdms_from_cpu, validate_rdms, validate_spin_square,
    validation_failure_snapshot,
)


def _load_extension():
    """仅在实际计算时加载已经显式构建的扩展。"""
    try:
        name = "tyxonq.applications.chem.algorithms.sqd.gpu._sbd_tpb_cuda"
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
            # --output-dir 构建结果可由用户显式放到 PYTHONPATH。
            return importlib.import_module("_sbd_tpb_cuda")
    except ImportError as error:
        raise ImportError(
            "SBD CUDA extension is unavailable. Build it explicitly with "
            "python -m tyxonq.applications.chem.algorithms.sqd.gpu.sbd_ext.build; "
            "ordinary CPU/Torch usage does not require this extension."
        ) from error


def interleaved_phase(ci_strings, norb, *, device):
    """在指定设备构造交错自旋轨道到先 α 后 β 排列的费米子符号。"""
    import torch

    a = torch.as_tensor(ci_strings[0], dtype=torch.int64, device=device)
    b = torch.as_tensor(ci_strings[1], dtype=torch.int64, device=device)
    parity = torch.zeros((a.numel(), b.numel()), dtype=torch.int8, device=device)
    lower_b = torch.zeros(b.numel(), dtype=torch.int8, device=device)
    for orbital in range(norb):
        occupied_a = ((a >> orbital) & 1).to(torch.int8)
        parity.bitwise_xor_(occupied_a[:, None] * lower_b[None, :])
        lower_b.bitwise_xor_(((b >> orbital) & 1).to(torch.int8))
    return (1.0 - 2.0 * parity.to(torch.float64)).contiguous()


def convert_raw_rdms(raw, norb):
    """把上游 s+2t 自旋块和 Fortran 指标顺序转成 PySCF RDM。"""
    one = raw["one_p_rdm"]
    two = raw["two_p_rdm"]
    if len(one) != 2 or len(two) != 4:
        raise ValueError("SBD must return two 1-RDM and four 2-RDM spin blocks.")
    dm1 = tuple(np.asarray(x, dtype=np.float64).reshape((norb, norb), order="F").copy()
                for x in one)
    dm2 = tuple(np.asarray(x, dtype=np.float64).reshape((norb,) * 4, order="F")
                .transpose(0, 2, 1, 3).copy() for x in two)
    # SBD 的块顺序是 aa,ba,ab,bb，而本方只保存 aa,ab,bb。
    # 检查被省略的 ba，避免总密度求和掩盖自旋块交换错误。
    if not np.allclose(dm2[1], dm2[2].transpose(2, 3, 0, 1), atol=1e-8, rtol=1e-8):
        raise RuntimeError("SBD alpha-beta and beta-alpha RDM blocks are inconsistent.")
    return dm1, (dm2[0], dm2[2], dm2[3])


class SBDContractor:
    """持有固定组态的连接表，明确更新积分并提供 PyTorch 设备接口。"""

    def __init__(self, h1e, eri, ci_strings, norb, nelec, *, device, memory_fraction=0.8):
        normalized, h1e, eri, norb, nelec = normalize_problem(ci_strings, h1e, eri, norb, nelec)
        if any(not np.array_equal(original, clean) for original, clean in zip(ci_strings, normalized)):
            raise ValueError("SBDContractor requires sorted unique determinant labels.")
        if not 0 < memory_fraction <= 0.8:
            raise ValueError("memory_fraction must be in (0, 0.8].")
        import torch

        self.device = torch.device(device)
        self.shape = tuple(map(len, ci_strings))
        self.norb = norb
        self.memory_fraction = memory_fraction
        self.ci_strings = tuple(np.array(x, copy=True) for x in ci_strings)
        self.nelec = nelec
        self.phase = interleaved_phase(ci_strings, norb, device=self.device)
        # 把 "cuda" 别名固定到实际分配的卡，避免之后切换 current_device 影响此上下文。
        self.device = self.phase.device
        device_id = torch.cuda.current_device() if self.device.index is None else self.device.index
        self.context = _load_extension().TPBContext(
            h1e, eri, *ci_strings, norb, nelec, device_id=device_id,
            memory_fraction=memory_fraction)

    def update_integrals(self, h1e, eri):
        """PCM 新积分必须刷新 C++ 积分及对角元，连接表保持不变。"""
        _, h1e, eri, _, _ = normalize_problem(self.ci_strings, h1e, eri, self.norb, self.nelec)
        self.context.update_integrals(h1e, eri)

    def assert_compatible(self, ci_strings, norb, nelec, device, *, memory_fraction=None):
        """禁止在不同组态、粒子数或设备上复用旧连接表。"""
        import torch

        requested = torch.device(device)
        index = torch.cuda.current_device() if requested.index is None else requested.index
        current = self.phase.device
        if (norb != self.norb or tuple(nelec) != tuple(self.nelec)
                or requested.type != current.type or index != current.index
                or any(not np.array_equal(x, y) for x, y in zip(ci_strings, self.ci_strings))):
            raise ValueError("sbd_context does not match determinants, norb, nelec or CUDA device.")
        if memory_fraction is not None and memory_fraction != self.memory_fraction:
            raise ValueError("sbd_context has a different memory_fraction budget.")

    def diagonal(self):
        import torch

        return torch.as_tensor(np.asarray(self.context.hdiag()).reshape(self.shape),
                               dtype=torch.float64, device=self.device)

    def _raw(self, coefficients):
        import torch

        if (coefficients.dtype != torch.float64 or coefficients.device != self.phase.device
                or coefficients.numel() != self.phase.numel()):
            raise ValueError("SBD coefficients must be FP64 on the context device with matching size.")
        return (coefficients.reshape(self.shape) * self.phase).contiguous()

    def contract(self, coefficients):
        """只做同卡复制；同步前后保持输入/输出 Tensor 引用存活。"""
        import torch

        raw = self._raw(coefficients)
        output = torch.empty_like(raw)
        torch.cuda.synchronize(self.device)
        self.context.matvec_device(raw.data_ptr(), output.data_ptr(), raw.numel())
        torch.cuda.synchronize(self.device)
        return output * self.phase

    def rdms(self, coefficients):
        import torch

        raw = self._raw(coefficients)
        torch.cuda.synchronize(self.device)
        result = self.context.rdms_device(raw.data_ptr(), raw.numel())
        torch.cuda.synchronize(self.device)
        return convert_raw_rdms(result, self.norb)


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
    from .torch_sci_solver import wait_for_davidson_memory

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
        started = perf_counter()
        raw = contractor.context.solve(tolerance=tolerance, max_cycles=max_cycles,
                                       max_space=effective_space, initial_coefficients=raw_initial,
                                       compute_rdms=rdm_backend == "sbd")
        torch.cuda.synchronize(device)
        native_total_seconds = perf_counter() - started
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
