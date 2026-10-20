"""固定轨道 PCM-SQD correlated-density SCRF 外循环。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from pyscf import gto, solvent

from .fermion import SCIResult, run_sqd_fermion, solve_sci_batch


PCM_ALIASES = {
    "c-pcm": "C-PCM",
    "cpcm": "C-PCM",
    "ief-pcm": "IEF-PCM",
    "iefpcm": "IEF-PCM",
}


@dataclass(frozen=True)
class PCMActiveSpaceProblem:
    """固定轨道、固定二电子积分的活性空间问题。"""

    mol: gto.Mole
    h1e_vacuum: np.ndarray
    eri_chemist: np.ndarray
    ecore_vacuum: float
    active_mo_coeff: np.ndarray
    inactive_core_mo_coeff: np.ndarray
    nelec: tuple[int, int]

    @property
    def norb(self) -> int:
        """返回活性空间轨道数。"""

        return int(self.h1e_vacuum.shape[0])

    def validate(self) -> None:
        """尽早拒绝 shape、粒子数和非有限数错误。"""

        norb = self.norb
        nao = int(self.mol.nao_nr())
        expected_eri_shape = (norb, norb, norb, norb)
        if self.h1e_vacuum.shape != (norb, norb):
            raise ValueError("h1e_vacuum 必须是方阵。")
        if self.eri_chemist.shape != expected_eri_shape:
            raise ValueError(
                f"eri_chemist shape={self.eri_chemist.shape}，预期 {expected_eri_shape}。"
            )
        if self.active_mo_coeff.shape != (nao, norb):
            raise ValueError(
                "active_mo_coeff shape 错误："
                f"{self.active_mo_coeff.shape} != {(nao, norb)}。"
            )
        if self.inactive_core_mo_coeff.ndim != 2 or (
            self.inactive_core_mo_coeff.shape[0] != nao
        ):
            raise ValueError("inactive_core_mo_coeff 的 AO 维度错误。")
        if any(value < 0 or value > norb for value in self.nelec):
            raise ValueError(f"活性电子数超出轨道范围：{self.nelec}。")
        for name, array in (
            ("h1e_vacuum", self.h1e_vacuum),
            ("eri_chemist", self.eri_chemist),
            ("active_mo_coeff", self.active_mo_coeff),
            ("inactive_core_mo_coeff", self.inactive_core_mo_coeff),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} 含非有限数。")


@dataclass(frozen=True)
class PCMSQDSettings:
    """一次 PCM 外循环中固定不变的 SQD 参数。"""

    samples_per_batch: int
    num_batches: int
    max_iterations: int
    max_dim: int | None
    spin_sq: float | None
    spin_shift: float = 0.1
    spin_square_tolerance: float = 5.0e-2
    symmetrize_spin: bool = False
    average_batch_occupancies: bool = True
    seed: int = 20260902
    energy_tol: float = 1.0e-8
    occupancies_tol: float = 1.0e-5
    carryover_threshold: float = 1.0e-4
    sci_tol: float = 1.0e-9
    sci_max_cycle: int = 80
    sci_max_space: int = 12
    sci_max_memory_mb: int = 2000


@dataclass(frozen=True)
class PCMSCRFSettings:
    """PCM correlated-density SCRF 外循环参数。"""

    pcm_method: str
    eps: float = 78.3553
    max_iterations: int = 12
    energy_tol: float = 1.0e-8
    density_tol: float = 1.0e-7
    mixing: float = 1.0
    pcm_max_memory_mb: int = 8000


@dataclass(frozen=True)
class PCMSCRFCheckpoint:
    """保存继续下一轮 PCM-SQD SCRF 所需的数值状态。"""

    next_outer_iteration: int
    active_rdm1: np.ndarray
    occupancies: tuple[np.ndarray, np.ndarray]
    previous_total_energy: float
    trajectory: tuple[dict[str, Any], ...]
    cached_pcm_response: tuple[float, np.ndarray] | None
    norb: int
    nelec: tuple[int, int]
    pcm_method: str
    eps: float
    mixing: float
    converged: bool


@dataclass(frozen=True)
class PCMSCRFResult:
    """保留最终 SQD 波函数、外循环轨迹和可恢复状态。"""

    converged: bool
    pcm_method: str
    sci_result: SCIResult
    active_rdm1: np.ndarray
    ao_rdm1: np.ndarray
    total_energy: float
    trajectory: tuple[dict[str, Any], ...]
    checkpoint: PCMSCRFCheckpoint


def canonical_pcm_method(method: str) -> str:
    """把常见写法统一成 PySCF 接受的 PCM 方法名称。"""

    canonical = PCM_ALIASES.get(method.strip().lower())
    if canonical is None:
        raise ValueError("pcm_method 必须是 C-PCM/CPCM 或 IEF-PCM/IEFPCM。")
    return canonical


def make_pcm_model(mol: gto.Mole, settings: PCMSCRFSettings):
    """按给定设置创建 PySCF PCM 对象。"""

    method = canonical_pcm_method(settings.pcm_method)
    model = solvent.PCM(mol)
    model.method = method
    model.eps = float(settings.eps)
    model.max_memory = int(settings.pcm_max_memory_mb)
    model.verbose = 0
    return model


def build_ao_density(
    active_rdm1: np.ndarray,
    active_mo_coeff: np.ndarray,
    inactive_core_mo_coeff: np.ndarray,
) -> np.ndarray:
    """把 correlated active 1-RDM 与双占据冻结核合成为 AO 1-RDM。"""

    active_rdm1 = np.asarray(active_rdm1, dtype=float)
    active = np.asarray(active_mo_coeff, dtype=float)
    core = np.asarray(inactive_core_mo_coeff, dtype=float)
    if active_rdm1.shape != (active.shape[1], active.shape[1]):
        raise ValueError("active_rdm1 与 active_mo_coeff 的轨道数不一致。")
    if core.ndim != 2 or core.shape[0] != active.shape[0]:
        raise ValueError("inactive_core_mo_coeff 的 AO 维度错误。")

    # 显式收缩避免服务器 LoongArch64/BLAS 对窄矩阵 matmul 的数值错误。
    density = np.einsum(
        "pi,ij,qj->pq",
        active,
        active_rdm1,
        active,
        optimize=False,
    )
    if core.shape[1]:
        density = density + 2.0 * np.einsum("pi,qi->pq", core, core, optimize=False)
    return 0.5 * (density + density.T)


def project_ao_density_to_active(
    mol: gto.Mole,
    ao_rdm1: np.ndarray,
    active_mo_coeff: np.ndarray,
    inactive_core_mo_coeff: np.ndarray,
) -> np.ndarray:
    """从 AO 密度中扣除冻结核，再投影回正交 active MO 基。"""

    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)
    active = np.asarray(active_mo_coeff, dtype=float)
    core = np.asarray(inactive_core_mo_coeff, dtype=float)
    active_ao = np.asarray(ao_rdm1, dtype=float)
    if active_ao.shape != overlap.shape:
        raise ValueError("ao_rdm1 与 AO overlap 的 shape 不一致。")
    if core.ndim != 2 or core.shape[0] != active.shape[0]:
        raise ValueError("inactive_core_mo_coeff 的 AO 维度错误。")
    if core.shape[1]:
        active_ao = active_ao - 2.0 * np.einsum("pi,qi->pq", core, core, optimize=False)
    overlap_active = np.einsum("pq,qi->pi", overlap, active, optimize=False)
    weighted = np.einsum("pi,pq->iq", overlap_active, active_ao, optimize=False)
    projected = np.einsum("ip,pj->ij", weighted, overlap_active, optimize=False)
    return 0.5 * (projected + projected.T)


def pcm_response_bookkeeping(
    problem: PCMActiveSpaceProblem,
    ao_rdm1: np.ndarray,
    pcm_energy: float,
    vpcm_ao: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """投影 PCM 势，并构造去除重复反应场贡献后的动态常数项。"""

    active = problem.active_mo_coeff
    core = problem.inactive_core_mo_coeff
    vpcm_ao = np.asarray(vpcm_ao, dtype=float)
    nao = int(problem.mol.nao_nr())
    if vpcm_ao.shape != (nao, nao):
        raise ValueError("vpcm_ao shape 错误。")
    if np.asarray(ao_rdm1).shape != (nao, nao):
        raise ValueError("ao_rdm1 shape 错误。")
    vpcm_times_active = np.einsum("pq,qi->pi", vpcm_ao, active, optimize=False)
    vpcm_active = np.einsum("pi,pj->ij", active, vpcm_times_active, optimize=False)
    if core.shape[1]:
        vpcm_times_core = np.einsum("pq,qi->pi", vpcm_ao, core, optimize=False)
        vpcm_core = 2.0 * float(
            np.einsum("pi,pi->", core, vpcm_times_core, optimize=False)
        )
    else:
        vpcm_core = 0.0
    vpcm_full = float(np.einsum("ij,ji->", vpcm_ao, np.asarray(ao_rdm1), optimize=True))
    double_count_correction = float(pcm_energy) - vpcm_full
    dynamic_ecore = float(problem.ecore_vacuum) + vpcm_core + double_count_correction
    return {
        "vpcm_active": vpcm_active,
        "vpcm_core": vpcm_core,
        "vpcm_full": vpcm_full,
        "double_count_correction": double_count_correction,
        "dynamic_ecore": dynamic_ecore,
    }


def _validate_settings(sqd: PCMSQDSettings, scrf: PCMSCRFSettings) -> None:
    """验证两层迭代参数，避免运行中途才发现明显错误。"""

    for name, value in (
        ("samples_per_batch", sqd.samples_per_batch),
        ("num_batches", sqd.num_batches),
        ("SQD max_iterations", sqd.max_iterations),
        ("sci_max_cycle", sqd.sci_max_cycle),
        ("sci_max_space", sqd.sci_max_space),
        ("sci_max_memory_mb", sqd.sci_max_memory_mb),
        ("SCRF max_iterations", scrf.max_iterations),
        ("pcm_max_memory_mb", scrf.pcm_max_memory_mb),
    ):
        if int(value) < 1:
            raise ValueError(f"{name} 必须至少为 1。")
    if sqd.max_dim is not None and int(sqd.max_dim) < 1:
        raise ValueError("max_dim 必须为正整数或 None。")
    if sqd.spin_sq is not None and float(sqd.spin_sq) < 0.0:
        raise ValueError("spin_sq 不能为负数。")
    if not np.isfinite(sqd.spin_shift) or float(sqd.spin_shift) <= 0.0:
        raise ValueError("spin_shift 必须为有限正数。")
    if not np.isfinite(sqd.spin_square_tolerance) or float(
        sqd.spin_square_tolerance
    ) < 0.0:
        raise ValueError("spin_square_tolerance 必须为有限非负数。")
    for name, value in (
        ("SQD energy_tol", sqd.energy_tol),
        ("occupancies_tol", sqd.occupancies_tol),
        ("carryover_threshold", sqd.carryover_threshold),
        ("sci_tol", sqd.sci_tol),
        ("SCRF energy_tol", scrf.energy_tol),
        ("density_tol", scrf.density_tol),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} 不能为负数。")
    if not 0.0 < float(scrf.mixing) <= 1.0:
        raise ValueError("mixing 必须位于 (0, 1]。")
    if float(scrf.eps) <= 0.0:
        raise ValueError("eps 必须为正数。")


def _validate_active_rdm1(
    active_rdm1: np.ndarray,
    problem: PCMActiveSpaceProblem,
) -> np.ndarray:
    """检查 active 1-RDM 的 shape、有限性和电子数迹。"""

    density = np.asarray(active_rdm1, dtype=float)
    if density.shape != (problem.norb, problem.norb):
        raise ValueError("active_rdm1 shape 错误。")
    if not np.all(np.isfinite(density)):
        raise ValueError("active_rdm1 含非有限数。")
    if not np.isclose(np.trace(density), sum(problem.nelec), atol=1.0e-7):
        raise ValueError("active_rdm1 的电子数迹错误。")
    return 0.5 * (density + density.T)


def _copy_occupancies(
    occupancies: tuple[np.ndarray, np.ndarray],
    problem: PCMActiveSpaceProblem,
) -> tuple[np.ndarray, np.ndarray]:
    """检查并复制 alpha/beta 轨道占据数。"""

    copied = tuple(np.asarray(values, dtype=float).copy() for values in occupancies)
    if any(values.shape != (problem.norb,) for values in copied):
        raise ValueError("occupancies 的 shape 错误。")
    if not np.allclose(
        [np.sum(values) for values in copied], problem.nelec, atol=1.0e-7, rtol=0.0
    ):
        raise ValueError("occupancies 的 alpha/beta 电子数错误。")
    return copied  # type: ignore[return-value]


def _validate_resume_checkpoint(
    checkpoint: PCMSCRFCheckpoint,
    problem: PCMActiveSpaceProblem,
    method: str,
    scrf: PCMSCRFSettings,
) -> None:
    """确认 checkpoint 与本次问题和 PCM 设置兼容。"""

    if checkpoint.converged:
        raise ValueError("checkpoint 已经收敛，不需要继续运行。")
    if checkpoint.norb != problem.norb or checkpoint.nelec != problem.nelec:
        raise ValueError("checkpoint 的活性空间或电子数与当前问题不一致。")
    if checkpoint.pcm_method != method:
        raise ValueError("checkpoint 的 PCM 方法与当前设置不一致。")
    if not np.isclose(checkpoint.eps, scrf.eps, atol=0.0, rtol=0.0):
        raise ValueError("checkpoint 的 eps 与当前设置不一致。")
    if not np.isclose(checkpoint.mixing, scrf.mixing, atol=0.0, rtol=0.0):
        raise ValueError("checkpoint 的 mixing 与当前设置不一致。")
    if not 1 <= checkpoint.next_outer_iteration <= scrf.max_iterations:
        raise ValueError(
            "checkpoint 的下一轮编号超出 SCRF max_iterations；"
            "请增大 max_iterations 后再继续。"
        )
    _validate_active_rdm1(checkpoint.active_rdm1, problem)
    _copy_occupancies(checkpoint.occupancies, problem)
    if not np.isfinite(checkpoint.previous_total_energy):
        raise ValueError("checkpoint 的上一轮总能量不是有限数。")
    if checkpoint.cached_pcm_response is not None:
        pcm_energy, vpcm_ao = checkpoint.cached_pcm_response
        nao = int(problem.mol.nao_nr())
        if not np.isfinite(pcm_energy) or np.asarray(vpcm_ao).shape != (nao, nao):
            raise ValueError("checkpoint 的 PCM 缓存无效。")


def run_pcm_sqd_scrf(
    problem: PCMActiveSpaceProblem,
    samples: Mapping[str, float | int] | np.ndarray | Sequence[Sequence[bool | int]],
    initial_active_rdm1: np.ndarray | None,
    sqd: PCMSQDSettings,
    scrf: PCMSCRFSettings,
    *,
    sample_order: str = "alpha_beta",
    include_configurations: tuple[list[int], list[int]] | None = None,
    initial_occupancies: tuple[np.ndarray, np.ndarray] | None = None,
    pcm_model: Any | None = None,
    initial_pcm_response: tuple[float, np.ndarray] | None = None,
    resume_from: PCMSCRFCheckpoint | None = None,
    iteration_callback: Callable[[PCMSCRFCheckpoint], None] | None = None,
    inner_iteration_callback: Callable[[int, int, list[SCIResult]], None] | None = None,
    sci_solver_override: Callable[..., list[SCIResult]] | None = None,
) -> PCMSCRFResult:
    """固定 samples、轨道和 ERI，只用 correlated 1-RDM 更新 PCM 与 h1。"""

    problem.validate()
    _validate_settings(sqd, scrf)
    method = canonical_pcm_method(scrf.pcm_method)
    model = pcm_model if pcm_model is not None else make_pcm_model(problem.mol, scrf)
    model.method = method
    model.eps = float(scrf.eps)
    model.max_memory = int(scrf.pcm_max_memory_mb)

    if resume_from is None:
        if initial_active_rdm1 is None:
            raise ValueError("新运行必须提供 initial_active_rdm1。")
        current_dm1 = _validate_active_rdm1(initial_active_rdm1, problem)
        current_occupancies = (
            None
            if initial_occupancies is None
            else _copy_occupancies(initial_occupancies, problem)
        )
        previous_energy: float | None = None
        trajectory: list[dict[str, Any]] = []
        cached_response = initial_pcm_response
        start_iteration = 1
    else:
        if initial_active_rdm1 is not None:
            raise ValueError("续跑时 initial_active_rdm1 必须为 None。")
        if initial_occupancies is not None or initial_pcm_response is not None:
            raise ValueError("续跑状态已包含 occupancies 和 PCM 缓存，不能重复提供。")
        _validate_resume_checkpoint(resume_from, problem, method, scrf)
        current_dm1 = _validate_active_rdm1(resume_from.active_rdm1, problem).copy()
        current_occupancies = _copy_occupancies(resume_from.occupancies, problem)
        previous_energy = float(resume_from.previous_total_energy)
        trajectory = [dict(item) for item in resume_from.trajectory]
        cached_response = resume_from.cached_pcm_response
        start_iteration = int(resume_from.next_outer_iteration)

    if cached_response is not None:
        pcm_energy_cached, vpcm_cached = cached_response
        nao = int(problem.mol.nao_nr())
        if not np.isfinite(pcm_energy_cached) or np.asarray(vpcm_cached).shape != (
            nao,
            nao,
        ):
            raise ValueError("initial_pcm_response 的数值或 shape 无效。")
        cached_response = (float(pcm_energy_cached), np.asarray(vpcm_cached).copy())

    if sci_solver_override is None:
        # spin_sq 通过现有求解器扩展点传入，不改变 SQD 内层公共接口。
        def sci_solver(strings, one_body, two_body, norb, nelec):
            return solve_sci_batch(
                strings,
                one_body,
                two_body,
                norb,
                nelec,
                spin_sq=sqd.spin_sq,
                spin_shift=float(sqd.spin_shift),
                tol=float(sqd.sci_tol),
                max_cycle=int(sqd.sci_max_cycle),
                max_space=int(sqd.sci_max_space),
                max_memory=int(sqd.sci_max_memory_mb),
                verbose=0,
            )
    else:
        # 服务器 runner 可在进程池中并行三个 batch；调用者必须按 sqd.spin_sq
        # 配置同一个受约束的 selected-CI 求解器。
        sci_solver = sci_solver_override

    final_result: SCIResult | None = None
    final_active_rdm1: np.ndarray | None = None
    final_ao_rdm1: np.ndarray | None = None
    final_total_energy: float | None = None
    final_checkpoint: PCMSCRFCheckpoint | None = None
    converged = False

    for outer_iteration in range(start_iteration, scrf.max_iterations + 1):
        iteration_started = perf_counter()
        input_ao_rdm1 = build_ao_density(
            current_dm1,
            problem.active_mo_coeff,
            problem.inactive_core_mo_coeff,
        )
        if cached_response is None:
            pcm_input_started = perf_counter()
            pcm_energy_in, vpcm_ao_in = model.kernel(input_ao_rdm1)
            pcm_input_seconds = perf_counter() - pcm_input_started
        else:
            pcm_energy_in, vpcm_ao_in = cached_response
            cached_response = None
            pcm_input_seconds = 0.0
        bookkeeping = pcm_response_bookkeeping(
            problem,
            input_ao_rdm1,
            float(pcm_energy_in),
            np.asarray(vpcm_ao_in),
        )
        vpcm_active = np.asarray(bookkeeping["vpcm_active"], dtype=float)

        # 保留每一轮 S-CORE 的全部 batch；最后一轮的 batch 平均 1-RDM
        # 用于更新反应场，和 David 等论文的式 (13)、式 (15) 一致。
        inner_iteration = 0
        final_inner_results: list[SCIResult] = []

        def capture_inner_results(results: list[SCIResult]) -> None:
            nonlocal inner_iteration, final_inner_results
            inner_iteration += 1
            final_inner_results = list(results)
            if inner_iteration_callback is not None:
                inner_iteration_callback(
                    outer_iteration,
                    inner_iteration,
                    final_inner_results,
                )

        sqd_started = perf_counter()
        result = run_sqd_fermion(
            problem.h1e_vacuum + vpcm_active,
            problem.eri_chemist,
            samples,
            samples_per_batch=int(sqd.samples_per_batch),
            norb=problem.norb,
            nelec=problem.nelec,
            sample_order=sample_order,
            nuclear_repulsion_energy=float(bookkeeping["dynamic_ecore"]),
            num_batches=int(sqd.num_batches),
            energy_tol=float(sqd.energy_tol),
            occupancies_tol=float(sqd.occupancies_tol),
            max_iterations=int(sqd.max_iterations),
            sci_solver=sci_solver,
            symmetrize_spin=bool(sqd.symmetrize_spin),
            max_dim=sqd.max_dim,
            include_configurations=include_configurations,
            initial_occupancies=current_occupancies,
            carryover_threshold=float(sqd.carryover_threshold),
            average_batch_occupancies=bool(sqd.average_batch_occupancies),
            callback=capture_inner_results,
            seed=int(sqd.seed),
            ext_sqd=False,
        )
        sqd_seconds = perf_counter() - sqd_started
        if not final_inner_results:
            # 兼容测试替身和旧的外部 run_sqd_fermion 实现；正式实现总会触发 callback。
            final_inner_results = [result]
        if any(item.rdm1 is None or item.rdm1s is None for item in final_inner_results):
            raise RuntimeError("SQD 没有返回 SCRF 所需的 batch 1-RDM。")
        batch_spin_squares = [
            float(
                item.spin_square()
                if hasattr(item, "spin_square")
                else item.sci_state.spin_square()
            )
            for item in final_inner_results
        ]
        if sqd.spin_sq is not None and any(
            abs(value - float(sqd.spin_sq)) > float(sqd.spin_square_tolerance)
            for value in batch_spin_squares
        ):
            raise RuntimeError(
                "selected-CI 的实际 <S^2> 未达到目标软约束："
                f"target={sqd.spin_sq}, values={batch_spin_squares}, "
                f"tolerance={sqd.spin_square_tolerance}, shift={sqd.spin_shift}。"
            )

        new_dm1 = np.mean(
            [np.asarray(item.rdm1, dtype=float) for item in final_inner_results],
            axis=0,
        )
        averaged_rdm1s = tuple(
            np.mean(
                [np.asarray(item.rdm1s[spin], dtype=float) for item in final_inner_results],
                axis=0,
            )
            for spin in (0, 1)
        )
        traces = tuple(float(np.trace(block)) for block in averaged_rdm1s)
        if not np.allclose(traces, problem.nelec, atol=1.0e-7, rtol=0.0):
            raise RuntimeError(f"SQD spin-resolved 1-RDM 粒子数错误：{traces}。")
        new_ao_rdm1 = build_ao_density(
            new_dm1,
            problem.active_mo_coeff,
            problem.inactive_core_mo_coeff,
        )
        pcm_output_started = perf_counter()
        batch_ao_rdm1s = [
            build_ao_density(
                np.asarray(item.rdm1, dtype=float),
                problem.active_mo_coeff,
                problem.inactive_core_mo_coeff,
            )
            for item in final_inner_results
        ]
        batch_pcm_energies: list[float] = []
        batch_pcm_potentials: list[np.ndarray] = []
        for batch_ao_rdm1 in batch_ao_rdm1s:
            batch_pcm_energy, batch_pcm_potential = model.kernel(batch_ao_rdm1)
            batch_pcm_energies.append(float(batch_pcm_energy))
            batch_pcm_potentials.append(np.asarray(batch_pcm_potential).copy())
        if len(final_inner_results) == 1:
            pcm_energy_out = batch_pcm_energies[0]
            vpcm_ao_out = batch_pcm_potentials[0]
        else:
            # 下一轮反应场只由 batch 平均总密度驱动。
            pcm_energy_out, vpcm_ao_out = model.kernel(new_ao_rdm1)
        pcm_output_seconds = perf_counter() - pcm_output_started

        # 各 batch 先从带旧反应场势的电子能中移除线性项，再用该 batch
        # 的总密度单独计算 PCM 自由能；最终按论文式 (16) 取最低 batch。
        batch_vacuum_active_energies = [
            float(item.energy)
            - float(
                np.einsum(
                    "pq,qp->",
                    vpcm_active,
                    np.asarray(item.rdm1, dtype=float),
                    optimize=True,
                )
            )
            for item in final_inner_results
        ]
        batch_total_energies = [
            vacuum_energy + float(problem.ecore_vacuum) + pcm_energy
            for vacuum_energy, pcm_energy in zip(
                batch_vacuum_active_energies,
                batch_pcm_energies,
                strict=True,
            )
        ]
        best_batch_index = int(np.argmin(batch_total_energies))
        result = final_inner_results[best_batch_index]
        physical_total_energy = batch_total_energies[best_batch_index]
        density_residual = float(np.max(np.abs(new_dm1 - current_dm1)))
        energy_change = (
            None if previous_energy is None else physical_total_energy - previous_energy
        )
        spin_square = batch_spin_squares[best_batch_index]
        trajectory.append(
            {
                "outer_iteration": outer_iteration,
                "physical_total_energy_hartree": physical_total_energy,
                "energy_change_hartree": energy_change,
                "active_density_max_abs_change": density_residual,
                "pcm_input_energy_hartree": float(pcm_energy_in),
                "pcm_output_energy_hartree": float(pcm_energy_out),
                "linearized_total_energy_hartree": float(result.total_energy),
                "dynamic_ecore_hartree": float(bookkeeping["dynamic_ecore"]),
                "vpcm_core_hartree": float(bookkeeping["vpcm_core"]),
                "vpcm_full_hartree": float(bookkeeping["vpcm_full"]),
                "pcm_double_count_correction_hartree": float(
                    bookkeeping["double_count_correction"]
                ),
                "rdm1_spin_traces": list(traces),
                "spin_square": spin_square,
                "batch_spin_squares": batch_spin_squares,
                "batch_vacuum_active_energies_hartree": batch_vacuum_active_energies,
                "batch_pcm_energies_hartree": batch_pcm_energies,
                "batch_total_energies_hartree": batch_total_energies,
                "inner_iterations_completed": inner_iteration,
                "n_alpha_strings": int(len(result.sci_state.ci_strs_a)),
                "n_beta_strings": int(len(result.sci_state.ci_strs_b)),
                "selected_ci_dimension": int(
                    len(result.sci_state.ci_strs_a) * len(result.sci_state.ci_strs_b)
                ),
                "pcm_input_seconds": pcm_input_seconds,
                "sqd_seconds": sqd_seconds,
                "pcm_output_seconds": pcm_output_seconds,
                "iteration_seconds": perf_counter() - iteration_started,
            }
        )

        converged = bool(
            energy_change is not None
            and abs(energy_change) < scrf.energy_tol
            and density_residual < scrf.density_tol
        )
        mixed_dm1 = (1.0 - float(scrf.mixing)) * current_dm1 + float(
            scrf.mixing
        ) * new_dm1
        next_dm1 = new_dm1 if converged else 0.5 * (mixed_dm1 + mixed_dm1.T)
        next_occupancies = tuple(
            np.mean(
                [
                    np.asarray(item.orbital_occupancies[spin], dtype=float)
                    for item in final_inner_results
                ],
                axis=0,
            )
            for spin in (0, 1)
        )
        next_cached_response = (
            (float(pcm_energy_out), np.asarray(vpcm_ao_out).copy())
            if scrf.mixing == 1.0
            else None
        )
        final_checkpoint = PCMSCRFCheckpoint(
            next_outer_iteration=outer_iteration + 1,
            active_rdm1=np.asarray(next_dm1).copy(),
            occupancies=next_occupancies,  # type: ignore[arg-type]
            previous_total_energy=physical_total_energy,
            trajectory=tuple(dict(item) for item in trajectory),
            cached_pcm_response=next_cached_response,
            norb=problem.norb,
            nelec=problem.nelec,
            pcm_method=method,
            eps=float(scrf.eps),
            mixing=float(scrf.mixing),
            converged=converged,
        )
        if iteration_callback is not None:
            iteration_callback(final_checkpoint)

        print(
            f"[SCRF {outer_iteration}/{scrf.max_iterations}] "
            f"E={physical_total_energy:.12f} "
            f"dE={energy_change} dD={density_residual:.3e} "
            f"S2={spin_square:.8f} "
            f"t_pcm_in={pcm_input_seconds:.1f}s "
            f"t_sqd={sqd_seconds:.1f}s "
            f"t_pcm_out={pcm_output_seconds:.1f}s",
            flush=True,
        )

        final_result = result
        final_active_rdm1 = new_dm1
        final_ao_rdm1 = new_ao_rdm1
        final_total_energy = physical_total_energy
        if converged:
            break
        current_dm1 = np.asarray(next_dm1)
        current_occupancies = next_occupancies  # type: ignore[assignment]
        previous_energy = physical_total_energy
        cached_response = next_cached_response

    if (
        final_result is None
        or final_active_rdm1 is None
        or final_ao_rdm1 is None
        or final_total_energy is None
        or final_checkpoint is None
    ):
        raise RuntimeError("SCRF 未产生任何 SQD 结果。")
    return PCMSCRFResult(
        converged=converged,
        pcm_method=method,
        sci_result=final_result,
        active_rdm1=final_active_rdm1,
        ao_rdm1=final_ao_rdm1,
        total_energy=final_total_energy,
        trajectory=tuple(trajectory),
        checkpoint=final_checkpoint,
    )


__all__ = [
    "PCMActiveSpaceProblem",
    "PCMSCRFCheckpoint",
    "PCMSCRFResult",
    "PCMSCRFSettings",
    "PCMSQDSettings",
    "build_ao_density",
    "canonical_pcm_method",
    "make_pcm_model",
    "pcm_response_bookkeeping",
    "project_ao_density_to_active",
    "run_pcm_sqd_scrf",
]
