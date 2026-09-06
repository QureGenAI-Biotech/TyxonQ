"""SQD → PySCF fcisolver 适配器。

把 ``solve_sci`` 包装成满足 ``pyscf.mcscf.CASCI`` / ``pyscf.grad.casci`` 鸭子类型
契约的求解器，使冻结子空间 SQD 可以复用 PySCF 现成的核梯度代码。

设计要点（依据 ``MD_INTEGRATION_RESEARCH.md``）：

- CASCI 传入的 ``h2`` 是 8 重对称压缩格式，而 ``solve_sci`` 需要全 4 指标张量，
  入口必须 ``ao2mo.restore(1, h2, norb)``（§4.5）。
- ``run_sqd_fermion(include_configurations=...)`` 不是冻结机制：它把指定串
  并入采样结果后仍会被 ``max_dim`` 截断。真冻结必须绕过 SQD 主循环，直接调
  ``solve_sci(ci_strs=...)``（§4.3）。
- 随机子空间下能量不是几何的函数，力噪声约 4.3e-03 Hartree/Bohr（§4.4 D），
  因此 ``"refresh"`` 模式只允许做单点能，不能用于解析梯度。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable

import numpy as np
from pyscf import ao2mo

from .fermion import SCIResult, SCIState, run_sqd_fermion, solve_sci
from .samples import reverse_bitstring_halves

_SUBSPACE_MODES = ("frozen", "refresh", "adaptive")


def _to_nelec_pair(nelec: int | tuple[int, int]) -> tuple[int, int]:
    """把 PySCF 的 ``nelec``（int 或 (na, nb)）统一成 (na, nb)。"""
    if isinstance(nelec, tuple):
        return (int(nelec[0]), int(nelec[1]))
    n = int(nelec)
    return (n // 2, n - n // 2)


def _strings_from_user(
    ci_strs: tuple[Sequence[int], Sequence[int]] | SCIState,
) -> tuple[np.ndarray, np.ndarray]:
    """把用户给的冻结串（整数序列或 :class:`SCIState`）转成内部表示。"""
    if isinstance(ci_strs, SCIState):
        return (np.asarray(ci_strs.ci_strs_a, dtype=np.int64), np.asarray(ci_strs.ci_strs_b, dtype=np.int64))
    strs_a, strs_b = ci_strs
    return (
        np.sort(np.unique(np.asarray(strs_a, dtype=np.int64))),
        np.sort(np.unique(np.asarray(strs_b, dtype=np.int64))),
    )


class SQDFCISolver:
    """PySCF fcisolver 鸭子类型的 SQD 实现。

    三种子空间策略：

    - ``"frozen"``：每个几何都在同一行列式子空间内对角化，能量面光滑，
      是唯一允许用于解析梯度 / MD 的模式（默认）。子空间来源按优先级：
      显式 ``ci_strs`` > 首帧用 ``sampler`` 采样确定并锁存。
    - ``"refresh"``：每个几何都重新采样确定子空间。仅可用于单点能；
      随机子空间下能量不是几何的函数（§4.4 D）。
    - ``"adaptive"``：每隔 ``refresh_every`` 个几何重新采样一次，步间冻结。
      能量面分段光滑，仅供研究用途。
    """

    def __init__(
        self,
        *,
        subspace: str = "frozen",
        ci_strs: tuple[Sequence[int], Sequence[int]] | SCIState | None = None,
        sampler: Callable[[np.ndarray, np.ndarray, int, tuple[int, int]], Mapping[str, int]] | None = None,
        sampler_kwargs: dict | None = None,
        spin_sq: float | None = None,
        refresh_every: int = 1,
    ):
        if subspace not in _SUBSPACE_MODES:
            raise ValueError(f"subspace must be one of {_SUBSPACE_MODES}, got {subspace!r}.")
        if subspace == "adaptive" and refresh_every < 1:
            raise ValueError("refresh_every must be >= 1.")
        self.subspace = subspace
        self._sampler = sampler
        # samples_per_batch 是 run_sqd_fermion 的必填位置参数，其余默认对齐 examples/h2o_sqd.py。
        defaults = {"samples_per_batch": 8, "num_batches": 4, "max_iterations": 5, "seed": 7}
        defaults.update(dict(sampler_kwargs or {}))
        self._sampler_kwargs = defaults
        self._spin_sq = spin_sq
        self._refresh_every = int(refresh_every)
        self._frozen_strings: tuple[np.ndarray, np.ndarray] | None = (
            _strings_from_user(ci_strs) if ci_strs is not None else None
        )
        self._last_state: SCIState | None = None
        self._kernel_calls = 0

    # ---- 子空间管理 ----

    @property
    def frozen_strings(self) -> tuple[np.ndarray, np.ndarray] | None:
        """当前锁存的 ``(alpha 串, beta 串)``；尚未冻结时为 ``None``。"""
        return self._frozen_strings

    @property
    def is_frozen(self) -> bool:
        return self._frozen_strings is not None

    @property
    def last_state(self) -> SCIState | None:
        """最近一次对角化得到的 :class:`SCIState`，可用于落盘或显式再冻结。"""
        return self._last_state

    def freeze_from(self, source: SCIResult | SCIState | tuple[Sequence[int], Sequence[int]]) -> None:
        """从 SQD 结果、SCIState 或显式串锁定子空间。"""
        if isinstance(source, SCIResult):
            source = source.sci_state
        self._frozen_strings = _strings_from_user(source)

    def refreeze(self) -> None:
        """丢弃当前锁存串，下一次 ``kernel`` 重新采样确定子空间。"""
        self._frozen_strings = None

    # ---- PySCF fcisolver 契约 ----

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0.0, **kwargs):
        """在当前子空间内对角化，返回 ``(e_tot, amplitudes)``。

        ``e_tot`` 含 ``ecore``（CASCI 的 ``get_h1eff`` 已把核排斥并入 ``h1``），
        不含其它核排斥项；``amplitudes`` 是子空间基下的系数矩阵。
        """
        h1 = np.asarray(h1)
        # CASCI 的 eri_cas 是 8 重压缩格式；solve_sci 需要全 4 指标张量（§4.5）。
        h2 = ao2mo.restore(1, np.asarray(h2), int(norb))
        nelec_pair = _to_nelec_pair(nelec)

        if self._frozen_strings is None:
            need_sample = True
        elif self.subspace == "refresh":
            need_sample = True
        elif self.subspace == "adaptive":
            need_sample = self._should_refresh()
        else:  # frozen
            need_sample = False
        if need_sample:
            if self._sampler is None and self._frozen_strings is None:
                raise ValueError(
                    "No frozen strings and no sampler provided. "
                    "Pass ci_strs=(strs_a, strs_b) or a sampler for the first geometry."
                )
            if self._sampler is not None:
                self._sample_and_freeze(h1, h2, int(norb), nelec_pair)

        result = solve_sci(
            self._frozen_strings,  # type: ignore[arg-type]
            h1,
            h2,
            norb=int(norb),
            nelec=nelec_pair,
            spin_sq=self._spin_sq,
        )
        self._last_state = result.sci_state
        self._kernel_calls += 1
        return float(result.energy) + float(ecore), np.asarray(result.sci_state.amplitudes)

    def make_rdm1(self, ci, norb, nelec):
        """一体约化密度矩阵（spin-summed，CAS-MO 基）。"""
        sci_vec = self._as_sci_vec(ci)
        return self._mc().make_rdm1(sci_vec, int(norb), _to_nelec_pair(nelec))

    def make_rdm12(self, ci, norb, nelec):
        """一体 + 二体约化密度矩阵（spin-summed，CAS-MO 基）。"""
        sci_vec = self._as_sci_vec(ci)
        mc = self._mc()
        nelec_pair = _to_nelec_pair(nelec)
        return mc.make_rdm1(sci_vec, int(norb), nelec_pair), mc.make_rdm2(sci_vec, int(norb), nelec_pair)

    def spin_square(self, ci, norb, nelec):
        """总自旋平方期望与 2S+1。直接转发给 PySCF selected_ci 的真值。"""
        sci_vec = self._as_sci_vec(ci)
        from pyscf.fci.selected_ci import spin_square as _spin_square

        return _spin_square(sci_vec, int(norb), _to_nelec_pair(nelec))

    # ---- 内部 ----

    def _mc(self):
        from pyscf import fci

        return fci.selected_ci.SelectedCI()

    def _as_sci_vec(self, ci):
        """把 kernel 返回的系数矩阵还原成 PySCF SCIvector（带行列式串）。"""
        if self._last_state is None:
            raise RuntimeError("make_rdm* called before kernel().")
        from pyscf.fci.selected_ci import _as_SCIvector

        return _as_SCIvector(np.asarray(ci), (self._last_state.ci_strs_a, self._last_state.ci_strs_b))

    def _should_refresh(self) -> bool:
        # adaptive：步间冻结，每 refresh_every 次 kernel 重采样一次。
        return self._kernel_calls > 0 and self._kernel_calls % self._refresh_every == 0

    def _sample_and_freeze(self, h1, h2, norb, nelec_pair):
        """跑一遍完整 SQD 采样，把最优批次锁存为冻结子空间。"""
        counts = self._sampler(h1, h2, norb, nelec_pair)  # type: ignore[misc]
        result = run_sqd_fermion(
            h1,
            h2,
            counts,
            norb=norb,
            nelec=nelec_pair,
            nuclear_repulsion_energy=0.0,
            **self._sampler_kwargs,
        )
        self.freeze_from(result)


def lucj_sampler(
    mf,
    n_layers: int = 1,
    topology: str = "square",
    shots: int = 4096,
    noise_p: float = 0.0,
    seed: int | None = None,
    optimize: bool = False,
    init_maxiter: int = 100,
    runtime: str = "numeric",
    provider: str = "simulator",
    device: str = "statevector",
) -> Callable[[np.ndarray, np.ndarray, int, tuple[int, int]], dict[str, int]]:
    """构造以 LUCJ 电路采样的 ``sampler``，链路对齐 ``examples/h2o_sqd.py``。

    前提：``mf`` 是收敛的分子平均场，活性空间取其中围绕费米面的连续轨道块，
    即 ``CASCI(mf, norb, nelecas)`` 的默认选法。LUCJ 参数由同基下的活性空间
    frozen CCSD 的 t1/t2 初始化（只算一次，后续几何复用）。
    ``noise_p`` 是 probability 层的 depolarizing 混合强度（0 为无噪声）。

    运行档（``runtime``）——SQD 的「上设备」与 UCCSD/HEA 架构不同：SQD 的量子
    部分是**采样 LUCJ 电路得到 counts**（经典 selected-CI 才对角化），故设备选项
    挂在采样器上、而非 solver_kwargs：

    - ``"numeric"``（默认）：本地 ``StatevectorEngine`` 精确概率 + ``rng.choice(shots)``
      抽样，``seed`` 可复现；
    - ``"device"``：LUCJ 电路补满 ``measure_z`` 后经 ``devices.base.run`` 提交到
      ``provider``/``device``（``shots`` 必须 >0）直接取回计数。与真机走**同一条**
      提交入口，故 ``provider="simulator"`` 验证通过后，切真机只需改
      ``provider``/``device`` 两个字符串（同 E10 第 5 节）；此档由设备自行采样，
      ``seed`` 不生效。

    两档返回的计数都是 TyxonQ/LUCJ raw order，末尾统一经 ``reverse_bitstring_halves``
    转成 SQD/PySCF order。
    """
    from pyscf import cc

    from tyxonq.applications.chem.algorithms.lucj import LUCJ, initialize_lucj_parameters_from_ccsd
    from tyxonq.devices.simulators.statevector.engine import StatevectorEngine

    if runtime not in ("numeric", "device"):
        raise ValueError(f"runtime must be 'numeric' or 'device', got {runtime!r}.")
    if runtime == "device" and int(shots) <= 0:
        raise ValueError("SQD device 采样需要 shots > 0（要有样本才能确定子空间）。")

    cached: dict = {}

    def sampler(h1, h2, norb, nelec_pair) -> dict[str, int]:
        n_occ = sum(nelec_pair) // 2
        nelec_int = nelec_pair[0] + nelec_pair[1]

        if "params" not in cached:
            nmo = np.asarray(mf.mo_coeff).shape[1]
            nelecas = nelec_int
            ncore = (mf.mol.nelectron - nelecas) // 2
            active = set(range(ncore, ncore + norb))
            if 0 < n_occ < norb:
                # 与 examples/h2o_sqd.py 同一思路：冻结活性空间以外的轨道。
                ccsd = cc.CCSD(mf)
                ccsd.frozen = [i for i in range(nmo) if i not in active]
                _, t1, t2 = ccsd.kernel()
            else:
                t1 = np.zeros((n_occ, norb - n_occ))
                t2 = np.zeros((n_occ, n_occ, norb - n_occ, norb - n_occ))
            cached["params"] = initialize_lucj_parameters_from_ccsd(
                t2,
                t1=t1,
                n_spatial_orbitals=norb,
                n_layers=n_layers,
                topology=topology,
                optimize=optimize,
                maxiter=init_maxiter,
            )

        circuit = LUCJ(norb, nelec_int, n_layers, topology).get_circuit(cached["params"])
        n_qubits = 2 * norb

        if runtime == "device":
            # 设备档：补满 measure_z 后经 devices.base.run 提交，直接取回计数。
            # 与真机同一提交入口（切 provider/device 即可上真机）；噪声经 use_noise/noise
            # 透传给引擎（depolarizing 与 numeric 档同物理）。
            from tyxonq.devices import base as device_base

            for q in range(n_qubits):
                circuit.measure_z(q)
            run_opts: dict = {"use_noise": noise_p > 0.0}
            if noise_p > 0.0:
                run_opts["noise"] = {"type": "depolarizing", "p": float(noise_p)}
            tasks = device_base.run(
                provider=provider,
                device=device,
                circuit=circuit,
                shots=int(shots),
                **run_opts,
            )
            payload = tasks[0].get_result(wait=False)
            raw_counts = {str(bs): int(c) for bs, c in (payload.get("result") or {}).items()}
        else:
            # numeric 档：本地精确概率 + rng.choice 抽样（seed 可复现）。
            probabilities = np.asarray(StatevectorEngine().probability(circuit), dtype=float).reshape(-1)
            probabilities = probabilities / np.sum(probabilities)
            if noise_p > 0.0:
                alpha = min(1.0, 4.0 * float(noise_p) / 3.0)
                probabilities = (1.0 - alpha) * probabilities + alpha / probabilities.size
            rng = np.random.default_rng(seed)
            samples = rng.choice(probabilities.size, size=int(shots), p=probabilities)
            unique, counts = np.unique(samples, return_counts=True)
            raw_counts = {format(int(i), f"0{n_qubits}b"): int(c) for i, c in zip(unique, counts)}

        # 两档计数都是 TyxonQ/LUCJ raw order [alpha0.. | beta0..]（qubit0=MSB），每个自旋
        # 半区需反转成 SQD/PySCF order [..alpha0 | ..beta0] 后再交给 run_sqd_fermion，否则
        # bitstring_matrix_to_integers（MSB 优先）会把轨道序整体读反（HF 串 3 误读成 12）。
        # 与 examples/h2o_sqd.py 一致（反转是调用方责任，run_sqd_fermion 内部不做）。
        return {reverse_bitstring_halves(bs): int(c) for bs, c in raw_counts.items()}

    return sampler


def as_pyscf_solver(
    *,
    subspace: str = "frozen",
    ci_strs: tuple[Sequence[int], Sequence[int]] | SCIState | None = None,
    sampler: Callable | None = None,
    sampler_kwargs: dict | None = None,
    spin_sq: float | None = None,
    refresh_every: int = 1,
) -> SQDFCISolver:
    """创建 PySCF 兼容的 SQD 求解器。别名：``SQD.as_pyscf_solver(...)``。"""
    return SQDFCISolver(
        subspace=subspace,
        ci_strs=ci_strs,
        sampler=sampler,
        sampler_kwargs=sampler_kwargs,
        spin_sq=spin_sq,
        refresh_every=refresh_every,
    )
