from __future__ import annotations

from typing import List, Tuple, Sequence, Callable, Union

import numpy as np
from scipy.optimize import minimize

from ..runtimes.hea_device_runtime import HEADeviceRuntime
from tyxonq.libs.circuits_library.blocks import build_hwe_ry_ops
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import (
    get_hop_from_integral,
    get_integral_from_hf,
)
from tyxonq.libs.hamiltonian_encoding.fermion_to_qubit import fop_to_qop, parity, binary
from openfermion import QubitOperator, FermionOperator, hermitian_conjugated
from pyscf.scf import RHF  # type: ignore


Hamiltonian = List[Tuple[float, List[Tuple[str, int]]]]


class HEA:
    """Hardware-Efficient Ansatz (HEA) / 硬件高效参数化电路

    核心思路：以交替的单比特旋转与纠缠层（CNOT 链）构成参数化电路，用于 VQE 等变分算法。
    本实现采用 RY-only 结构：初始 RY 层 + L 层(纠缠 + RY)。层与层之间插入 barrier（IR 指令），
    便于可视化与编译边界控制。

    - 参数个数： (layers + 1) * n
    - 电路结构：
        L0:  逐比特 RY(θ0,i)
        对每层 l=1..L：CNOT 链 (0→1→...→n-1) + 逐比特 RY(θl,i)

    该类支持：
    - 从“哈密顿量项列表”（counts 评估路径）直接构建并在设备路径上进行能量与参数移位梯度评估；
    - 从分子积分/活性空间（PySCF）与费米子算符映射（parity/JW/BK）构建 HEA；
    - 与旧版 static/hea.py 的功能对应，但实现已迁移到 algorithms/runtimes/libs，移除张量网络依赖。
    """
    def __init__(self, n: int, layers: int, hamiltonian: Hamiltonian, engine: str = "device"):
        self.n = int(n)
        self.layers = int(layers)
        self.hamiltonian = list(hamiltonian)
        self.engine = engine
        # RY ansatz: (layers + 1) * n parameters
        self.n_params = (self.layers + 1) * self.n
        self.init_guess = np.zeros(self.n_params, dtype=np.float64)
        # Optional chemistry metadata (used by RDM与求解器适配)
        self.mapping: str | None = None
        self.int1e: np.ndarray | None = None
        self.int2e: np.ndarray | None = None
        self.n_elec: int | None = None
        self.spin: int | None = None
        self.e_core: float | None = None
        # Optimization artifacts
        self._grad: str = "param-shift"
        self.scipy_minimize_options: dict | None = None
        self._params: np.ndarray | None = None
        self.opt_res: dict | None = None

    def get_circuit(self, params: Sequence[float] | None = None):
        """构建 HEA 的门级电路（IR Circuit）。

        参数
        ----
        params: 序列
            长度为 (layers + 1) * n。若为空则使用 init_guess。

        返回
        ----
        Circuit
            包含初始 RY 层、每层的 CNOT 链与 RY，以及层间 barrier 指令。
        """
        if params is None:
            params = self.init_guess
        return build_hwe_ry_ops(self.n, self.layers, params)

    def energy(self, params: Sequence[float] | None = None, **device_opts) -> float:
        """基于计数的能量评估（设备路径）。

        内部对哈密顿量进行按基分组的测量流程：对每个基组应用相应的基变换（X→H，Y→S^†H），
        然后做 Z 基测量并从计数中估计 <H>。
        """
        if self.engine == "device":
            rt = HEADeviceRuntime(self.n, self.layers, self.hamiltonian)
            p = self.init_guess if params is None else params
            return rt.energy(p, **device_opts)
        raise NotImplementedError("numeric engines to be added")

    def energy_and_grad(self, params: Sequence[float] | None = None, **device_opts):
        """参数移位法梯度（设备路径）。

        对每个可移位参数 θ_k 使用标准移位 s=π/2 计算：
            g_k = 0.5 * (E(θ_k+s) - E(θ_k-s))
        与 energy 一样采用计数估计。
        """
        if self.engine == "device":
            rt = HEADeviceRuntime(self.n, self.layers, self.hamiltonian)
            p = self.init_guess if params is None else params
            return rt.energy_and_grad(p, **device_opts)
        raise NotImplementedError("numeric engines to be added")

    # ---------- Optimization (SciPy) ----------
    def get_opt_function(self, *, with_time: bool = False) -> Union[Callable, Tuple[Callable, float]]:
        """返回用于 SciPy 的目标函数封装。

        当 self.grad == "free" 时，仅返回能量函数；否则返回 (能量, 梯度)。
        """
        import time as _time

        def f_only(x: np.ndarray) -> float:
            return float(self.energy(x))

        def f_with_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            e, g = self.energy_and_grad(x)
            return float(e), np.asarray(g, dtype=np.float64)

        t1 = _time.time()
        func = f_only if self.grad == "free" else f_with_grad
        # 轻量“预热”，便于可能的 lazy 初始化
        _ = func(self.init_guess.copy())
        t2 = _time.time()
        if with_time:
            return func, (t2 - t1)
        return func

    def kernel(self) -> float:
        """运行变分优化，返回最优能量并保存 `opt_res` 与 `params`。"""
        func = self.get_opt_function()
        if self.grad == "free":
            res = minimize(lambda x: func(x), x0=self.init_guess, jac=False, method="COBYLA", options=self.scipy_minimize_options)
        else:
            res = minimize(lambda x: func(x)[0], x0=self.init_guess, jac=lambda x: func(x)[1], method="L-BFGS-B", options=self.scipy_minimize_options)
        self.opt_res = {
            "success": bool(getattr(res, "success", True)),
            "x": np.asarray(getattr(res, "x", self.init_guess), dtype=np.float64),
            "fun": float(getattr(res, "fun", self.energy(self.init_guess))),
            "message": str(getattr(res, "message", "")),
            "nit": int(getattr(res, "nit", 0)),
        }
        self.params = np.asarray(self.opt_res["x"], dtype=np.float64).copy()
        return float(self.opt_res["fun"])  # type: ignore[index]

    # ---------- Builders from chemistry inputs ----------
    @staticmethod
    def _qop_to_term_list(qop: QubitOperator, n_qubits: int) -> Hamiltonian:
        terms: Hamiltonian = []
        # identity term
        const = qop.terms.get((), 0.0)
        if const != 0.0:
            coeff_val = float(getattr(const, "real", float(const)))
            terms.append((coeff_val, []))
        for term, coeff in qop.terms.items():
            if term == ():
                continue
            ops: List[Tuple[str, int]] = []
            for (idx, sym) in term:
                ops.append((sym.upper(), int(idx)))
            terms.append((float(getattr(coeff, "real", float(coeff))), ops))
        return terms

    @classmethod
    def from_integral(
        cls,
        int1e: np.ndarray,
        int2e: np.ndarray,
        n_elec: Union[int, Tuple[int, int]],
        e_core: float,
        *,
        n_layers: int = 1,
        mapping: str = "parity",
        engine: str = "device",
    ) -> "HEA":
        """从分子积分构建 HEA。

        流程：
        1) 由 (int1e, int2e) 构造费米子算符 H_f；
        2) 按映射（parity/JW/BK）将 H_f → QubitOperator；
        3) 转为轻量哈密顿量列表 [(coeff, [(P, q), ...]) ...]，加上 e_core；
        4) 以 n_qubits = n_sorb or n_sorb-2（parity 两比特节省）实例化 HEA。
        """
        n_sorb = 2 * len(int1e)
        if isinstance(n_elec, int):
            if n_elec % 2 != 0:
                raise ValueError("Odd total electrons: pass (na, nb) tuple instead")
            n_elec_s = (n_elec // 2, n_elec // 2)
        else:
            n_elec_s = n_elec
        fop = get_hop_from_integral(int1e, int2e)
        qop = fop_to_qop(fop, mapping, n_sorb, n_elec_s)
        terms = cls._qop_to_term_list(qop, n_qubits=(n_sorb - 2 if mapping == "parity" else n_sorb))
        if e_core and abs(e_core) > 0:
            terms = [(float(e_core), [])] + terms
        n_qubits = (n_sorb - 2) if mapping == "parity" else n_sorb
        inst = cls(n=n_qubits, layers=int(n_layers), hamiltonian=terms, engine=engine)
        # record chemistry metadata for downstream features (RDM等)
        inst.mapping = str(mapping)
        inst.int1e = np.array(int1e)
        inst.int2e = np.array(int2e)
        inst.n_elec = int(sum(n_elec_s))
        inst.spin = int(abs(n_elec_s[0] - n_elec_s[1]))
        inst.e_core = float(e_core)
        return inst

    @classmethod
    def from_molecule(
        cls,
        m,
        *,
        active_space=None,
        n_layers: int = 1,
        mapping: str = "parity",
        engine: str = "device",
    ) -> "HEA":
        """从 PySCF 分子对象构建 HEA。

        - 自动运行 RHF 得到积分 (int1e, int2e) 与 e_core；
        - 根据分子总电子数与自旋计算 (n_alpha, n_beta)；
        - 复用 from_integral 流程完成映射与实例化。
        """
        hf = RHF(m)
        # avoid serialization warnings in some envs
        hf.chkfile = None
        hf.verbose = 0
        hf.kernel()
        int1e, int2e, e_core = get_integral_from_hf(hf, active_space=active_space)
        # derive (na, nb) from m
        if hasattr(m, "nelectron"):
            tot = int(getattr(m, "nelectron"))
        else:
            tot = int(getattr(m, "n_elec", 0))
        if hasattr(m, "spin"):
            spin = int(getattr(m, "spin"))
        else:
            spin = 0
        na = (tot + spin) // 2
        nb = (tot - spin) // 2
        inst = cls.from_integral(int1e, int2e, (na, nb), e_core, n_layers=n_layers, mapping=mapping, engine=engine)
        return inst

    @classmethod
    def ry(
        cls,
        int1e: np.ndarray,
        int2e: np.ndarray,
        n_elec: Union[int, Tuple[int, int]],
        e_core: float,
        n_layers: int,
        *,
        mapping: str = "parity",
        engine: str = "device",
    ) -> "HEA":
        """兼容旧接口的 RY 构造器（等价于 from_integral(..., n_layers, mapping, engine)）。"""
        return cls.from_integral(int1e, int2e, n_elec, e_core, n_layers=n_layers, mapping=mapping, engine=engine)

    # ---------- PySCF solver 适配（可选） ----------
    @classmethod
    def as_pyscf_solver(cls, *, n_layers: int = 1, mapping: str = "parity", engine: str = "device", config_function: Callable | None = None):
        """返回一个最小 PySCF FCI 求解器兼容对象，内部以 HEA 优化。

        仅实现 kernel/make_rdm1/make_rdm2 所需的最小接口，便于与 CASSCF 对接。
        """

        class _FCISolver:
            def __init__(self):
                self.instance: HEA | None = None

            def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
                self.instance = cls.from_integral(h1, h2, nelec, ecore, n_layers=n_layers, mapping=mapping, engine=engine)
                if config_function is not None:
                    config_function(self.instance)
                e = self.instance.kernel()
                return float(e), self.instance.params

            def make_rdm1(self, params, norb, nelec):
                assert self.instance is not None
                return self.instance.make_rdm1(params)

            def make_rdm12(self, params, norb, nelec):
                assert self.instance is not None
                rdm1 = self.instance.make_rdm1(params)
                rdm2 = self.instance.make_rdm2(params)
                return rdm1, rdm2

            def spin_square(self, params, norb, nelec):
                return 0.0, 1.0

        return _FCISolver()

    # ---------- RDM（基于 counts 的期望估计） ----------
    @property
    def n_elec_s(self) -> Tuple[int, int] | None:
        if self.n_elec is None or self.spin is None:
            return None
        na = (self.n_elec + self.spin) // 2
        nb = (self.n_elec - self.spin) // 2
        return int(na), int(nb)

    def _expect_qubit_operator(self, qop: QubitOperator, params: Sequence[float]) -> float:
        terms = self._qop_to_term_list(qop, n_qubits=self.n)
        rt = HEADeviceRuntime(self.n, self.layers, terms)
        return float(rt.energy(params))

    def make_rdm1(self, params: Sequence[float] | None = None) -> np.ndarray:
        """计算自旋约化的一体 RDM（spin-traced 1RDM）。需要在 from_integral/from_molecule 构建后使用。"""
        if params is None:
            params = self.init_guess
        if self.mapping is None or self.n_elec_s is None:
            raise ValueError("RDM 需要在 from_integral/from_molecule 构建并携带 mapping 与电子数信息")
        mapping = str(self.mapping)
        if mapping == "parity":
            n_sorb = self.n + 2
        else:
            n_sorb = self.n
        n_orb = n_sorb // 2
        rdm1 = np.zeros((n_orb, n_orb), dtype=np.float64)
        for i in range(n_orb):
            for j in range(i + 1):
                if int(self.spin or 0) == 0:
                    fop = 2 * FermionOperator(f"{i}^ {j}")
                else:
                    fop = FermionOperator(f"{i}^ {j}") + FermionOperator(f"{i+n_orb}^ {j+n_orb}")
                fop = fop + hermitian_conjugated(fop)
                qop = fop_to_qop(fop, mapping, n_sorb, self.n_elec_s)
                val = 0.5 * self._expect_qubit_operator(qop, params)
                rdm1[i, j] = rdm1[j, i] = float(val)
        return rdm1

    def make_rdm2(self, params: Sequence[float] | None = None) -> np.ndarray:
        """计算自旋约化的二体 RDM（spin-traced 2RDM）。需要在 from_integral/from_molecule 构建后使用。"""
        if params is None:
            params = self.init_guess
        if self.mapping is None or self.n_elec_s is None:
            raise ValueError("RDM 需要在 from_integral/from_molecule 构建并携带 mapping 与电子数信息")
        mapping = str(self.mapping)
        if mapping == "parity":
            n_sorb = self.n + 2
        else:
            n_sorb = self.n
        n_orb = n_sorb // 2
        rdm2 = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=np.float64)
        calculated: set[Tuple[int, int, int, int]] = set()
        for p in range(n_orb):
            for q in range(n_orb):
                for r in range(n_orb):
                    for s in range(n_orb):
                        if (p, q, r, s) in calculated:
                            continue
                        fop_aaaa = FermionOperator(f"{p}^ {q}^ {r} {s}")
                        fop_abba = FermionOperator(f"{p}^ {q+n_orb}^ {r+n_orb} {s}")
                        if int(self.spin or 0) == 0:
                            fop = 2 * (fop_aaaa + fop_abba)
                        else:
                            fop_bbbb = FermionOperator(f"{p+n_orb}^ {q+n_orb}^ {r+n_orb} {s+n_orb}")
                            fop_baab = FermionOperator(f"{p+n_orb}^ {q}^ {r} {s+n_orb}")
                            fop = fop_aaaa + fop_abba + fop_bbbb + fop_baab
                        fop = fop + hermitian_conjugated(fop)
                        qop = fop_to_qop(fop, mapping, n_sorb, self.n_elec_s)
                        val = 0.5 * self._expect_qubit_operator(qop, params)
                        idxs = [(p, q, r, s), (s, r, q, p), (q, p, s, r), (r, s, p, q)]
                        for idx in idxs:
                            rdm2[idx] = float(val)
                            calculated.add(idx)
        # 转置到 PySCF 约定：rdm2[p,q,r,s] = <p^+ r^+ s q>
        rdm2 = rdm2.transpose(0, 3, 1, 2)
        return rdm2

    # ---------- 打印辅助 ----------
    def print_circuit(self):
        from tyxonq.libs.circuits_library.analysis import get_circuit_summary
        c = self.get_circuit(self.init_guess)
        summary = get_circuit_summary(c)
        try:
            # 若是 DataFrame-like
            print(summary.to_string(index=False))  # type: ignore[attr-defined]
        except Exception:
            print(summary)

    def print_summary(self):
        print("############################### Circuit ###############################")
        self.print_circuit()
        print("######################### Optimization Result #########################")
        if self.opt_res is None:
            print("Optimization not run yet")
        else:
            print(self.opt_res)

    # ---------- properties ----------
    @property
    def grad(self) -> str:
        return self._grad

    @grad.setter
    def grad(self, v: str) -> None:
        if v not in ("param-shift", "free"):
            raise ValueError(f"Invalid gradient method {v}")
        self._grad = v

    @property
    def params(self) -> np.ndarray | None:
        return self._params

    @params.setter
    def params(self, p: Sequence[float]) -> None:
        self._params = np.asarray(p, dtype=np.float64)


# Re-exports for legacy imports convenience
__all__ = [
    "HEA",
    "parity",
    "binary",
]

