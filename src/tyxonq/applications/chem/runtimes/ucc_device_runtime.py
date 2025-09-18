from __future__ import annotations

from typing import List, Tuple, Dict, Sequence
from math import pi

import numpy as np
from openfermion import QubitOperator

from tyxonq.core.ir.circuit import Circuit
from tyxonq.compiler.api import compile as compile_api
from tyxonq.compiler.utils.hamiltonian_grouping import (
    group_qubit_operator_terms,
)
 


try:
    # 统一从 circuits_library 抽象构造 UCC 电路
    from tyxonq.libs.circuits_library.ucc import build_ucc_circuit  # type: ignore
except Exception:
    build_ucc_circuit = None  # type: ignore


class UCCDeviceRuntime:
    """Device runtime for UCC energy/gradient via counts with parameterized ansatz.

    - 支持 HF 初态 + 可配置 excitation/param_ids 的参数化线路
    - 通过参数移位（π/2）计算梯度
    """

    def __init__(
        self,
        n_qubits: int,
        n_elec_s: Tuple[int, int],
        h_qubit_op: QubitOperator,
        mode: str = "fermion",
        *,
        ex_ops: List[Tuple] | None = None,
        param_ids: List[int] | None = None,
        init_state: Sequence[float] | None = None,
        decompose_multicontrol: bool = False,
        trotter: bool = False,
    ):
        self.n_qubits = int(n_qubits)
        self.n_elec_s = (int(n_elec_s[0]), int(n_elec_s[1]))
        self.mode = str(mode)
        self.h_qubit_op = h_qubit_op

        self.ex_ops = list(ex_ops) if ex_ops is not None else None
        self.param_ids = list(param_ids) if param_ids is not None else None
        self.init_state = init_state
        self.decompose_multicontrol = bool(decompose_multicontrol)
        self.trotter = bool(trotter)

        # 推断参数个数
        if self.ex_ops is not None:
            if self.param_ids is None:
                self.n_params = len(self.ex_ops)
            else:
                self.n_params = max(self.param_ids) + 1 if len(self.param_ids) > 0 else 0
        else:
            self.n_params = 0

        # TODO (device-runtime backlog):
        # 1) Add adjoint differentiation for simulator shots=0 to replace finite-difference (faster, exact on statevector)
        # 2) Support SPSA/gradient-free optimizers for hardware shots>0 to reduce evaluations
        # 3) Batch/parallel parameter shifts and group evaluations; reuse compiled prefixes/suffixes
        # 4) Adaptive shots allocation per parameter/group based on variance/sensitivity
        # 5) Optional low-rank/commuting-group Hamiltonian transforms to reduce measurement cost
        # 6) Caching of expectation terms across close parameters during local line-search

    def _build_hf_circuit(self) -> Circuit:
        c = Circuit(self.n_qubits, ops=[])
        if self.mode in ["fermion", "qubit"]:
            na, nb = self.n_elec_s
            # 与 circuits_library.ucc._hf_init_ops 保持一致：先置 alpha（高位半区），再置 beta（低位半区）
            for i in range(na):
                c.ops.append(("x", self.n_qubits - 1 - i))
            for i in range(nb):
                c.ops.append(("x", self.n_qubits // 2 - 1 - i))
        else:
            na = sum(self.n_elec_s) // 2
            for i in range(na):
                c.ops.append(("x", self.n_qubits - 1 - i))
        return c

    def _build_ucc_circuit(self, params: Sequence[float]) -> Circuit:
        if self.ex_ops is None or self.n_params == 0 or build_ucc_circuit is None:
            return self._build_hf_circuit()
        if len(params) != self.n_params:
            raise ValueError(f"params length {len(params)} != {self.n_params}")
        c = build_ucc_circuit(
            params,
            self.n_qubits,
            self.n_elec_s,
            tuple(self.ex_ops),
            tuple(self.param_ids) if self.param_ids is not None else None,
            mode=self.mode,
            # 设备路径不消费 init_state（无论 ndarray 还是 Circuit）
            init_state=None,
            decompose_multicontrol=self.decompose_multicontrol,
            trotter=self.trotter,
        )
        # build_ucc_circuit 返回 Circuit
        return c

    def _energy_core(
        self,
        c_builder,
        *,
        shots: int,
        provider: str,
        device: str,
        postprocessing: dict | None,
        noise: dict | None = None,
        **device_kwargs,
    ) -> float:
        # Prefer compiler-provided grouping when available
        # Fallback to internal grouping during transition
        identity_const, groups = group_qubit_operator_terms(self.h_qubit_op, self.n_qubits)
        energy_val = float(identity_const)
        for bases, items in groups.items():
            c = c_builder()
            for q, p in enumerate(bases):
                if p == "X":
                    c.ops.append(("h", q))
                elif p == "Y":
                    c.ops.append(("rz", q, -pi/2)); c.ops.append(("h", q))
            for q in range(self.n_qubits):
                c.ops.append(("measure_z", q))
            dev = c.device(provider=provider, device=device, shots=shots, noise=noise, **device_kwargs)
            # Chainable postprocessing: aggregate energy for this group's items
            pp_opts = dict(postprocessing or {})
            pp_opts.update({
                "method": "expval_pauli_sum",
                "identity_const": 0.0,
                "items": items,
            })
            dev = dev.postprocessing(**pp_opts)
            res = dev.run()
            payload = res[0]["postprocessing"]["result"] if isinstance(res, list) else (res.get("postprocessing", {}) or {}).get("result", {})
            # counts_expval.expval_pauli_sum returns dict with 'energy'
            energy_val += float((payload or {}).get("energy", 0.0))
        return float(energy_val)

    def energy(
        self,
        params: Sequence[float] | None = None,
        *,
        shots: int = 1024,
        provider: str = "simulator",
        device: str = "statevector",
        postprocessing: dict | None = None,
        noise: dict | None = None,
        **device_kwargs,
    ) -> float:
        # Fast path: simulator/local + shots==0 → 使用解析期望；返回电子能（去掉常数项）
        if (provider in ("simulator", "local")) and int(shots) == 0:
            p = np.asarray(
                params
                if params is not None
                else (np.zeros(self.n_params, dtype=np.float64) if self.n_params > 0 else np.zeros(0, dtype=np.float64)),
                dtype=np.float64,
            )
            # 数值库直算电子能（与 numeric/statevector 完全一致）
            from tyxonq.applications.chem.chem_libs.quantum_chem_library.statevector_ops import (
                energy_statevector as _energy_statevector,
            )
            e_elec = _energy_statevector(
                p,
                self.h_qubit_op,
                self.n_qubits,
                self.n_elec_s,
                self.ex_ops,
                self.param_ids,
                mode=self.mode,
                init_state=None,
            )
            return float(e_elec)

        if self.n_params == 0:
            def _builder():
                return self._build_hf_circuit()
        else:
            if params is None:
                params = np.zeros(self.n_params, dtype=np.float64)
            p = np.asarray(params, dtype=np.float64)
            def _builder():
                return self._build_ucc_circuit(p)
        return self._energy_core(_builder, shots=shots, provider=provider, device=device, postprocessing=postprocessing, noise=noise, **device_kwargs)

    def energy_and_grad(
        self,
        params: Sequence[float] | None = None,
        *,
        shots: int = 1024,
        provider: str = "simulator",
        device: str = "statevector",
        postprocessing: dict | None = None,
        noise: dict | None = None,
        **device_kwargs,
    ) -> Tuple[float, np.ndarray]:
        # shots 路径统一由 driver+engine 归一；shots=0 时通过 device 层解析通道计算梯度
        if (provider in ("simulator", "local")) and int(shots) == 0:
            if self.n_params == 0:
                e0 = self.energy(None, shots=0, provider=provider, device=device, postprocessing=postprocessing)
                return float(e0), np.zeros(0, dtype=np.float64)
            x = np.asarray(
                params if params is not None else np.zeros(self.n_params, dtype=np.float64), dtype=np.float64
            )
            # 数值库直算（value_and_grad 等价路径），返回电子能与梯度
            from tyxonq.applications.chem.chem_libs.quantum_chem_library.statevector_ops import (
                energy_and_grad_statevector as _energy_and_grad_statevector,
            )
            e_elec, g = _energy_and_grad_statevector(
                x,
                self.h_qubit_op,
                self.n_qubits,
                self.n_elec_s,
                self.ex_ops,
                self.param_ids,
                mode=self.mode,
                init_state=None,
            )
            return float(e_elec), np.asarray(g, dtype=np.float64)

        if self.n_params == 0:
            e0 = self.energy(None, shots=shots, provider=provider, device=device, postprocessing=postprocessing)
            return e0, np.zeros(0, dtype=np.float64)

        base = np.asarray(params if params is not None else np.zeros(self.n_params, dtype=np.float64), dtype=np.float64)
        def _builder_with(pv: np.ndarray):
            def _b():
                return self._build_ucc_circuit(pv)
            return _b

        e0 = self._energy_core(_builder_with(base), shots=shots, provider=provider, device=device, postprocessing=postprocessing, noise=noise, **device_kwargs)
        g = np.zeros_like(base)
        # 对于 shots>0 的硬件/采样路径，使用对称有限差分以避免参数移位缩放不一致问题
        # ~2°，数值稳定且差分信号明显
        # Gradient strategy for counts (shots > 0): finite-difference vs parameter-shift
        #
        # - Finite-difference (FD):
        #   Estimates dE/dθ ≈ [E(θ+δ) − E(θ−δ)] / (2δ). It works for any circuit and observable,
        #   requires no gate-level derivative rules, but introduces truncation bias O(δ^2) and is
        #   sensitive to sampling noise (variance roughly scales like 1/(δ^2·shots)).
        #
        # - Parameter-shift (PS):
        #   Provides an exact analytic gradient for gates with suitable generators (e.g., ±1 spectra),
        #   using two evaluations with ±s shifts (commonly s = π/2). PS has no truncation bias but still
        #   carries sampling variance and requires compiling/executing the shifted circuits on hardware.
        #
        # - Why FD here for the counts path:
        #   Our counts path measures grouped, basis-rotated Pauli sums. A naive PS over circuit parameters
        #   can lead to scaling/mismatch with grouping unless a full compiler-level gradient transform is applied
        #   (to batch and share shifts safely across measurement groups). Until that pass exists, FD is consistent
        #   with the exact energy functional we are sampling. The chosen step (~2 degrees) balances truncation bias
        #   against shot noise; increasing shots reduces variance, while decreasing step reduces bias.
        #
        # - Simulators (shots == 0): handled elsewhere via an analytic numeric path (value_and_grad), not this branch.
        step = float(np.pi / 90.0)  # ~2 degrees: numerically stable for counts
        for i in range(len(base)):
            p_plus = base.copy(); p_plus[i] += step
            p_minus = base.copy(); p_minus[i] -= step
            e_plus = self._energy_core(_builder_with(p_plus), shots=shots, provider=provider, device=device, postprocessing=postprocessing, noise=noise, **device_kwargs)
            e_minus = self._energy_core(_builder_with(p_minus), shots=shots, provider=provider, device=device, postprocessing=postprocessing, noise=noise, **device_kwargs)
            g[i] = (e_plus - e_minus) / (2.0 * step)
        return float(e0), g

