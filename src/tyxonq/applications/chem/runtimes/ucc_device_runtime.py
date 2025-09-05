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

    def _build_hf_circuit(self) -> Circuit:
        c = Circuit(self.n_qubits, ops=[])
        if self.mode in ["fermion", "qubit"]:
            na, nb = self.n_elec_s
            for i in range(nb):
                c.ops.append(("x", self.n_qubits - 1 - i))
            for i in range(na):
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
            dev = c.device(provider=provider, device=device, shots=shots)
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
        shots: int = 8192,
        provider: str = "simulator",
        device: str = "statevector",
        postprocessing: dict | None = None,
    ) -> float:
        if self.n_params == 0:
            def _builder():
                return self._build_hf_circuit()
        else:
            if params is None:
                params = np.zeros(self.n_params, dtype=np.float64)
            p = np.asarray(params, dtype=np.float64)
            def _builder():
                return self._build_ucc_circuit(p)
        return self._energy_core(_builder, shots=shots, provider=provider, device=device, postprocessing=postprocessing)

    def energy_and_grad(
        self,
        params: Sequence[float] | None = None,
        *,
        shots: int = 8192,
        provider: str = "simulator",
        device: str = "statevector",
        postprocessing: dict | None = None,
    ) -> Tuple[float, np.ndarray]:
        if self.n_params == 0:
            e0 = self.energy(None, shots=shots, provider=provider, device=device, postprocessing=postprocessing)
            return e0, np.zeros(0, dtype=np.float64)

        base = np.asarray(params if params is not None else np.zeros(self.n_params, dtype=np.float64), dtype=np.float64)
        def _builder_with(pv: np.ndarray):
            def _b():
                return self._build_ucc_circuit(pv)
            return _b

        e0 = self._energy_core(_builder_with(base), shots=shots, provider=provider, device=device, postprocessing=postprocessing)
        g = np.zeros_like(base)
        s = 0.5 * pi
        for i in range(len(base)):
            p_plus = base.copy(); p_plus[i] += s
            p_minus = base.copy(); p_minus[i] -= s
            e_plus = self._energy_core(_builder_with(p_plus), shots=shots, provider=provider, device=device, postprocessing=postprocessing)
            e_minus = self._energy_core(_builder_with(p_minus), shots=shots, provider=provider, device=device, postprocessing=postprocessing)
            g[i] = 0.5 * (e_plus - e_minus)
        return float(e0), g

