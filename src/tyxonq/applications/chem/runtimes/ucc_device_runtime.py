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
 
from tyxonq.postprocessing import apply_postprocessing
from tyxonq.devices import base as device_base


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
        trotter: bool = False
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

        # ---- Precompute grouping and cache measurement prefixes ----
        # Group once per runtime instance; reuse across energy/grad evaluations
        identity_const, groups = group_qubit_operator_terms(self.h_qubit_op, self.n_qubits)
        self._identity_const: float = float(identity_const)
        self._groups = groups  # Dict[Tuple[str,...], List[(items)]]
        self._prefix_cache: Dict[Tuple[str, ...], List[Tuple]] = {}

    def _prefix_ops_for_bases(self, bases: Tuple[str, ...]) -> List[Tuple]:
        if bases in self._prefix_cache:
            return self._prefix_cache[bases]
        ops: List[Tuple] = []
        # Map OpenFermion little-endian bases (q=0 is LSB) to IR qubit indices by bit-reversal
        n = self.n_qubits
        for q, p in enumerate(bases):
            qq = n - 1 - q
            if p == "X":
                ops.append(("h", qq))
            elif p == "Y":
                ops.append(("rz", qq, -pi/2)); ops.append(("h", qq))
        # Measure all qubits in the mapped order
        for q in range(n):
            ops.append(("measure_z", n - 1 - q))
        self._prefix_cache[bases] = ops
        return ops

        # TODO (device-runtime backlog):
        # 1) Add adjoint differentiation for simulator shots=0 to replace finite-difference (faster, exact on statevector)
        # 2) Support SPSA/gradient-free optimizers for hardware shots>0 to reduce evaluations
        # 3) Batch/parallel parameter shifts and group evaluations; reuse compiled prefixes/suffixes
        # 4) Adaptive shots allocation per parameter/group based on variance/sensitivity
        # 5) Optional low-rank/commuting-group Hamiltonian transforms to reduce measurement cost
        # 6) Caching of expectation terms across close parameters during local line-search

    def _build_hf_circuit(self) -> Circuit:
        # c = Circuit(self.n_qubits, ops=[])
        # if self.mode in ["fermion", "qubit"]:
        #     na, nb = self.n_elec_s
        #     # 与 circuits_library.ucc._hf_init_ops 保持一致：先置 alpha（高位半区），再置 beta（低位半区）
        #     for i in range(na):
        #         c.ops.append(("x", self.n_qubits - 1 - i))
        #     for i in range(nb):
        #         c.ops.append(("x", self.n_qubits // 2 - 1 - i))
        # else:
        #     na = sum(self.n_elec_s) // 2
        #     for i in range(na):
        #         c.ops.append(("x", self.n_qubits - 1 - i))
        # return c
        n = int(self.n_qubits)
        c = Circuit(n, ops=[])
        if isinstance(self.n_elec_s, (tuple, list)):
            na = int(self.n_elec_s[0])
            nb = int(self.n_elec_s[1])
        else:
            ne = int(self.n_elec_s)
            na = nb = ne // 2
        if self.mode in ("fermion", "qubit"):
            for i in range(nb):
                c.X(n - 1 - i)
            for i in range(na):
                c.X(n // 2 - 1 - i)
        else:
            assert self.mode == "hcb"
            for i in range(na):
                c.X(n - 1 - i)
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
        # Use cached grouping and measurement prefixes
        energy_val = float(self._identity_const)
        # 仅构建一次基电路；将各分组电路合并为一批提交
        base_circuit = c_builder()
        circuits: List[Circuit] = []
        items_by_idx: List[List[Tuple]] = []  # type: ignore[type-arg]
        for bases, items in self._groups.items():
            circuits.append(base_circuit.extended(self._prefix_ops_for_bases(bases)))
            items_by_idx.append(items)

        # 设备层批量执行，保持顺序一致
        tasks = device_base.run(provider=provider, device=device, circuit=circuits, shots=shots, noise=noise, **device_kwargs)

        # 手动应用与原先一致的后处理，累加每组能量
        for k, t in enumerate(tasks):
            rr = t.get_result(wait=False)
            pp_opts = dict(postprocessing or {})
            pp_opts.update({
                "method": "expval_pauli_sum",
                "identity_const": 0.0,
                "items": items_by_idx[k],
            })
            post = apply_postprocessing(rr, pp_opts)
            payload = post.get("result", {})
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
        gradient_method: str = "fd",
        **device_kwargs,
    ) -> Tuple[float, np.ndarray]:
        if self.n_params == 0:
            e0 = self.energy(None, shots=shots, provider=provider, device=device, postprocessing=postprocessing)
            return e0, np.zeros(0, dtype=np.float64)

        base = np.asarray(params if params is not None else np.zeros(self.n_params, dtype=np.float64), dtype=np.float64)
        def _builder_with(pv: np.ndarray):
            def _b():
                return self._build_ucc_circuit(pv)
            return _b

        # ---- Batch energy and gradients in one device submission ----
        # Prepare circuits for base and all ± shifts across all groups
        groups_seq = list(self._groups.items())
        circuits_all: List[Circuit] = []
        items_by_circuit: List[List[Tuple]] = []  # type: ignore[type-arg]
        tags: List[Tuple[str, int]] = []  # (variant, param_index)

        def _append_variant(pvec: np.ndarray, tag: Tuple[str, int]):
            c0 = self._build_ucc_circuit(pvec)
            for bases, items in groups_seq:
                circuits_all.append(c0.extended(self._prefix_ops_for_bases(bases)))
                items_by_circuit.append(items)
                tags.append(tag)

        # base
        _append_variant(base, ("base", -1))

        method = str(gradient_method).lower()
        if method == "fd":
            shift = float(np.pi / 2.0)
            for i in range(len(base)):
                p_plus = base.copy(); p_plus[i] += shift
                p_minus = base.copy(); p_minus[i] -= shift
                _append_variant(p_plus, ("plus", i))
                _append_variant(p_minus, ("minus", i))
        else:
            step = float(np.pi / 90.0)
            for i in range(len(base)):
                p_plus = base.copy(); p_plus[i] += step
                p_minus = base.copy(); p_minus[i] -= step
                _append_variant(p_plus, ("plus_s", i))
                _append_variant(p_minus, ("minus_s", i))

        tasks = device_base.run(provider=provider, device=device, circuit=circuits_all, shots=shots, noise=noise, **device_kwargs)

        e0 = float(self._identity_const)
        n_params = len(base)
        plus_energy = np.zeros(n_params, dtype=np.float64)
        minus_energy = np.zeros(n_params, dtype=np.float64)

        for k, t in enumerate(tasks):
            rr = t.get_result(wait=False)
            pp_opts = dict(postprocessing or {})
            pp_opts.update({
                "method": "expval_pauli_sum",
                "identity_const": 0.0,
                "items": items_by_circuit[k],
            })
            post = apply_postprocessing(rr, pp_opts)
            e = float((post.get("result", {}) or {}).get("energy", 0.0))
            tag, idx = tags[k]
            if tag == "base":
                e0 += e
            elif tag.startswith("plus"):
                if 0 <= idx < n_params:
                    plus_energy[idx] += e
            elif tag.startswith("minus"):
                if 0 <= idx < n_params:
                    minus_energy[idx] += e

        g = np.zeros_like(base)
        if method == "fd":
            for i in range(n_params):
                g[i] = 0.5 * (plus_energy[i] - minus_energy[i])
            return float(e0), g
        else:
            step = float(np.pi / 90.0)
            for i in range(n_params):
                g[i] = (plus_energy[i] - minus_energy[i]) / (2.0 * step)
            return float(e0), g

