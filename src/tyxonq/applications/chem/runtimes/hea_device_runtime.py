from __future__ import annotations

from typing import List, Tuple, Dict, Sequence
from math import pi

import numpy as np

from tyxonq.core.ir.circuit import Circuit
from tyxonq.libs.circuits_library.blocks import build_hwe_ry_ops
from tyxonq.libs.circuits_library.qiskit_real_amplitudes import build_circuit_from_template
from tyxonq.compiler.utils.hamiltonian_grouping import (
    group_hamiltonian_pauli_terms,
)
from openfermion import QubitOperator
from openfermion.linalg import get_sparse_operator
from tyxonq.postprocessing import apply_postprocessing
from tyxonq.devices import base as device_base


Hamiltonian = List[Tuple[float, List[Tuple[str, int]]]]


class HEADeviceRuntime:
    def __init__(self, n: int, layers: int, hamiltonian: Hamiltonian, *, n_elec_s: Tuple[int, int] | None = None, mapping: str | None = None, circuit_template: list | None = None, qop: QubitOperator | None = None):
        self.n = int(n)
        self.layers = int(layers)
        self.hamiltonian = list(hamiltonian)
        self.n_elec_s = n_elec_s
        self.mapping = mapping
        self.circuit_template = circuit_template
        # RY ansatz uses (layers + 1) * n parameters
        self.n_params = (self.layers + 1) * self.n
        rng = np.random.default_rng(7)
        self.init_guess = rng.random(self.n_params, dtype=np.float64)

        # Pre-group Hamiltonian and cache measurement prefixes
        identity_const, groups = group_hamiltonian_pauli_terms(self.hamiltonian, self.n)
        self._identity_const: float = float(identity_const)
        self._groups = groups
        self._prefix_cache: Dict[Tuple[str, ...], List[Tuple]] = {}
        
        # 可选：直接消费上游映射缓存的 QubitOperator（shots==0 快径）
        self._qop_cached = qop

    def _build_circuit(self, params: Sequence[float]) -> Circuit:
        # If external template exists, instantiate from it
        if self.circuit_template is not None:
            return build_circuit_from_template(self.circuit_template, np.asarray(params, dtype=np.float64), n_qubits=self.n)
        # Default: RY-only ansatz
        return build_hwe_ry_ops(self.n, self.layers, params)

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
        if params is None:
            params = self.init_guess
        # If using template, parameter length is defined by template; skip RY param-length check
        if self.circuit_template is None and len(params) != self.n_params:
            raise ValueError(f"params length {len(params)} != {self.n_params}")

        # Fast analytic path: shots==0 → single statevector + full-H expectation (no grouping)
        # if (provider in ("simulator", "local")) and int(shots) == 0:
        #     from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
        #     from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import pauli_sum_to_qubit_operator
        #     from tyxonq.applications.chem.chem_libs.quantum_chem_library.statevector_ops import energy_from_statevector
        #     c = self._build_circuit(params)
        #     eng = StatevectorEngine()
        #     psi = eng.state(c)
        #     qop = self._qop_cached if self._qop_cached is not None else pauli_sum_to_qubit_operator(self.hamiltonian, self.n)
        #     return float(energy_from_statevector(psi, qop, self.n))

        # Use cached grouping and prefixes for shots>0
        energy_val = self._identity_const
        # 仅构建一次基电路；将各分组电路合并为一批提交
        base_circuit = self._build_circuit(params)
        circuits: List[Circuit] = []
        items_by_idx: List[List[Tuple]] = []  # type: ignore[type-arg]
        for bases, items in self._groups.items():
            circuits.append(base_circuit.extended(self._prefix_ops_for_bases(bases)))
            items_by_idx.append(items)

        tasks = device_base.run(provider=provider, device=device, circuit=circuits, shots=shots, noise=noise, **device_kwargs)
        for k, t in enumerate(tasks):
            rr = t.get_result(wait=False)
            pp_opts = dict(postprocessing or {})
            pp_opts.update({"method": "expval_pauli_sum", "identity_const": 0.0, "items": items_by_idx[k]})
            post = apply_postprocessing(rr, pp_opts)
            payload = post.get("result", {})
            energy_val += float((payload or {}).get("energy", 0.0))
        return float(energy_val)

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
        if params is None:
            params = self.init_guess
        base = np.asarray(params, dtype=np.float64)

        # shots==0: use parameter-shift (s=pi/2) over analytic energy path
        # if (provider in ("simulator", "local")) and int(shots) == 0:
        #     e0 = self.energy(base, shots=0, provider=provider, device=device, postprocessing=postprocessing, noise=noise, **device_kwargs)
        #     if len(base) == 0:
        #         return float(e0), np.zeros(0, dtype=np.float64)
        #     g = np.zeros_like(base)
        #     s = 0.5 * pi
        #     for i in range(len(base)):
        #         p_plus = base.copy(); p_plus[i] += s
        #         p_minus = base.copy(); p_minus[i] -= s
        #         e_plus = self.energy(p_plus, shots=0, provider=provider, device=device, postprocessing=postprocessing, noise=noise, **device_kwargs)
        #         e_minus = self.energy(p_minus, shots=0, provider=provider, device=device, postprocessing=postprocessing, noise=noise, **device_kwargs)
        #         g[i] = 0.5 * (e_plus - e_minus)
        #     return float(e0), g

        # ---- Batch base energy and parameter-shift in one submission ----
        groups_seq = list(self._groups.items())
        circuits_all: List[Circuit] = []
        items_by_circuit: List[List[Tuple]] = []  # type: ignore[type-arg]
        tags: List[Tuple[str, int]] = []

        def _append_variant(pvec: np.ndarray, tag: Tuple[str, int]):
            c0 = self._build_circuit(pvec)
            for bases, items in groups_seq:
                circuits_all.append(c0.extended(self._prefix_ops_for_bases(bases)))
                items_by_circuit.append(items)
                tags.append(tag)

        # base
        _append_variant(base, ("base", -1))

        s = 0.5 * pi
        for i in range(len(base)):
            p_plus = base.copy(); p_plus[i] += s
            p_minus = base.copy(); p_minus[i] -= s
            _append_variant(p_plus, ("plus", i))
            _append_variant(p_minus, ("minus", i))
        tasks = device_base.run(provider=provider, device=device, circuit=circuits_all, shots=shots, noise=noise, **device_kwargs)


        e0 = float(self._identity_const)
        plus_energy = np.zeros(len(base), dtype=np.float64)
        minus_energy = np.zeros(len(base), dtype=np.float64)

        for k, t in enumerate(tasks):
            rr = t.get_result(wait=False)
            pp_opts = dict(postprocessing or {})
            pp_opts.update({"method": "expval_pauli_sum", "identity_const": 0.0, "items": items_by_circuit[k]})
            post = apply_postprocessing(rr, pp_opts)
            e = float((post.get("result", {}) or {}).get("energy", 0.0))
            tag, idx = tags[k]
            if tag == "base":
                e0 += e
            elif tag == "plus":
                plus_energy[idx] += e
            elif tag == "minus":
                minus_energy[idx] += e

        g = np.zeros_like(base)
        for i in range(len(base)):
            g[i] = 0.5 * (plus_energy[i] - minus_energy[i])
        return float(e0), g

    def _prefix_ops_for_bases(self, bases: Tuple[str, ...]) -> List[Tuple]:
        if bases in self._prefix_cache:
            return self._prefix_cache[bases]
        ops: List[Tuple] = []
        for lsb_q, p in enumerate(bases):
            q = self.n - 1 - int(lsb_q)
            if p == "X":
                ops.append(("h", q))
            elif p == "Y":
                ops.append(("sdg", q)); ops.append(("h", q))
        for q in range(self.n):
            ops.append(("measure_z", q))
        self._prefix_cache[bases] = ops
        return ops

