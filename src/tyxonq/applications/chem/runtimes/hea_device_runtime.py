from __future__ import annotations

from typing import List, Tuple, Dict, Sequence
from math import pi

import numpy as np

from tyxonq.core.ir.circuit import Circuit
from tyxonq.libs.circuits_library.blocks import build_hwe_ry_ops
from tyxonq.compiler.utils.hamiltonian_grouping import (
    group_hamiltonian_pauli_terms,
)


Hamiltonian = List[Tuple[float, List[Tuple[str, int]]]]


class HEADeviceRuntime:
    def __init__(self, n: int, layers: int, hamiltonian: Hamiltonian):
        self.n = int(n)
        self.layers = int(layers)
        self.hamiltonian = list(hamiltonian)
        # RY ansatz uses (layers + 1) * n parameters
        self.n_params = (self.layers + 1) * self.n
        self.init_guess = np.zeros(self.n_params, dtype=np.float64)

    def _build_circuit(self, params: Sequence[float]) -> Circuit:
        return build_hwe_ry_ops(self.n, self.layers, params)

    def energy(
        self,
        params: Sequence[float] | None = None,
        *,
        shots: int = 4096,
        provider: str = "simulator",
        device: str = "statevector",
        postprocessing: dict | None = None,
    ) -> float:
        if params is None:
            params = self.init_guess
        if len(params) != self.n_params:
            raise ValueError(f"params length {len(params)} != {self.n_params}")

        # Fast path: simulator/local + shots==0 → exact expectation without grouping
        # shots 路径统一由 driver+engine 归一；shots=0 时通过 device.base.expval 调用解析快径
        if (provider in ("simulator", "local")) and int(shots) == 0:
            from openfermion import QubitOperator
            qop = QubitOperator()
            for coeff, ops in self.hamiltonian:
                if not ops:
                    qop += coeff
                    continue
                term = tuple((int(q), str(P).upper()) for (P, q) in ops)
                qop += QubitOperator(term, float(coeff))
            c = self._build_circuit(params)
            from tyxonq.devices import base as device_base
            return float(device_base.expval(provider=provider, device=device, circuit=c, observable=qop))

        # simple grouping by basis pattern
        identity_const, groups = group_hamiltonian_pauli_terms(self.hamiltonian, self.n)

        energy_val = identity_const
        for bases, items in groups.items():
            c = self._build_circuit(params)
            # apply basis rotations
            for q, p in enumerate(bases):
                if p == "X":
                    c.ops.append(("h", q))
                elif p == "Y":
                    c.ops.append(("sdg", q)); c.ops.append(("h", q))
            for q in range(self.n):
                c.ops.append(("measure_z", q))
            dev = c.device(provider=provider, device=device, shots=shots)
            # Chainable postprocessing per group
            pp_opts = dict(postprocessing or {})
            pp_opts.update({
                "method": "expval_pauli_sum",
                "identity_const": 0.0,
                "items": items,
            })
            dev = dev.postprocessing(**pp_opts)
            res = dev.run()
            payload = res[0]["postprocessing"]["result"] if isinstance(res, list) else (res.get("postprocessing", {}) or {}).get("result", {})
            energy_val += float((payload or {}).get("energy", 0.0))
        return float(energy_val)

    def energy_and_grad(
        self,
        params: Sequence[float] | None = None,
        *,
        shots: int = 4096,
        provider: str = "simulator",
        device: str = "statevector",
        postprocessing: dict | None = None,
    ) -> Tuple[float, np.ndarray]:
        if params is None:
            params = self.init_guess
        base = np.asarray(params, dtype=np.float64)

        # shots 路径统一由 driver+engine 归一；shots=0 时通过 device.base.expval 调用解析快径 + 有限差分
        if (provider in ("simulator", "local")) and int(shots) == 0:
            e0 = self.energy(base, shots=0, provider=provider, device=device, postprocessing=postprocessing)
            g = np.zeros_like(base)
            eps = 1e-7
            for i in range(len(base)):
                p_plus = base.copy(); p_plus[i] += eps
                p_minus = base.copy(); p_minus[i] -= eps
                e_plus = self.energy(p_plus, shots=0, provider=provider, device=device, postprocessing=postprocessing)
                e_minus = self.energy(p_minus, shots=0, provider=provider, device=device, postprocessing=postprocessing)
                g[i] = (e_plus - e_minus) / (2.0 * eps)
            return float(e0), g

        e0 = self.energy(base, shots=shots, provider=provider, device=device, postprocessing=postprocessing)
        g = np.zeros_like(base)
        s = 0.5 * pi
        for i in range(len(base)):
            p_plus = base.copy(); p_plus[i] += s
            p_minus = base.copy(); p_minus[i] -= s
            e_plus = self.energy(p_plus, shots=shots, provider=provider, device=device, postprocessing=postprocessing)
            e_minus = self.energy(p_minus, shots=shots, provider=provider, device=device, postprocessing=postprocessing)
            g[i] = 0.5 * (e_plus - e_minus)
        return e0, g

