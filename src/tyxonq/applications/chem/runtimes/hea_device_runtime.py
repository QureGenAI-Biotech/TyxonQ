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


Hamiltonian = List[Tuple[float, List[Tuple[str, int]]]]


class HEADeviceRuntime:
    def __init__(self, n: int, layers: int, hamiltonian: Hamiltonian, *, n_elec_s: Tuple[int, int] | None = None, mapping: str | None = None, circuit_template: list | None = None):
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
            return float(device_base.expval(provider=provider, device=device, circuit=c, observable=qop, noise=noise, **device_kwargs))

        # simple grouping by basis pattern
        identity_const, groups = group_hamiltonian_pauli_terms(self.hamiltonian, self.n)

        energy_val = identity_const
        for bases, items in groups.items():
            c = self._build_circuit(params)
            # apply basis rotations
            # bases order originates from grouping which assumes OpenFermion little-endian (q=0 is LSB)
            # Our IR uses big-endian indices in ops. Map index accordingly: be_idx = n-1-lsb_idx
            for lsb_q, p in enumerate(bases):
                q = self.n - 1 - int(lsb_q)
                if p == "X":
                    c.ops.append(("h", q))
                elif p == "Y":
                    c.ops.append(("sdg", q)); c.ops.append(("h", q))
            for q in range(self.n):
                c.ops.append(("measure_z", q))
            dev = c.device(provider=provider, device=device, shots=shots, noise=noise, **device_kwargs)
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

        # shots 路径统一由 driver+engine 归一；shots=0 时通过 device.base.expval 调用解析快径 + 有限差分
        if (provider in ("simulator", "local")) and int(shots) == 0:
            e0 = self.energy(base, shots=0, provider=provider, device=device, postprocessing=postprocessing, noise=noise, **device_kwargs)
            g = np.zeros_like(base)
            eps = 1e-7
            for i in range(len(base)):
                p_plus = base.copy(); p_plus[i] += eps
                p_minus = base.copy(); p_minus[i] -= eps
                e_plus = self.energy(p_plus, shots=0, provider=provider, device=device, postprocessing=postprocessing, noise=noise, **device_kwargs)
                e_minus = self.energy(p_minus, shots=0, provider=provider, device=device, postprocessing=postprocessing, noise=noise, **device_kwargs)
                g[i] = (e_plus - e_minus) / (2.0 * eps)
            return float(e0), g

        e0 = self.energy(base, shots=shots, provider=provider, device=device, postprocessing=postprocessing, noise=noise, **device_kwargs)
        g = np.zeros_like(base)
        s = 0.5 * pi
        for i in range(len(base)):
            p_plus = base.copy(); p_plus[i] += s
            p_minus = base.copy(); p_minus[i] -= s
            e_plus = self.energy(p_plus, shots=shots, provider=provider, device=device, postprocessing=postprocessing, noise=noise, **device_kwargs)
            e_minus = self.energy(p_minus, shots=shots, provider=provider, device=device, postprocessing=postprocessing, noise=noise, **device_kwargs)
            g[i] = 0.5 * (e_plus - e_minus)
        return e0, g

