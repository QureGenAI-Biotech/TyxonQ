from __future__ import annotations

from typing import List, Tuple, Dict, Sequence, Any
from math import pi
from functools import lru_cache

import numpy as np

from tyxonq.core.ir.circuit import Circuit
from tyxonq.libs.circuits_library.blocks import build_hwe_ry_ops
from tyxonq.libs.circuits_library.qiskit_real_amplitudes import build_circuit_from_template
from tyxonq.compiler.utils.hamiltonian_grouping import (
    group_hamiltonian_pauli_terms,
)
from openfermion import QubitOperator


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
        
        # Cache for built circuits with parameter shift variants
        self._circuit_cache: Dict[str, Circuit] = {}
        
        # 可选：直接消费上游映射缓存的 QubitOperator（shots==0 快径）
        self._qop_cached = qop

    def _build_circuit(self, params: Sequence[float]) -> Circuit:
        # If external template exists, instantiate from it
        if self.circuit_template is not None:
            return build_circuit_from_template(self.circuit_template, np.asarray(params, dtype=np.float64), n_qubits=self.n)
        # Default: RY-only ansatz
        return build_hwe_ry_ops(self.n, self.layers, params)
    
    def _execute_circuits(
        self,
        circuits: List[Circuit],
        provider: str,
        device: str,
        shots: int,
        pauli_items_list: List[List[Tuple]] | None = None,
        postprocessing: dict | None = None,
        noise: dict | None = None,
        **device_kwargs,
    ) -> List[Dict]:
        """Execute a batch of circuits using device_base.run() with proper Pauli postprocessing.
        
        Args:
            circuits: List of Circuit objects to execute
            provider: Device provider (e.g., "simulator")
            device: Device name (e.g., "statevector")
            shots: Number of measurement shots
            pauli_items_list: List of Pauli items for each circuit (for postprocessing)
            postprocessing: Optional postprocessing configuration
            noise: Optional noise configuration
            **device_kwargs: Additional device options
            
        Returns:
            List of processed result dictionaries with extracted energies
        """
        from tyxonq.devices import base as device_base
        from tyxonq.postprocessing import apply_postprocessing
        
        # Use device_base.run() for proper hardware support
        tasks = device_base.run(
            provider=provider,
            device=device,
            circuit=circuits,
            shots=shots,
            noise=noise,
            **device_kwargs
        )
        
        results = []
        for k, t in enumerate(tasks):
            rr = t.get_result(wait=False)
            # Apply Pauli-based postprocessing with per-circuit Pauli items
            pp_opts = dict(postprocessing or {})
            if pauli_items_list and k < len(pauli_items_list):
                pp_opts.update({
                    "method": "expval_pauli_sum",
                    "identity_const": 0.0,
                    "items": pauli_items_list[k]
                })
            post = apply_postprocessing(rr, pp_opts)
            results.append(post)
        return results

    @staticmethod
    def _extract_energy_from_postprocessing(post: Dict) -> float:
        """Extract energy value from postprocessing result.
        
        Args:
            post: Postprocessing result dict
            
        Returns:
            Energy value as float, or 0.0 if extraction fails
        """
        payload = post.get("result", {})
        return float((payload or {}).get("energy", 0.0))

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
        """Compute energy using Circuit.run() for proper compilation and execution.
        
        This method follows the TyxonQ architecture:
        Problem → Hamiltonian → Circuit → Compile → Execute → Postprocess
        
        Key improvements:
        - Uses Circuit.run() to ensure compilation step is executed
        - Supports both simulator and hardware targets
        - Batches all measurement circuits for efficient submission
        - Integrates with Circuit's built-in postprocessing pipeline
        """
        if params is None:
            params = self.init_guess
        # If using template, parameter length is defined by template; skip RY param-length check
        if self.circuit_template is None and len(params) != self.n_params:
            raise ValueError(f"params length {len(params)} != {self.n_params}")

        # Build base circuit and extended variants with Pauli basis rotations
        energy_val = self._identity_const
        # 仅构建一次基电路；将各分组电路合并为一批提交
        base_circuit = self._build_circuit(params)
        circuits: List[Circuit] = []
        items_by_idx: List[List[Tuple]] = []  # type: ignore[type-arg]
        for bases, items in self._groups.items():
            circuits.append(base_circuit.extended(self._prefix_ops_for_bases(bases)))
            items_by_idx.append(items)

        # Execute batch of circuits with Pauli postprocessing
        results = self._execute_circuits(
            circuits=circuits,
            provider=provider,
            device=device,
            shots=shots,
            pauli_items_list=items_by_idx,
            postprocessing=postprocessing,
            noise=noise,
            **device_kwargs
        )
        
        # Aggregate energy from postprocessed results
        for result in results:
            energy_contrib = self._extract_energy_from_postprocessing(result)
            energy_val += energy_contrib
        
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
        """Compute energy and gradient using parameter shift rule with Circuit.run().
        
        Implements efficient parameter shift rule (PSR) gradient calculation:
        - ∂E/∂θ_i ≈ (E[θ_i + π/2] - E[θ_i - π/2]) / 2
        
        Optimizations:
        - Batches all variant circuits (base + param shifts) for submission
        - Uses Circuit.run() for proper compilation
        - Caches prefix operations for repeated basis rotations
        - Aggregates energy results across all Pauli basis groups
        """
        if params is None:
            params = self.init_guess
        base = np.asarray(params, dtype=np.float64)

        # ---- Build all circuit variants: base + parameter shifts ----
        groups_seq = list(self._groups.items())
        circuits_all: List[Circuit] = []
        items_by_circuit: List[List[Tuple]] = []  # type: ignore[type-arg]
        tags: List[Tuple[str, int]] = []

        def _append_variant(pvec: np.ndarray, tag: Tuple[str, int]):
            """Build all basis-rotated circuits for a given parameter vector."""
            c0 = self._build_circuit(pvec)
            for bases, items in groups_seq:
                circuits_all.append(c0.extended(self._prefix_ops_for_bases(bases)))
                items_by_circuit.append(items)
                tags.append(tag)

        # Base energy evaluation
        _append_variant(base, ("base", -1))

        # Parameter shift evaluations: E[θ + π/2] and E[θ - π/2] for each parameter
        s = 0.5 * pi
        for i in range(len(base)):
            p_plus = base.copy(); p_plus[i] += s
            p_minus = base.copy(); p_minus[i] -= s
            _append_variant(p_plus, ("plus", i))
            _append_variant(p_minus, ("minus", i))

        # Execute batch of circuits with Pauli postprocessing
        results = self._execute_circuits(
            circuits=circuits_all,
            provider=provider,
            device=device,
            shots=shots,
            pauli_items_list=items_by_circuit,
            postprocessing=postprocessing,
            noise=noise,
            **device_kwargs
        )

        # Aggregate results: extract energies and accumulate by parameter shift type
        e0 = float(self._identity_const)
        plus_energy = np.zeros(len(base), dtype=np.float64)
        minus_energy = np.zeros(len(base), dtype=np.float64)

        for k, result in enumerate(results):
            # Extract energy from postprocessed result
            energy_contrib = self._extract_energy_from_postprocessing(result)
            
            # Accumulate energy by shift type
            tag, idx = tags[k]
            if tag == "base":
                e0 += energy_contrib
            elif tag == "plus":
                plus_energy[idx] += energy_contrib
            elif tag == "minus":
                minus_energy[idx] += energy_contrib

        # Compute gradients: g_i = (E[+] - E[-]) / 2
        g = np.zeros_like(base)
        for i in range(len(base)):
            g[i] = 0.5 * (plus_energy[i] - minus_energy[i])
        return float(e0), g

    def _prefix_ops_for_bases(self, bases: Tuple[str, ...]) -> List[Tuple]:
        """Generate cached basis rotation and measurement operations.
        
        For measuring in non-Z bases:
        - X basis: Apply H gate before Z measurement
        - Y basis: Apply S† then H before Z measurement
        - Z basis: Measure directly (no rotation)
        
        Args:
            bases: Tuple of Pauli characters ('X', 'Y', 'Z') for each qubit
            
        Returns:
            List of operation tuples (gate_name, qubit_indices)
        """
        if bases in self._prefix_cache:
            return self._prefix_cache[bases]
        ops: List[Tuple] = []
        # Apply basis rotations to convert from measured basis to Z basis
        for lsb_q, p in enumerate(bases):
            q = self.n - 1 - int(lsb_q)
            if p == "X":
                ops.append(("h", q))
            elif p == "Y":
                # Y basis: Apply S† (inverse phase gate) then H
                ops.append(("sdg", q))
                ops.append(("h", q))
        # Measure all qubits in Z basis (after basis rotations)
        for q in range(self.n):
            ops.append(("measure_z", q))
        self._prefix_cache[bases] = ops
        return ops

