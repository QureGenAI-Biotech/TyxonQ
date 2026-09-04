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
        # bases 按 OpenFermion 比特索引（LSB 优先）枚举；电路态制备与
        # counts/probabilities 聚合都按 IR 索引直读（位串位置 = IR 比特），
        # 故旋转与测量必须放在 IR 比特 q 上，否则聚合会读到镜像比特。
        n = self.n_qubits
        for q, p in enumerate(bases):
            if p == "X":
                ops.append(("h", q))
            elif p == "Y":
                ops.append(("rz", q, -pi/2)); ops.append(("h", q))
        for q in range(n):
            ops.append(("measure_z", q))
        self._prefix_cache[bases] = ops
        return ops

    # TODO (device-runtime backlog):
    # 1) Add adjoint differentiation for simulator shots=0 to replace finite-difference (faster, exact on statevector)
    # 2) Support SPSA/gradient-free optimizers for hardware shots>0 to reduce evaluations
    # 3) Batch/parallel parameter shifts and group evaluations; reuse compiled prefixes/suffixes
    # 4) Adaptive shots allocation per parameter/group based on variance/sensitivity
    # 5) Optional low-rank/commuting-group Hamiltonian transforms to reduce measurement cost
    # 6) Caching of expectation terms across close parameters during local line-search
    # 7) RESOLVED (2026-09): the "ansatz equivalence gap" (gate-level singles block
    #    energy-inert, ~1e-4 Ha gap vs numeric on H4, doubles grad 0.416 vs 0.454)
    #    was NOT a decomposition defect. The missing "cry" branch in state() was only
    #    a SYMPTOM of a systemic design defect: every engine carried SEVERAL independent
    #    op-dispatch loops (run()/state()/expval()/expectation_pauli(), plus Circuit.state()'s
    #    inline MPS re-execution loop and its silent density_matrix->statevector fallback)
    #    -- 8+ forks, each covering a different gate set, ALL silently `continue`-ing on
    #    unknown ops. Concretely y/z/t/tdg/cy were dropped in EVERY loop of ALL THREE
    #    engines (0 grep hits), so the shots=0 analytic path degenerated the singles block
    #    (cx + parity + cry + parity^-1 + cx) into the identity. The decomposition itself is
    #    exact: same structure as TenCirChem evolve_tensornetwork.evolve_excitation
    #    (arXiv:2005.14475), whose gate-level circuit matches its operator-level (civector)
    #    energies to 1e-10. FIXED by single-sourcing op dispatch (2026-09): each engine now
    #    has ONE dispatch loop (_evolve / run) driven by the authoritative
    #    gate_table.resolve_unitary vocabulary (all 1q/2q unitaries + special + control ops);
    #    run()/state()/expval()/expectation_pauli() all route through it, unknown/unsupported
    #    ops raise instead of silently skipping, and driver.run(shots=0) reuses the same run()
    #    output (no second state() call). gate mode matches numeric to ~1e-10 (H2 sweep, H4
    #    random points); gate-mode PSR grad matches numeric to ~2e-6 (two-shift rule, item 8).
    #    Regression: tests_applications_chem/test_device_runtime_regression.py (blind spots
    #    4/5) + tests_core_module/test_engine_dispatch_completeness.py (dispatch completeness
    #    / run()-state() parity / unknown-op raise / driver shots=0 single-source).
    # 8) RESOLVED (2026-09): the UCC energy surface E(theta) contains only EVEN
    #    harmonics {2,4,...,2m} (exp(theta*A) with A^2=-I gives amplitudes ~cos/sin(theta);
    #    m = blocks sharing one parameter, plus k=4 from trotterized commuting string
    #    pairs), so the old ±pi/2 shift gave sin(2k*pi/2)=0 -> gradient identically ~0.
    #    Replaced with the two-shift rule g = 2*D(pi/8) + (1-sqrt(2))*D(pi/4),
    #    D(s) = E(theta+s) - E(theta-s), exact for harmonics {2,4} (solves
    #    sum_j a_j sin(2k s_j) = k for k=1,2). Verified: both trotter AND gate mode
    #    match numeric grad to ~1e-6 (the earlier "fractional-harmonic gate-mode
    #    artifacts" were the missing-cry illusion of item 7). HEA runtime keeps
    #    ±pi/2 (RY gates are frequency-1, half-angle convention -- correct there).
    #    TenCirChem reference uses JAX AD (jit(value_and_grad)) for UCC, never PSR.


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
        
        # Use device_base.run() for proper device support
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

    def _build_hf_circuit(self) -> Circuit:
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
        """Compute energy using batched circuits and unified postprocessing.
        
        Args:
            c_builder: Callable that returns a Circuit object
            shots: Number of measurement shots
            provider: Device provider
            device: Device name
            postprocessing: Postprocessing options
            noise: Noise configuration
            **device_kwargs: Additional device options
            
        Returns:
            Energy value as float
        """
        # Use cached grouping and measurement prefixes
        energy_val = float(self._identity_const)
        # Build base circuit once; batch all grouped circuits for single submission
        base_circuit = c_builder()
        circuits: List[Circuit] = []
        items_by_idx: List[List[Tuple]] = []  # type: ignore[type-arg]
        for bases, items in self._groups.items():
            circuits.append(base_circuit.extended(self._prefix_ops_for_bases(bases)))
            items_by_idx.append(items)

        # Execute batch with unified postprocessing
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
        """Compute energy and gradient using batched parameter shifts.

        Gradient rule (method="fd", the default): two-shift PSR
            g = 2*[E(θ+π/8) − E(θ−π/8)] + (1−√2)*[E(θ+π/4) − E(θ−π/4)]
        The UCC energy surface only contains EVEN harmonics cos(2kθ+φ), k=1..m
        (exp(θA) with A²=−I ⇒ state amplitudes ~cos θ, sin θ; m grows with the
        number of excitation blocks sharing a parameter and with trotterized
        commuting-string products). The classic ±π/2 rule evaluates sin(2k·π/2)=0
        and returns ~0 for EVERY parameter; the two-shift rule above is exact for
        harmonics {2,4} (weights solve Σ_j a_j·sin(2k·s_j) = k for k=1,2).
        Use gradient_method="ps" for a small-step central finite difference
        (robust for arbitrary harmonics, but amplifies shot noise by 1/(2δ)).

        Args:
            params: Parameter vector
            shots: Number of measurement shots
            provider: Device provider
            device: Device name
            postprocessing: Postprocessing options
            noise: Noise configuration
            gradient_method: "fd" (two-shift PSR, default) or "ps" (central finite difference)
            **device_kwargs: Additional device options
            
        Returns:
            Tuple of (energy, gradient)
        """
        if self.n_params == 0:
            e0 = self.energy(None, shots=shots, provider=provider, device=device, postprocessing=postprocessing)
            return e0, np.zeros(0, dtype=np.float64)

        base = np.asarray(params if params is not None else np.zeros(self.n_params, dtype=np.float64), dtype=np.float64)
        
        # ---- Build all circuit variants: base + parameter shifts ----
        groups_seq = list(self._groups.items())
        circuits_all: List[Circuit] = []
        items_by_circuit: List[List[Tuple]] = []  # type: ignore[type-arg]
        tags: List[Tuple[str, int]] = []  # (variant, param_index)

        def _append_variant(pvec: np.ndarray, tag: Tuple[str, int]):
            """Build all basis-rotated circuits for a given parameter vector."""
            c0 = self._build_ucc_circuit(pvec)
            for bases, items in groups_seq:
                circuits_all.append(c0.extended(self._prefix_ops_for_bases(bases)))
                items_by_circuit.append(items)
                tags.append(tag)

        # Base energy evaluation
        _append_variant(base, ("base", -1))

        # Parameter shift evaluations
        method = str(gradient_method).lower()
        # Two-shift PSR: exact for the even harmonics {2,4} of the UCC energy surface.
        # Weights solve a1*sin(2k*s1) + a2*sin(2k*s2) = k for k=1,2 with s=(π/8, π/4).
        psr_shifts = (float(np.pi / 8.0), float(np.pi / 4.0))
        psr_weights = (2.0, 1.0 - float(np.sqrt(2.0)))
        if method == "fd":
            for i in range(len(base)):
                for j, shift in enumerate(psr_shifts):
                    p_plus = base.copy(); p_plus[i] += shift
                    p_minus = base.copy(); p_minus[i] -= shift
                    _append_variant(p_plus, (f"plus{j}", i))
                    _append_variant(p_minus, (f"minus{j}", i))
        else:
            # Parameter shift with smaller step (numerical gradient)
            step = float(np.pi / 90.0)
            for i in range(len(base)):
                p_plus = base.copy(); p_plus[i] += step
                p_minus = base.copy(); p_minus[i] -= step
                _append_variant(p_plus, ("plus_s", i))
                _append_variant(p_minus, ("minus_s", i))

        # Execute batch with unified postprocessing
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

        # Aggregate results: extract energies and accumulate by (tag, param) variant
        e0 = float(self._identity_const)
        n_params = len(base)
        shifted_energy: Dict[Tuple[str, int], float] = {}

        for k, result in enumerate(results):
            # Extract energy from postprocessed result
            energy_contrib = self._extract_energy_from_postprocessing(result)
            
            # Accumulate energy by shift type
            tag, idx = tags[k]
            if tag == "base":
                e0 += energy_contrib
            elif 0 <= idx < n_params:
                key = (tag, idx)
                shifted_energy[key] = shifted_energy.get(key, 0.0) + energy_contrib

        def _diff(tag_plus: str, tag_minus: str, i: int) -> float:
            return (shifted_energy.get((tag_plus, i), 0.0)
                    - shifted_energy.get((tag_minus, i), 0.0))

        # Compute gradients using appropriate rule
        g = np.zeros_like(base)
        if method == "fd":
            # Two-shift PSR: g = Σ_j w_j * [E(θ+s_j) − E(θ−s_j)]
            for i in range(n_params):
                g[i] = sum(
                    w * _diff(f"plus{j}", f"minus{j}", i)
                    for j, w in enumerate(psr_weights)
                )
            return float(e0), g
        else:
            # Numerical gradient: (E[+δ] - E[-δ]) / (2δ)
            step = float(np.pi / 90.0)
            for i in range(n_params):
                g[i] = _diff("plus_s", "minus_s", i) / (2.0 * step)
            return float(e0), g

