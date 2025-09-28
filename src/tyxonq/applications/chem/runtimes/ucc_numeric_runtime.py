from __future__ import annotations

from typing import List, Tuple, Sequence, Any
from math import pi

import numpy as np
from openfermion import QubitOperator

from tyxonq.core.ir.circuit import Circuit
# Use simulator engine for exact statevector
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
from tyxonq.libs.circuits_library.ucc import build_ucc_circuit
from tyxonq.applications.chem.chem_libs.quantum_chem_library.ci_state_mapping import (
    get_ci_strings
)
from tyxonq.applications.chem.chem_libs.quantum_chem_library.civector_ops import (
    civector as _civector_build,
    energy_and_grad_civector as _energy_and_grad_civector,
    apply_excitation_civector as _apply_excitation_civ,
    apply_excitation_civector_nocache as _apply_excitation_civ_nc,
)
from tyxonq.applications.chem.chem_libs.quantum_chem_library.statevector_ops import (
    energy_statevector as _energy_statevector,
    energy_and_grad_statevector as _energy_and_grad_statevector,
)
from tyxonq.applications.chem.chem_libs.quantum_chem_library.civector_ops import apply_h_qubit_to_ci as _apply_h_qubit_to_ci
from tyxonq.applications.chem.chem_libs.quantum_chem_library.pyscf_civector import apply_excitation_pyscf as _apply_excitation_pyscf


class UCCNumericRuntime:
    def __init__(
        self,
        n_qubits: int,
        n_elec_s: Tuple[int, int],
        h_qubit_op: QubitOperator,
        *,
        ex_ops: List[Tuple] | None = None,
        param_ids: List[int] | None = None,
        init_state: Sequence[float] | Circuit | None = None,
        mode: str = "fermion",
        trotter: bool = False,
        decompose_multicontrol: bool = False,
        numeric_engine: str | None = None,
        ci_hamiltonian: Any | None = None,
        qop: QubitOperator | None = None,
        
        
    ):
        self.n_qubits = int(n_qubits)
        self.n_elec_s = (int(n_elec_s[0]), int(n_elec_s[1]))
        self.h_qubit_op = h_qubit_op
        self.ex_ops = list(ex_ops) if ex_ops is not None else None
        self.param_ids = list(param_ids) if param_ids is not None else None
        # Normalize default param_ids to 0..len(ex_ops)-1 when not provided
        if self.ex_ops is not None and (self.param_ids is None or len(self.param_ids) == 0):
            self.param_ids = list(range(len(self.ex_ops)))
        self.init_state = init_state
        self.mode = str(mode)
        self.trotter = bool(trotter)
        self.decompose_multicontrol = bool(decompose_multicontrol)
        self.numeric_engine = (numeric_engine or "statevector").lower()
        # Optional CI-space Hamiltonian apply (callable) or matrix for CI engines
        self.ci_hamiltonian = ci_hamiltonian
        # Optional: pre-mapped QubitOperator for statevector path reuse
        self._qop_cached = qop
        

        if self.ex_ops is not None:
            self.n_params = (max(self.param_ids) + 1) if (self.param_ids and len(self.param_ids) > 0) else len(self.ex_ops)
        else:
            self.n_params = 0
        # Internal caches for civector ops (per TCC style)
        self._ci_cache = {}
        # Cache keys removed; use centralized cache in statevector_ops

    def _build(self, params: Sequence[float]) -> Circuit:
        if self.ex_ops is None or self.n_params == 0:
            return Circuit(self.n_qubits, ops=[])
        return build_ucc_circuit(
            params,
            self.n_qubits,
            self.n_elec_s,
            tuple(self.ex_ops),
            tuple(self.param_ids) if self.param_ids is not None else None,
            mode=self.mode,
            init_state=self.init_state,
            decompose_multicontrol=self.decompose_multicontrol,
            trotter=self.trotter,
        )

    def _state(self, params: Sequence[float]) -> np.ndarray:
        if self.numeric_engine == "statevector":
            # Build CI vector using the same excitation semantics, then embed into full statevector
            civ = self._civector(params)
            ci_strings = np.asarray(get_ci_strings(self.n_qubits, self.n_elec_s, "fermion"), dtype=np.uint64)
            psi = np.zeros(1 << self.n_qubits, dtype=np.complex128)
            psi[ci_strings] = np.asarray(civ, dtype=np.complex128)
            return psi
        if self.numeric_engine in ("civector", "civector-large", "pyscf"):
            # Build CI vector and embed into statevector positions directly (OpenFermion ordering)
            civ = self._civector(params)
            ci_strings = np.asarray(get_ci_strings(self.n_qubits, self.n_elec_s, "fermion"), dtype=np.uint64)
            psi = np.zeros(1 << self.n_qubits, dtype=np.complex128)
            psi[ci_strings] = np.asarray(civ, dtype=np.complex128)
            return psi
        if self.numeric_engine == "mps":
            # TODO: replace with MatrixProductStateEngine exact MPS contraction when available
            c = self._build(params)
            eng = StatevectorEngine()
            psi = np.asarray(eng.state(c), dtype=np.complex128)
            return self._align_statevector_order(psi)
        # Fallback
        c = self._build(params)
        eng = StatevectorEngine()
        psi = np.asarray(eng.state(c), dtype=np.complex128)
        return self._align_statevector_order(psi)

    def _civector(self, params: Sequence[float]) -> np.ndarray:
        """Build CI vector in CI space (no embedding), following TCC conventions."""
        base = (
            np.zeros(self.n_params, dtype=np.float64)
            if (isinstance(params, Sequence) and len(params) == 0 and self.n_params > 0)
            else np.asarray(params, dtype=np.float64)
        )
        ex_ops = tuple(self.ex_ops) if self.ex_ops is not None else tuple()
        param_ids = self.param_ids if self.param_ids is not None else list(range(self.n_params))

        civ_init = None
        init = self.init_state
        if init is not None:
            try:
                if isinstance(init, Circuit):
                    eng0 = StatevectorEngine()
                    psi0 = np.asarray(eng0.state(init), dtype=np.complex128)
                    ci_strings = get_ci_strings(self.n_qubits, self.n_elec_s, "fermion")
                    civ_init = np.real(psi0[ci_strings]).astype(np.float64, copy=False)
                elif isinstance(init, np.ndarray):
                    arr = np.asarray(init)
                    ci_strings = get_ci_strings(self.n_qubits, self.n_elec_s, "fermion")
                    if arr.size == (1 << self.n_qubits):
                        civ_init = np.real(arr[ci_strings]).astype(np.float64, copy=False)
                    elif arr.size == len(ci_strings):
                        civ_init = np.real(arr).astype(np.float64, copy=False)
            except Exception:
                civ_init = None

        civ = _civector_build(base, self.n_qubits, self.n_elec_s, list(ex_ops), param_ids, mode=self.mode, init_state=civ_init)
        return np.asarray(civ, dtype=np.float64).reshape(-1)

    # removed energy helpers; statevector path aligns to CI baseline for numeric parity

    # (diagnostic helper removed; engine ordering matches ci_strings directly)

    def _expect(self, psi: np.ndarray) -> float:
        # Use cached sparse operator via statevector_ops cache util
        from tyxonq.applications.chem.chem_libs.quantum_chem_library.statevector_ops import _cached_sparse_from_key, _qop_to_key
        key = _qop_to_key(self.h_qubit_op, int(self.n_qubits))
        H = _cached_sparse_from_key(key)
        vec = psi.reshape(-1)
        e = np.vdot(vec, H.dot(vec))
        return float(np.real(e))

    def _align_statevector_order(self, psi: np.ndarray) -> np.ndarray:
        """Align engine statevector bit order to OpenFermion's convention by bit-reversal.

        OpenFermion assumes qubit 0 acts on least-significant bit. If the simulator
        uses most-significant as qubit 0, reverse axes.
        """
        n = self.n_qubits
        if psi.ndim != 1 or psi.size != (1 << n):
            return psi
        try:
            arr = psi.reshape((2,) * n)
            arr = np.transpose(arr, axes=tuple(range(n))[::-1])
            return arr.reshape(-1)
        except Exception:
            return psi

    def energy(self, params: Sequence[float] | None = None) -> float:
        if params is None:
            base = np.zeros(self.n_params, dtype=np.float64) if self.n_params > 0 else np.zeros(0, dtype=np.float64)
        else:
            base = np.asarray(params, dtype=np.float64)
        if self.numeric_engine in ("civector", "civector-large"):
            # Native CI-space expectation: bra = H|ket|
            civector = self._civector(base)
            if self.ci_hamiltonian is not None:
                bra = self.ci_hamiltonian(civector)
            else:
                bra = _apply_h_qubit_to_ci(self.h_qubit_op, self.n_qubits, self.n_elec_s, civector, mode=self.mode)
            denom = float(np.dot(civector, civector))
            if denom == 0.0:
                return float("nan")
            return float(float(np.dot(bra, civector)) / denom)
        if self.numeric_engine == "pyscf":
            if self.ci_hamiltonian is None:
                raise RuntimeError("ci_hamiltonian is required for CI-based numeric_engine")
            civector = self._civector(base)
            bra = self.ci_hamiltonian(civector)
            denom = float(np.dot(civector, civector))
            if denom == 0.0:
                return float("nan")
            return float(float(np.dot(bra, civector)) / denom)
        if self.numeric_engine == "statevector":
            qop = self._qop_cached if self._qop_cached is not None else self.h_qubit_op
            return float(_energy_statevector(base, qop, self.n_qubits, self.n_elec_s, self.ex_ops, self.param_ids, mode=self.mode, init_state=self.init_state))
        # Default: evaluate in full statevector space
        psi = self._state(base)
        return float(self._expect(psi))

    def energy_and_grad(self, params: Sequence[float] | None = None) -> Tuple[float, np.ndarray]:
        if params is None:
            base = np.zeros(self.n_params, dtype=np.float64) if self.n_params > 0 else np.zeros(0, dtype=np.float64)
        else:
            base = np.asarray(params, dtype=np.float64)
        if self.numeric_engine == "pyscf":
            if self.ci_hamiltonian is None:
                raise RuntimeError("ci_hamiltonian is required for CI-based numeric_engine")
            return self._get_energy_and_grad_pyscf(base)
        # Statevector: dedicated energy and parameter-shift gradient
        if self.numeric_engine == "statevector":
            qop = self._qop_cached if self._qop_cached is not None else self.h_qubit_op
            e, g = _energy_and_grad_statevector(base, qop, self.n_qubits, self.n_elec_s, self.ex_ops, self.param_ids, mode=self.mode, init_state=self.init_state)
            return float(e), np.asarray(g, dtype=np.float64)
        # civector / civector-large: implement native analytic gradient following TCC
        if self.numeric_engine in ("civector", "civector-large"):
            ex_ops = list(self.ex_ops or [])
            param_ids = self.param_ids or list(range(self.n_params))
            if self.numeric_engine == "civector-large":
                from tyxonq.applications.chem.chem_libs.quantum_chem_library.civector_ops import energy_and_grad_civector_nocache as _eg_nc
                e, g = _eg_nc(
                    base,
                    self.h_qubit_op,
                    self.n_qubits,
                    self.n_elec_s,
                    ex_ops,
                    param_ids,
                    mode=self.mode,
                    init_state=None,
                    ci_apply=self.ci_hamiltonian,
                )
            else:
                e, g = _energy_and_grad_civector(
                    base,
                    self.h_qubit_op,
                    self.n_qubits,
                    self.n_elec_s,
                    ex_ops,
                    param_ids,
                    mode=self.mode,
                    init_state=None,
                    ci_apply=self.ci_hamiltonian,
                )
            return float(e), np.asarray(g, dtype=np.float64)
        e0 = self.energy(base)
        if self.n_params == 0:
            return float(e0), np.zeros(0, dtype=np.float64)
        g = np.zeros_like(base)
        s = 0.5 * pi
        for i in range(len(base)):
            p_plus = base.copy(); p_plus[i] += s
            p_minus = base.copy(); p_minus[i] -= s
            e_plus = self.energy(p_plus)
            e_minus = self.energy(p_minus)
            g[i] = 0.5 * (e_plus - e_minus)
        return float(e0), g

    # ---- CI-space Hamiltonian application and gradients (TCC-style) ----
    def _get_energy_and_grad_civector(self, params: np.ndarray, *, nocache: bool = False):
        # Deprecated in runtime; use civector_ops.energy_and_grad_civector directly
        return _energy_and_grad_civector(
            params,
            self.h_qubit_op,
            self.n_qubits,
            self.n_elec_s,
            list(self.ex_ops or []),
            self.param_ids or list(range(self.n_params)),
            mode=self.mode,
            init_state=None,
        )

    def _get_energy_and_grad_pyscf(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute normalized CI energy and analytic gradient in CI space using PySCF-style helpers."""
        if self.ci_hamiltonian is None:
            raise RuntimeError("ci_hamiltonian is required for CI-based numeric_engine")
        from tyxonq.applications.chem.chem_libs.quantum_chem_library.pyscf_civector import (
            get_energy_and_grad_pyscf as _geag,
        )
        e, g = _geag(
            params,
            self.ci_hamiltonian,
            self.n_qubits,
            self.n_elec_s,
            tuple(self.ex_ops) if self.ex_ops is not None else tuple(),
            self.param_ids if self.param_ids is not None else list(range(self.n_params)),
            mode=self.mode,
            init_state=None,
        )
        return float(e), np.asarray(g, dtype=np.float64)

# Compatibility helper for tests (replaces legacy engine_ucc.apply_excitation)
def apply_excitation(state: np.ndarray, n_qubits: int, n_elec_s, ex_op: tuple, mode: str, numeric_engine: str | None = None, engine: str | None = None) -> np.ndarray:
    eng = (numeric_engine or "statevector").lower()
    if eng == "statevector":
        from tyxonq.applications.chem.chem_libs.quantum_chem_library.statevector_ops import apply_excitation_statevector as _apply_excitation_sv
        n_elec = int(sum(n_elec_s)) if isinstance(n_elec_s, (tuple, list)) else int(n_elec_s)
        return _apply_excitation_sv(state, n_qubits, n_elec, ex_op, mode)
    if eng == "civector":
        return _apply_excitation_civ(state, n_qubits, n_elec_s, ex_op, mode)
    if eng == "civector-large":
        return _apply_excitation_civ_nc(state, n_qubits, n_elec_s, ex_op, mode)
    if eng == "pyscf":
        return _apply_excitation_pyscf(state, n_qubits, n_elec_s, ex_op, mode)
    # fallback
    from tyxonq.applications.chem.chem_libs.quantum_chem_library.statevector_ops import apply_excitation_statevector as _apply_excitation_sv
    n_elec = int(sum(n_elec_s)) if isinstance(n_elec_s, (tuple, list)) else int(n_elec_s)
    return _apply_excitation_sv(state, n_qubits, n_elec, ex_op, mode)



