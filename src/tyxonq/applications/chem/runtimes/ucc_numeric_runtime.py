from __future__ import annotations

from typing import List, Tuple, Sequence, Any
from math import pi
from functools import lru_cache

import numpy as np
from openfermion import QubitOperator

from tyxonq.applications.chem.runtimes.hea_device_runtime import Hamiltonian
from tyxonq.core.ir.circuit import Circuit
# Use Circuit.state() API instead of StatevectorEngine for modern interface
from tyxonq.libs.circuits_library.ucc import build_ucc_circuit
from tyxonq.applications.chem.chem_libs.quantum_chem_library.ci_state_mapping import (
    get_ci_strings,
    statevector_to_civector,
    civector_to_statevector
)
from tyxonq.applications.chem.chem_libs.quantum_chem_library.civector_ops import (
    get_civector,
    energy_and_grad_civector_nocache,
    get_civector_nocache,
    energy_and_grad_civector,
    apply_excitation_civector as _apply_excitation_civ,
    apply_excitation_civector_nocache as _apply_excitation_civ_nc,
)
from tyxonq.applications.chem.chem_libs.quantum_chem_library.statevector_ops import (
    get_statevector,
    energy_and_grad_statevector
)
from tyxonq.applications.chem.chem_libs.quantum_chem_library.civector_ops import apply_h_qubit_to_ci as _apply_h_qubit_to_ci
from tyxonq.applications.chem.chem_libs.quantum_chem_library.pyscf_civector import (
    apply_excitation_pyscf as _apply_excitation_pyscf,
    get_civector_pyscf,
    get_energy_and_grad_pyscf
)

from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import apply_op
from tyxonq.numerics import NumericBackend as nb


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
        hamiltonian: Any | None = None
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
        # Optional Hamiltonian apply (callable) or matrix for CI engines
        self.hamiltonian = hamiltonian

        if self.ex_ops is not None:
            self.n_params = (max(self.param_ids) + 1) if (self.param_ids and len(self.param_ids) > 0) else len(self.ex_ops)
        else:
            self.n_params = 0
        # Internal caches for civector ops (per TCC style)
        self._ci_cache = {}
        # Cache keys removed; use centralized cache in statevector_ops
        
        # Instance-level cache for init_state translation (specific to this instance)
        self._init_state_translated = None

    def _build(self, params: Sequence[float]) -> Circuit:
        if self.ex_ops is None or self.n_params == 0:
            return Circuit(self.n_qubits, ops=[])
        return build_ucc_circuit(
            params,
            self.n_qubits,
            self.n_elec_s,
            list(self.ex_ops),
            list(self.param_ids) if self.param_ids is not None else None,
            mode=self.mode,
            init_state=self.init_state,
            decompose_multicontrol=self.decompose_multicontrol,
            trotter=self.trotter,
        )

    @staticmethod
    @lru_cache(maxsize=64)
    def _get_ci_strings_cached(n_qubits: int, n_elec_s_tuple: Tuple, mode: str) -> np.ndarray:
        """Cache CI strings globally across all instances.
        
        Args:
            n_qubits: Number of qubits
            n_elec_s_tuple: Number of electrons as tuple (for hashability)
            mode: Fermionic mode
        
        Returns:
            CI strings array
        """
        from tyxonq.applications.chem.chem_libs.quantum_chem_library.ci_state_mapping import get_ci_strings
        return get_ci_strings(n_qubits, n_elec_s_tuple, mode)
    
    @staticmethod
    @lru_cache(maxsize=64)
    def _get_operator_tensors_cached(n_qubits: int, n_elec_s_tuple: Tuple, ex_ops_tuple: Tuple, mode: str) -> Tuple:
        """Cache operator tensors (fperm, fphase, f2phase) globally across all instances.
        
        Args:
            n_qubits: Number of qubits
            n_elec_s_tuple: Number of electrons as tuple (for hashability)
            ex_ops_tuple: Excitation operators as tuple (for hashability)
            mode: Fermionic mode
        
        Returns:
            Tuple of (ci_strings, fperm, fphase, f2phase)
        """
        from tyxonq.applications.chem.chem_libs.quantum_chem_library.civector_ops import get_operator_tensors
        ex_ops = [tuple(op) for op in ex_ops_tuple]
        return get_operator_tensors(n_qubits, n_elec_s_tuple, ex_ops, mode)
    
    @property
    def ci_strings(self) -> np.ndarray:
        """Get cached CI strings for this instance.
        
        Returns:
            CI strings array (cached globally)
        """
        if not hasattr(self, '_ci_strings_cached'):
            key = (self.n_qubits, tuple(self.n_elec_s), self.mode)
            self._ci_strings_cached = self._get_ci_strings_cached(*key)
        return self._ci_strings_cached
    
    @property
    def operator_tensors_cached(self) -> Tuple | None:
        """Get cached operator tensors for this instance (civector engine only).
        
        Returns:
            Tuple of (ci_strings, fperm, fphase, f2phase) or None if not applicable
        """
        if self.numeric_engine not in ("civector",):
            return None
        
        if not hasattr(self, '_operator_tensors_cached'):
            key = (
                self.n_qubits,
                tuple(self.n_elec_s),
                tuple(tuple(op) for op in (self.ex_ops or [])),
                self.mode
            )
            self._operator_tensors_cached = self._get_operator_tensors_cached(*key)
        return self._operator_tensors_cached
    
    @property
    def init_state_translated(self) -> np.ndarray | None:
        """Get cached translated initial state (instance-specific).
        
        Returns:
            Translated initial state in CI vector format, or None
        """
        if self._init_state_translated is None:
            if self.init_state is not None:
                self._init_state_translated = translate_init_state(
                    self.init_state, self.n_qubits, self.ci_strings
                )
        return self._init_state_translated

    def _state(self, params: Sequence[float]):
        """Get statevector using backend-agnostic approach.
        
        Supports both NumPy and PyTorch backends:
        - NumPy backend: returns NumPy array
        - PyTorch backend: returns differentiable tensor (supports autograd)
        """
        if self.numeric_engine == "statevector":
            # Build CI vector using analytical excitation evolution
            civ = self._civector(params)
            ci_strings = np.asarray(get_ci_strings(self.n_qubits, self.n_elec_s, "fermion"), dtype=np.uint64)
            # Embed CI vector into full statevector
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
        # MPS and fallback paths
        c = self._build(params)
        # Use Circuit.state() API (respects backend and preserves autograd)
        psi = c.state(form="numpy")  # Get as NumPy for embedding
        return self._align_statevector_order(psi)

    def _civector(self, params: Sequence[float]) -> np.ndarray:
        """Build CI vector in CI space (no embedding), following TCC conventions.
        
        Uses cached CI strings and initial state to avoid recomputation.
        """
        if params is None:
            base =  np.zeros(self.n_params, dtype=np.float64) if self.n_params > 0 else np.zeros(0, dtype=np.float64)
        else:
            base = np.asarray(params, dtype=np.float64)

        if self.numeric_engine == "civector":
            ket = get_civector(base, self.n_qubits, self.n_elec_s, list(self.ex_ops or ()), list(self.param_ids or ()), mode=self.mode, init_state=self.init_state_translated)
        elif self.numeric_engine == "civector-large":
            ket = get_civector_nocache(base, self.n_qubits, self.n_elec_s, list(self.ex_ops or ()), list(self.param_ids or ()), mode=self.mode, init_state=self.init_state_translated)
        elif self.numeric_engine == "pyscf":
            ket = get_civector_pyscf(base, self.n_qubits, self.n_elec_s, list(self.ex_ops or ()), list(self.param_ids or ()), mode=self.mode, init_state=self.init_state_translated)
        else:
            #statevector
            ket = get_statevector(base,self.n_qubits, self.n_elec_s, list(self.ex_ops or ()), list(self.param_ids or ()), mode=self.mode, init_state=self.init_state_translated)

        if self.numeric_engine.startswith("civector") or self.numeric_engine == "pyscf":
            civ = ket
        else:
            civ = ket[self.ci_strings]
        # Safely handle potential complex values from statevector conversion
        return np.asarray(np.real(civ), dtype=np.float64)

    # removed energy helpers; statevector path aligns to CI baseline for numeric parity

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
            arr = np.transpose(arr, axes=list(range(n))[::-1])
            return arr.reshape(-1)
        except Exception:
            return psi

    def _prepare_ket(self, params: Sequence[float] | None) -> np.ndarray:
        """Build ket vector using selected numeric engine (共用的初始化逻辑).
        
        Args:
            params: Parameter vector (None → zeros)
        
        Returns:
            ket vector (CI vector or statevector)
        """
        if params is None:
            base = np.zeros(self.n_params, dtype=np.float64) if self.n_params > 0 else np.zeros(0, dtype=np.float64)
        else:
            base = np.asarray(params, dtype=np.float64)

        if self.numeric_engine == "civector":
            return get_civector(base, self.n_qubits, self.n_elec_s, list(self.ex_ops or ()), list(self.param_ids or ()), mode=self.mode, init_state=self.init_state_translated)
        elif self.numeric_engine == "civector-large":
            return get_civector_nocache(base, self.n_qubits, self.n_elec_s, list(self.ex_ops or ()), list(self.param_ids or ()), mode=self.mode, init_state=self.init_state_translated)
        elif self.numeric_engine == "pyscf":
            return get_civector_pyscf(base, self.n_qubits, self.n_elec_s, list(self.ex_ops or ()), list(self.param_ids or ()), mode=self.mode, init_state=self.init_state_translated)
        else:
            # statevector
            return get_statevector(base, self.n_qubits, self.n_elec_s, list(self.ex_ops or ()), list(self.param_ids or ()), mode=self.mode, init_state=self.init_state_translated)

    def energy(self, params: Sequence[float] | None = None) -> float:
        """Compute energy using cached CI strings and initial state."""
        ket = self._prepare_ket(params)
        hket = apply_op(self.hamiltonian, ket)
        # ⚠️ Force explicit conversion to NumPy for np.dot() to avoid NumPy 2.0 __array__ issues
        hket = np.asarray(hket, dtype=np.float64)
        ket = np.asarray(ket, dtype=np.float64)
        return float(np.dot(ket, hket))

    def energy_and_grad(self, params: Sequence[float] | None = None, *, gradient_method: str = "analytic") -> Tuple[float, np.ndarray]:
        """Compute energy and gradient with caching and multiple gradient methods.
        
        Args:
            params: Parameter vector (np.ndarray or torch.Tensor)
            gradient_method: "analytic" (default, fastest) or "fd" (finite difference)
        
        Returns:
            (energy, gradient) tuple
        """
        if params is None:
            base = np.zeros(self.n_params, dtype=np.float64) if self.n_params > 0 else np.zeros(0, dtype=np.float64)
        else:
            # Use backend-aware conversion (handles PyTorch tensor gradient preservation)
            base = nb.asarray(params, dtype=nb.float64)

        if gradient_method == "fd":
            return self._energy_and_grad_fd(base)
        
        # Analytic gradient computation (engine-specific)
        if self.numeric_engine == 'civector':
            e, g = energy_and_grad_civector(
                base, self.hamiltonian, self.n_qubits, self.n_elec_s,
                list(self.ex_ops or ()), list(self.param_ids or ()), 
                mode=self.mode, init_state=self.init_state_translated,
                precomputed_tensors=self.operator_tensors_cached
            )
        elif self.numeric_engine == 'civector-large':
            e, g = energy_and_grad_civector_nocache(
                base, self.hamiltonian, self.n_qubits, self.n_elec_s,
                list(self.ex_ops or ()), list(self.param_ids or ()), 
                mode=self.mode, init_state=self.init_state_translated
            )
        elif self.numeric_engine == 'pyscf':
            e, g = get_energy_and_grad_pyscf(
                base, self.hamiltonian, self.n_qubits, self.n_elec_s,
                list(self.ex_ops or ()), list(self.param_ids or ()), 
                mode=self.mode, init_state=self.init_state_translated
            )
        else:
            # statevector
            e, g = energy_and_grad_statevector(base, self.hamiltonian, self.n_qubits, self.n_elec_s, self.ex_ops, self.param_ids, mode=self.mode, init_state=self.init_state_translated)

        return float(e), np.asarray(g, dtype=np.float64)
    
    def _energy_and_grad_fd(self, base: np.ndarray, fd_eps: float = 1e-5) -> Tuple[float, np.ndarray]:
        """Compute energy and gradient using finite difference.
        
        Args:
            base: Parameter vector
            fd_eps: Finite difference step size
        
        Returns:
            (energy, gradient) tuple
        """
        e0 = self.energy(base)
        g = np.zeros(len(base), dtype=np.float64)
        
        for i in range(len(base)):
            base_plus = base.copy()
            base_plus[i] += fd_eps
            e_plus = self.energy(base_plus)
            
            base_minus = base.copy()
            base_minus[i] -= fd_eps
            e_minus = self.energy(base_minus)
            
            g[i] = (e_plus - e_minus) / (2.0 * fd_eps)
        
        return e0, g


# Compatibility helper for tests (replaces legacy engine_ucc.apply_excitation)
def apply_excitation(state: np.ndarray, n_qubits: int, n_elec_s, ex_op: tuple, mode: str, numeric_engine: str | None = None, engine: str | None = None) -> np.ndarray:
    # eng = (numeric_engine or "statevector").lower()
    is_statevector_input = len(state) == (1 << n_qubits)
    is_statevector_engine = numeric_engine  == "statevector"

    n_elec = int(sum(n_elec_s)) if isinstance(n_elec_s, (tuple, list)) else int(n_elec_s)
    if is_statevector_input and not is_statevector_engine:
        ci_strings = get_ci_strings(n_qubits, n_elec_s, mode)
        state = statevector_to_civector(state, ci_strings)

    if not is_statevector_input and is_statevector_engine:
        ci_strings = get_ci_strings(n_qubits, n_elec_s, mode)
        state = civector_to_statevector(state, n_qubits, ci_strings)

    if numeric_engine == "statevector":
        from tyxonq.applications.chem.chem_libs.quantum_chem_library.statevector_ops import apply_excitation_statevector as _apply_excitation_sv
        
        res_state = _apply_excitation_sv(state, n_qubits, ex_op, mode)
    if numeric_engine == "civector":
        res_state = _apply_excitation_civ(state, n_qubits, n_elec_s, ex_op, mode)
    if numeric_engine == "civector-large":
        res_state =  _apply_excitation_civ_nc(state, n_qubits, n_elec_s, ex_op, mode)
    if numeric_engine == "pyscf":
        res_state = _apply_excitation_pyscf(state, n_qubits, n_elec_s, ex_op, mode)
    if is_statevector_input and not is_statevector_engine:
        return civector_to_statevector(res_state, n_qubits, ci_strings)
    if not is_statevector_input and is_statevector_engine:
        return statevector_to_civector(res_state, ci_strings)
    return res_state




def translate_init_state(init_state, n_qubits, ci_strings):
    """Convert initial state to CI vector format (backend-agnostic).
    
    Supports:
    - Circuit objects (uses Circuit.state() with form="numpy")
    - Statevectors (full 2^n_qubits)
    - CI vectors (len(ci_strings))
    """
    if init_state is None:
        return None
    if isinstance(init_state, Circuit):
        # Use Circuit.state() API instead of StatevectorEngine
        psi0 = init_state.state(form="numpy")  # Explicitly get NumPy for CI conversion
        return np.real(statevector_to_civector(psi0, np.asarray(ci_strings, dtype=int))).astype(np.float64, copy=False)
    arr = np.asarray(init_state)
    if arr.size == (1 << int(n_qubits)):
        return np.real(statevector_to_civector(arr, np.asarray(ci_strings, dtype=int))).astype(np.float64, copy=False)
    if arr.size == len(ci_strings):
        return np.real(arr).astype(np.float64, copy=False)
    return None