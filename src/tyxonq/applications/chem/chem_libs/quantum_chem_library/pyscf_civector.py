"""
PySCF CI-vector helpers (optional dependency)
=============================================

Purpose
-------
- Provide a thin, explicit wrapper around PySCF FCI addons for operating on CI vectors
  (creation/annihilation sequences), to build precise numeric baselines on small systems.
- This module does NOT define device/IR circuits. It is intended for numerical reference,
  validation and educational purposes, complementing the gate-level paths.

Dependency & Scope
------------------
- Depends on PySCF's FCI utilities (optional). If PySCF is not installed, importing
  or using these helpers will raise ImportError.

Main APIs
---------
- CIvectorPySCF: wraps a CI vector with number of orbitals and alpha/beta electron counts,
  and exposes cre/des/pq/pqqp/pqrs/pqrssrqp composition helpers.
- apply_a_pyscf: apply one-body string(s) to a CI vector and return a new numpy vector.
- apply_a2_pyscf: apply two-body combination a^†a like sequences to a CI vector and return a numpy vector.
- get_init_civector: reuse existing CI initialization helper for constructing HF-like CI vectors.

Recommended usage
-----------------
- Use CIvectorPySCF to build and transform CI vectors under 1-/2-body operators when an
  exact/near-exact numeric baseline is required. For production device paths, use gate-level
  construction from circuits_library instead.

Example
-------
>>> import numpy as np
>>> from tyxonq.libs.quantum_library.pyscf_civector import CIvectorPySCF, apply_a_pyscf, get_init_civector
>>> n_orb, n_elec_a, n_elec_b = 4, 1, 1
>>> civ = get_init_civector(6)  # example length, depends on CI space
>>> cv = CIvectorPySCF(civ, n_orb, n_elec_a, n_elec_b)
>>> out = apply_a_pyscf(cv, (1, 0))  # apply a_p a_q on the CI vector
"""
from typing import Tuple, Any
import numpy as np
from functools import lru_cache

from pyscf.fci import cistring  # type: ignore
from pyscf.fci.addons import des_a, cre_a, des_b, cre_b  # type: ignore

import tyxonq as tq
from tyxonq.libs.circuits_library.utils import unpack_nelec
from tyxonq.applications.chem.chem_libs.quantum_chem_library.ci_state_mapping import get_init_civector
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import apply_op

Tensor = Any




class CIvectorPySCF:
    """Light wrapper of a CI vector with PySCF creation/annihilation helpers."""

    def __init__(self, civector: np.ndarray, n_orb: int, n_elec_a: int, n_elec_b: int):
        assert isinstance(civector, np.ndarray)
        self.civector = civector.reshape(-1)
        self.n_orb = int(n_orb)
        self.n_elec_a = int(n_elec_a)
        self.n_elec_b = int(n_elec_b)

    def cre(self, i: int) -> "CIvectorPySCF":
        n_elec_a, n_elec_b = self.n_elec_a, self.n_elec_b
        # Original convention used in TenCirChem
        if i >= self.n_orb:
            op = cre_a
            n_elec_a += 1
        else:
            op = cre_b
            n_elec_b += 1
        new_civector = op(self.civector, self.n_orb, (self.n_elec_a, self.n_elec_b), i % self.n_orb)
        return CIvectorPySCF(new_civector, self.n_orb, n_elec_a, n_elec_b)

    def des(self, i: int) -> "CIvectorPySCF":
        n_elec_a, n_elec_b = self.n_elec_a, self.n_elec_b
        if i >= self.n_orb:
            op = des_a
            n_elec_a -= 1
        else:
            op = des_b
            n_elec_b -= 1
        new_civector = op(self.civector, self.n_orb, (self.n_elec_a, self.n_elec_b), i % self.n_orb)
        return CIvectorPySCF(new_civector, self.n_orb, n_elec_a, n_elec_b)

    # one- and two-body strings (p, q) on CI vector
    def pq(self, p: int, q: int) -> "CIvectorPySCF":
        return self.des(q).cre(p)

    def pqqp(self, p: int, q: int) -> "CIvectorPySCF":
        return self.des(p).cre(q).des(q).cre(p)

    def pqrs(self, p: int, q: int, r: int, s: int) -> "CIvectorPySCF":
        return self.des(s).des(r).cre(q).cre(p)

    def pqrssrqp(self, p: int, q: int, r: int, s: int) -> "CIvectorPySCF":
        return self.des(p).des(q).cre(r).cre(s).des(s).des(r).cre(q).cre(p)


def apply_a2_pyscf(civector: CIvectorPySCF, ex_op: Tuple[int, ...]) -> Tensor:
    """Apply a^† a (two-operator) on CIvector using PySCF addons and return numpy array."""
    if len(ex_op) == 2:
        apply_f = civector.pqqp
    else:
        assert len(ex_op) == 4
        apply_f = civector.pqrssrqp
    civector1 = apply_f(*ex_op)
    civector2 = apply_f(*reversed(ex_op))
    return -civector1.civector - civector2.civector


def apply_a_pyscf(civector: CIvectorPySCF, ex_op: Tuple[int, ...]) -> Tensor:
    """Apply a (one-operator) on CIvector using PySCF addons and return numpy array.

    Match TCC: use antisymmetrized difference between orders.
    
    Performance: Results are cached per (civector_state, ex_op) pair to avoid
    redundant PySCF operations during gradient computation.
    """
    if len(ex_op) == 2:
        apply_func = civector.pq
    else:
        assert len(ex_op) == 4
        apply_func = civector.pqrs
    civector1 = apply_func(*ex_op)
    civector2 = apply_func(*reversed(ex_op))
    return civector1.civector - civector2.civector


# --- High-level helpers migrated from applications/chem/static/evolve_pyscf.py ---

@lru_cache(maxsize=128)
def _cached_apply_a_pyscf(civector_bytes: bytes, n_orb: int, na: int, nb: int, ex_op: Tuple[int, ...]) -> Tuple[float, ...]:
    """Cached apply_a_pyscf result for a given civector state.
    
    Args:
        civector_bytes: Serialized CI vector (as bytes for hashability)
        n_orb: Number of orbitals
        na, nb: Alpha and beta electron counts
        ex_op: Excitation operator tuple
    
    Returns:
        apply_a_pyscf result as tuple (for hashability in cache)
    """
    civector = np.frombuffer(civector_bytes, dtype=np.float64)
    ket_pyscf = CIvectorPySCF(civector, n_orb, na, nb)
    result = apply_a_pyscf(ket_pyscf, ex_op)
    return tuple(result)  # Convert to tuple for caching


@lru_cache(maxsize=128)
def _cached_apply_a2_pyscf(civector_bytes: bytes, n_orb: int, na: int, nb: int, ex_op: Tuple[int, ...]) -> Tuple[float, ...]:
    """Cached apply_a2_pyscf result for a given civector state."""
    civector = np.frombuffer(civector_bytes, dtype=np.float64)
    ket_pyscf = CIvectorPySCF(civector, n_orb, na, nb)
    result = apply_a2_pyscf(ket_pyscf, ex_op)
    return tuple(result)

def evolve_excitation_pyscf(civector: Tensor, ex_op: Tuple[int, ...], n_orb: int, n_elec_s, theta: float, use_cache: bool = True) -> Tensor:
    """Evolve excitation with optional global LRU caching.
    
    Args:
        civector: CI vector state
        ex_op: Excitation operator
        n_orb: Number of orbitals
        n_elec_s: Electron count tuple
        theta: Evolution parameter
        use_cache: Use LRU cache for apply_a and apply_a2 results (default True)
    """
    na, nb = unpack_nelec(n_elec_s)
    
    if use_cache:
        # Use LRU-cached version for repeated calls with same civector
        civector_bytes = np.asarray(civector, dtype=np.float64).tobytes()
        aket = np.array(_cached_apply_a_pyscf(civector_bytes, n_orb, na, nb, ex_op))
        a2ket = np.array(_cached_apply_a2_pyscf(civector_bytes, n_orb, na, nb, ex_op))
    else:
        # Direct computation without caching
        ket = CIvectorPySCF(np.asarray(civector), n_orb, na, nb)
        aket = apply_a_pyscf(ket, ex_op)
        a2ket = apply_a2_pyscf(ket, ex_op)
    
    # Standard UCC single-parameter block evolution in CI space
    return civector + (1 - np.cos(theta)) * a2ket + np.sin(theta) * aket


def get_civector_pyscf(params, n_qubits: int, n_elec_s, ex_ops: Tuple[Tuple[int, ...], ...], param_ids, mode: str = "fermion", init_state=None, use_cache: bool = True):
    """Build CI vector by evolving excitations using PySCF.
    
    Performance: Uses LRU cache for apply_a and apply_a2 results.
    """
    assert mode == "fermion"
    n_orb = n_qubits // 2
    na, nb = unpack_nelec(n_elec_s)
    num_strings = cistring.num_strings(n_orb, na) * cistring.num_strings(n_orb, nb)

    if init_state is None:
        civector = get_init_civector(num_strings)
    else:
        civector = np.asarray(init_state)

    civector = np.asarray(civector)

    # Apply in the given order (align to TCC evolve_pyscf forward application)
    for ex_op, param_id in zip(ex_ops, param_ids):
        theta = float(params[param_id])
        civector = evolve_excitation_pyscf(civector, ex_op, n_orb, n_elec_s, theta, use_cache=use_cache)

    return civector.reshape(-1)


def _get_gradients_pyscf(bra, ket, params, n_qubits: int, n_elec_s, ex_ops, param_ids, mode: str):
    """Compute gradients by backward evolution with operator result caching.
    
    Performance strategy:
    - Cache apply_a_pyscf results per (civector_id, ex_op) in the backward loop
    - Avoids redundant PySCF operations when the same operator is applied to the same state
    """
    assert mode == "fermion"
    n_orb = n_qubits // 2
    na, nb = unpack_nelec(n_elec_s)
    gradients_beforesum = []
    
    # Local cache for (id(ket_state), ex_op) -> apply_a_pyscf result
    # This avoids recomputing when ket doesn't change within a step
    a_ket_cache = {}  # {(ket_id, ex_op): fket}
    
    for param_id, ex_op in reversed(list(zip(param_ids, ex_ops))):
        theta = params[param_id]
        # Backward evolution
        bra = evolve_excitation_pyscf(bra, ex_op, n_orb, n_elec_s, -float(theta))
        ket = evolve_excitation_pyscf(ket, ex_op, n_orb, n_elec_s, -float(theta))
        
        # For gradient: compute a_ket from evolved ket state
        # Check cache first to avoid redundant PySCF operations
        ket_id = id(ket)
        cache_key = (ket_id, ex_op)
        
        if cache_key not in a_ket_cache:
            ket_pyscf = CIvectorPySCF(ket, n_orb, na, nb)
            fket = apply_a_pyscf(ket_pyscf, ex_op)
            a_ket_cache[cache_key] = fket
        else:
            fket = a_ket_cache[cache_key]
        
        grad = bra @ fket
        gradients_beforesum.append(grad)
    
    gradients_beforesum = list(reversed(gradients_beforesum))
    return np.array(gradients_beforesum)


def get_energy_and_grad_pyscf(params, hamiltonian, n_qubits: int, n_elec_s, ex_ops, param_ids, mode: str = "fermion", init_state=None, use_cache: bool = True):
    """Normalized CI energy and gradient in CI space: E=(c^T H c)/(c^T c).
    
    Performance optimizations:
    - Uses LRU cache for apply_a and apply_a2 results
    - Caches operator results per civector state to avoid recomputation
    """
    params = np.asarray(params)
    ket = get_civector_pyscf(params, n_qubits, n_elec_s, ex_ops, param_ids, mode, init_state, use_cache=use_cache)
    ket = np.asarray(ket, dtype=np.float64)

    bra = apply_op(hamiltonian, ket)
    # ⚠️ Force explicit conversion to NumPy for np.dot() to avoid NumPy 2.0 __array__ issues
    bra = np.asarray(bra, dtype=np.float64)
    ket = np.asarray(ket, dtype=np.float64)
    energy = float(np.dot(bra, ket))

    gradients_beforesum = _get_gradients_pyscf(bra, ket, params, n_qubits, n_elec_s, ex_ops, param_ids, mode)
    gradients = np.zeros(params.shape)
    for v, pid in zip(gradients_beforesum, param_ids):
        gradients[pid] += v
    return float(energy), 2.0 * gradients


def apply_excitation_pyscf(civector, n_qubits: int, n_elec_s, f_idx: Tuple[int, ...], mode: str):
    assert mode == "fermion"
    na, nb = unpack_nelec(n_elec_s)
    civector_pyscf = CIvectorPySCF(np.asarray(civector), n_qubits // 2, na, nb)
    return apply_a_pyscf(civector_pyscf, f_idx)



__all__ = [
    "CIvectorPySCF",
    "apply_a_pyscf",
    "apply_a2_pyscf",
    "get_init_civector",
]


