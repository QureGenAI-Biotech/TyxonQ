from __future__ import annotations

import numpy as np
from openfermion import jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermion import QubitOperator  # type: ignore
import tyxonq as tq

from tyxonq.applications.chem.constants import (
    ad_a_hc2,
    adad_aa_hc2,
    ad_a_hc,
    adad_aa_hc,
)
from tyxonq.core.ir.circuit import Circuit
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
from tyxonq.libs.hamiltonian_encoding.pauli_io import ex_op_to_fop
from .civector_ops import get_operator_tensors
from .ci_state_mapping import get_ci_strings, civector_to_statevector
from math import comb

# from tyxonq.libs.circuits_library.qubit_state_preparation import get_init_circuit

def apply_excitation_statevector(statevector, n_qubits, f_idx, mode):
    """Apply a single excitation on a statevector (TCC-style).

    - Apply local unitary ad_a_hc/adad_aa_hc on reversed bit-indices
    - For fermion mode, apply JW Z-string phase vector with sign per qop coefficient
    """
    # 1) apply local fermionic unitary in the computational basis
    psi = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    n = int(n_qubits)
    qubit_idx = [n - 1 - int(i) for i in f_idx]
    if len(qubit_idx) == 2:
        U = ad_a_hc
    else:
        assert len(qubit_idx) == 4
        U = adad_aa_hc
    psi = _apply_kqubit_unitary(psi, U, qubit_idx, n)

    if mode != "fermion":
        return psi.reshape(-1)

    # 2) apply Z-string phase from JW mapping of the excitation operator
    fop = ex_op_to_fop(tuple(f_idx))
    qop = jordan_wigner(fop)
    z_indices: list[int] = []
    for idx, term in next(iter(qop.terms.keys())):
        if term != "Z":
            assert idx in f_idx
            continue
        z_indices.append(n - 1 - int(idx))
    sign = 1 if sorted(qop.terms.items())[0][1].real > 0 else -1
    phase = np.array([sign], dtype=np.int8)
    for i in range(n):
        if i in z_indices:
            phase = np.kron(phase, np.array([1, -1], dtype=np.int8))
        else:
            phase = np.kron(phase, np.array([1, 1], dtype=np.int8))
    psi = psi * phase
    return psi.reshape(-1)




def get_statevector(
    params: np.ndarray,
    n_qubits: int,
    n_elec_s,
    ex_ops,
    param_ids,
    *,
    mode: str = "fermion",
    init_state=None,
) -> np.ndarray:
    # TCC-style: prepare HF initial circuit, then evolve by excitations analytically
    n = int(n_qubits)
    
    if isinstance(init_state, Circuit):
        eng = StatevectorEngine()
        psi = np.asarray(eng.state(init_state), dtype=np.complex128).reshape(-1)
    elif isinstance(init_state, np.ndarray):
        # init_state is civector or statevector (TCC get_init_circuit semantics)
        ci_strings = get_ci_strings(n, n_elec_s, mode)
        arr = np.asarray(init_state)
        if arr.size == (1 << n):
            psi = np.asarray(arr, dtype=np.complex128).reshape(-1)
        elif arr.size == len(ci_strings):
            psi = np.asarray(civector_to_statevector(arr, n, ci_strings), dtype=np.complex128).reshape(-1)
        else:
            raise ValueError(f"init_state size {arr.size} incompatible for n_qubits={n} or civector_size={len(ci_strings)}")
    else:
        # c0 = get_init_circuit(n_qubits=n_qubits, n_elec_s=n_elec_s, mode=mode,init_state=init_state,runtime='numeric')
        c0 = get_init_circuit(n_qubits=n_qubits, n_elec_s=n_elec_s, mode=mode)
        eng = StatevectorEngine()
        psi = np.asarray(eng.state(c0), dtype=np.complex128).reshape(-1)

    if ex_ops is None or len(ex_ops) == 0:
        return psi
    ids = param_ids if param_ids is not None else list(range(len(ex_ops)))
    for pid, f_idx in zip(ids, ex_ops):
        theta = float(np.asarray(params[pid]))
        psi = evolve_excitation(psi, tuple(f_idx), theta, mode)
    return psi.real.reshape(-1)


def _apply_kqubit_unitary(state: np.ndarray, unitary: np.ndarray, qubit_idx: list[int], n_qubits: int) -> np.ndarray:
    # Bring target axes to the end (numpy uses LSB-first axes order), then apply local unitary
    k = int(len(qubit_idx))
    if k == 0:
        return state
    axes_all = list(range(n_qubits))
    # Match StatevectorEngine bit order: axis == qubit index (LSB at index 0)
    target_axes = [int(q) for q in qubit_idx]
    keep_axes = [ax for ax in axes_all if ax not in target_axes]
    perm = keep_axes + target_axes
    psi_nd = state.reshape([2] * n_qubits).transpose(perm).reshape(-1, 1 << k)
    U = np.asarray(unitary, dtype=np.complex128).reshape((1 << k), (1 << k))
    out = psi_nd @ U.T
    out = out.reshape([2] * n_qubits).transpose(np.argsort(perm)).reshape(-1)
    return out


def evolve_excitation(statevector: np.ndarray, f_idx: tuple[int, ...], theta: float, mode: str) -> np.ndarray:
    # Follow TCC evolve_excitation: psi + (1-cos) * F2 psi + sin * F psi
    n_qubits = round(np.log2(statevector.shape[0]))
    qubit_idx = [n_qubits - 1 - int(idx) for idx in f_idx]
    if len(qubit_idx) == 2:
        U2 = ad_a_hc2
        U1 = ad_a_hc
    else:
        assert len(qubit_idx) == 4
        U2 = adad_aa_hc2
        U1 = adad_aa_hc
    f2ket = _apply_kqubit_unitary(statevector, U2, qubit_idx, n_qubits)
    fket = apply_excitation_statevector(statevector, n_qubits,f_idx,mode)
    # Match TCC sign convention: sin term carries a negative sign
    return statevector + (1.0 - np.cos(theta)) * f2ket + np.sin(theta) * fket



def get_init_circuit(n_qubits: int, n_elec_s, mode: str) -> Circuit:
    """Construct HF initial circuit consistent with TCC ordering.

    For fermion/qubit mode:
    - First half qubits: alpha spin orbitals
    - Second half qubits: beta spin orbitals
    Occupation set by X gates on highest indices within each block.
    For hcb mode:
    - Occupy the last `na` sites.
    """
    n = int(n_qubits)
    c = Circuit(n, ops=[])
    if isinstance(n_elec_s, (tuple, list)):
        na = int(n_elec_s[0])
        nb = int(n_elec_s[1])
    else:
        ne = int(n_elec_s)
        na = nb = ne // 2
    if mode in ("fermion", "qubit"):
        for i in range(nb):
            c.X(n - 1 - i)
        for i in range(na):
            c.X(n // 2 - 1 - i)
    else:
        assert mode == "hcb"
        for i in range(na):
            c.X(n - 1 - i)
    return c



def energy_and_grad_statevector(
    params: np.ndarray,
    hamiltonian,
    n_qubits: int,
    n_elec_s,
    ex_ops,
    param_ids,
    *,
    mode: str = "fermion",
    init_state=None,
) -> tuple[float, np.ndarray]:
    # Use backend value_and_grad wrapper to match TCC style (torchlib.func.grad_and_value)
    from tyxonq.numerics import NumericBackend as nb

    # Precompute sparse operator once for grad loop (important for numpy backend fallback)
    
    def _f(p):
        psi = get_statevector(p, n_qubits, n_elec_s, ex_ops, param_ids, mode=mode, init_state=init_state)
        e = np.vdot(psi, hamiltonian.dot(psi))
        return float(np.real(e))

    vag = nb.value_and_grad(_f, argnums=0)
    e0, g = vag(np.asarray(params, dtype=np.float64))
    return float(e0), np.asarray(g, dtype=np.float64)


def energy_from_statevector(
    psi: np.ndarray,
    qop: QubitOperator,
    n_qubits: int,
) -> float:
    """Compute <psi|H|psi> using cached sparse operator for the full Hamiltonian.

    Args:
        psi: statevector (complex128) of length 2**n_qubits
        qop: OpenFermion QubitOperator representing full Hamiltonian
        n_qubits: number of qubits
    Returns:
        Real energy value
    """
    H = get_sparse_operator(qop, n_qubits=n_qubits)
    e = np.vdot(psi, H.dot(psi))
    return float(np.real(e))
