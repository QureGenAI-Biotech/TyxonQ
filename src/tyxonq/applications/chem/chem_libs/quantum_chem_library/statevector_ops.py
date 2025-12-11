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
from tyxonq.numerics import NumericBackend as nb

# from tyxonq.libs.circuits_library.qubit_state_preparation import get_init_circuit

def apply_excitation_statevector(statevector, n_qubits, f_idx, mode):
    """Apply a single excitation on a statevector (TCC-style).

    - Apply local unitary ad_a_hc/adad_aa_hc on reversed bit-indices
    - For fermion mode, apply JW Z-string phase vector with sign per qop coefficient
    """
    # 1) apply local fermionic unitary in the computational basis
    psi = nb.asarray(statevector, dtype=nb.complex128).reshape(-1)
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
    # Convert to float64 immediately to preserve gradient chain with PyTorch
    phase = nb.array([sign], dtype=nb.float64)
    for i in range(n):
        if i in z_indices:
            phase = nb.kron(phase, nb.array([1, -1], dtype=nb.float64))
        else:
            phase = nb.kron(phase, nb.array([1, 1], dtype=nb.float64))
    psi = psi * phase
    return psi.reshape(-1)




def get_statevector(
    params,  # Can be np.ndarray or torch.Tensor
    n_qubits: int,
    n_elec_s,
    ex_ops,
    param_ids,
    *,
    mode: str = "fermion",
    init_state=None,
):
    """Get statevector with excitation evolution (backend-agnostic).
    
    Supports automatic differentiation:
    - NumPy backend: params as np.ndarray
    - PyTorch backend: params as torch.Tensor with requires_grad=True
    
    Returns:
        Statevector in current backend format
    """
    # TCC-style: prepare HF initial circuit, then evolve by excitations analytically
    n = int(n_qubits)
    
    if isinstance(init_state, Circuit):
        # 使用 Circuit.state() 方法，转换为 backend 数组
        psi = nb.asarray(init_state.state(form="numpy"), dtype=nb.complex128).reshape(-1)
    elif isinstance(init_state, np.ndarray):
        # init_state is civector or statevector (TCC get_init_circuit semantics)
        ci_strings = get_ci_strings(n, n_elec_s, mode)
        arr = np.asarray(init_state)
        if arr.size == (1 << n):
            psi = nb.asarray(arr, dtype=nb.complex128).reshape(-1)
        elif arr.size == len(ci_strings):
            psi = nb.asarray(civector_to_statevector(arr, n, ci_strings), dtype=nb.complex128).reshape(-1)
        else:
            raise ValueError(f"init_state size {arr.size} incompatible for n_qubits={n} or civector_size={len(ci_strings)}")
    else:
        c0 = get_init_circuit(n_qubits=n_qubits, n_elec_s=n_elec_s, mode=mode)
        # 使用 Circuit.state() 方法
        psi = nb.asarray(c0.state(form="numpy"), dtype=nb.complex128).reshape(-1)

    if ex_ops is None or len(ex_ops) == 0:
        return psi
    ids = param_ids if param_ids is not None else list(range(len(ex_ops)))
    for pid, f_idx in zip(ids, ex_ops):
        # Convert theta to backend tensor/array for autograd support
        theta = nb.asarray(params[pid])
        psi = evolve_excitation(psi, tuple(f_idx), theta, mode)
    # 返回 backend 数组（保持 complex，不强制 .real）
    return psi


def _apply_kqubit_unitary(state, unitary, qubit_idx: list[int], n_qubits: int):
    """Apply k-qubit unitary to statevector (backend-agnostic).
    
    This function now delegates to the unified implementation in quantum_library.kernels.
    Kept for backward compatibility with existing chem code.
    
    Args:
        state: Statevector of shape (2^n_qubits,) - backend array/tensor
        unitary: Unitary matrix of shape (2^k, 2^k)
        qubit_idx: List of target qubit indices
        n_qubits: Total number of qubits
        
    Returns:
        Updated statevector in same backend format as input
    """
    from tyxonq.libs.quantum_library.kernels.statevector import apply_kqubit_unitary
    
    # Delegate to unified implementation，保持 backend 类型
    return apply_kqubit_unitary(state, unitary, qubit_idx, n_qubits, backend=nb)


def evolve_excitation(statevector, f_idx: tuple[int, ...], theta: float, mode: str):
    """Evolve excitation (backend-agnostic).
    
    Args:
        statevector: Backend array/tensor
        f_idx: Fermion indices
        theta: Rotation angle
        mode: Fermion mode
    
    Returns:
        Updated statevector in same backend format
    """
    # Follow TCC evolve_excitation: psi + (1-cos) * F2 psi + sin * F psi
    n_qubits = round(float(np.log2(statevector.shape[0])))
    qubit_idx = [n_qubits - 1 - int(idx) for idx in f_idx]
    if len(qubit_idx) == 2:
        U2 = ad_a_hc2
        U1 = ad_a_hc
    else:
        assert len(qubit_idx) == 4
        U2 = adad_aa_hc2
        U1 = adad_aa_hc
    f2ket = _apply_kqubit_unitary(statevector, U2, qubit_idx, n_qubits)
    fket = apply_excitation_statevector(statevector, n_qubits, f_idx, mode)
    # Match TCC sign convention: sin term carries a negative sign
    # Use backend's cos/sin (supports PyTorch autograd for tensor theta)
    cos_theta = nb.cos(theta)
    sin_theta = nb.sin(theta)
    return statevector + (1.0 - cos_theta) * f2ket + sin_theta * fket



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
    """Compute energy and gradient using backend value_and_grad.
    
    关键设计：
    - _f() 返回 backend tensor/array（保持梯度链）
    - 不在 _f 中转为 Python float（否则断开 autograd）
    - 让 nb.value_and_grad 处理 float 转换和梯度计算
    - 这样 PyTorch backend 就能使用 autograd，NumPy backend 使用有限差分
    """
    from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import apply_op

    def _f(p):
        """计算能量（返回 tensor/array，保持梯度链）。"""
        psi = get_statevector(p, n_qubits, n_elec_s, ex_ops, param_ids, mode=mode, init_state=init_state)
        # 使用新的 apply_op（支持 backend-agnostic）
        hpsi = apply_op(hamiltonian, psi)
        # 计算 <psi|H|psi> = sum(conj(psi) * hpsi)
        # ⚠️ 关键：返回 tensor/array，NOT Python float
        # 这样 PyTorch autograd 才能追踪梯度！
        e = nb.sum(nb.conj(psi) * hpsi)
        # 只取实部，但保持 tensor 类型
        return nb.real(e)

    # value_and_grad 会：
    # - PyTorch backend: 使用 torch.autograd.grad() 自动计算梯度
    # - NumPy backend: 使用有限差分计算梯度
    vag = nb.value_and_grad(_f, argnums=0)
    
    # 使用 nb.asarray() 自动处理类型转换和梯度保留
    params_for_vag = nb.asarray(params, dtype=nb.float64)
    e0, g = vag(params_for_vag)
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
    from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import apply_op
    
    H = get_sparse_operator(qop, n_qubits=n_qubits)
    # 使用新的 apply_op
    hpsi = apply_op(H, psi)
    e = nb.sum(nb.conj(psi) * hpsi)
    return float(nb.real(e))
