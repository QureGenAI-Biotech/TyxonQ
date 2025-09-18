from __future__ import annotations

from typing import List, Tuple, Sequence

from ...core.ir.circuit import Circuit
from openfermion.transforms import jordan_wigner

from tyxonq.libs.hamiltonian_encoding.pauli_io import ex_op_to_fop, reverse_qop_idx
from .utils import unpack_nelec, evolve_pauli_ops, build_multicontrol_ry_ops


def _hf_init_ops(n_qubits: int, n_elec_s: Tuple[int, int], mode: str) -> List[Tuple]:
    ops: List[Tuple] = []
    na, nb = unpack_nelec(n_elec_s)
    if mode in ["fermion", "qubit"]:
        # Legacy engine wire convention: reverse OpenFermion indices to wires
        for i in range(na):
            ops.append(("x", n_qubits - 1 - i))
        for i in range(nb):
            ops.append(("x", n_qubits // 2 - 1 - i))
    else:
        assert mode == "hcb"
        for i in range(na):
            ops.append(("x", n_qubits - 1 - i))
    return ops


def _parity_chain_ops(n_qubits: int, z_indices: List[int], target: int, reverse: bool = False) -> List[Tuple]:
    ops: List[Tuple] = []
    if len(z_indices) == 0:
        return ops
    if not reverse:
        for ii in range(len(z_indices) - 1):
            ops.append(("cx", z_indices[ii], z_indices[ii + 1]))
        ops.append(("cz", z_indices[-1], target))
    else:
        ops.append(("cz", z_indices[-1], target))
        for ii in range(len(z_indices) - 1)[::-1]:
            ops.append(("cx", z_indices[ii], z_indices[ii + 1]))
    return ops


def _evolve_excitation_ops(n_qubits: int, f_idx: Tuple[int, ...], qop, theta: float, mode: str, decompose_multicontrol: bool) -> List[Tuple]:
    # f_idx map to engine wires via reverse ordering
    f_idx = [n_qubits - 1 - idx for idx in f_idx]
    z_indices: List[int] = []
    if mode == "fermion":
        for idx, term in next(iter(qop.terms.keys())):
            if term == "Z":
                z_indices.append(idx)
    ops: List[Tuple] = []
    if len(f_idx) == 2:
        k, l = f_idx
        ops.append(("cx", k, l))
        ops.extend(_parity_chain_ops(n_qubits, z_indices, k, reverse=False))
        # cry(l->k, theta)
        ops.append(("cry", l, k, theta))
        ops.extend(_parity_chain_ops(n_qubits, z_indices, k, reverse=True))
        ops.append(("cx", k, l))
    else:
        # 4-body excitation
        assert len(f_idx) == 4
        k, l, i, j = f_idx
        ops.append(("cx", l, k))
        ops.append(("cx", j, i))
        ops.append(("cx", l, j))
        ops.extend(_parity_chain_ops(n_qubits, z_indices, l, reverse=False))
        ops.extend(build_multicontrol_ry_ops(i, j, k, l, theta, prefer_subcircuit=True))
        ops.extend(_parity_chain_ops(n_qubits, z_indices, l, reverse=True))
        ops.append(("cx", l, j))
        ops.append(("cx", j, i))
        ops.append(("cx", l, k))
    return ops


def build_ucc_circuit(
    params: Sequence[float],
    n_qubits: int,
    n_elec_s: Tuple[int, int],
    ex_ops: Sequence[Tuple],
    param_ids: Sequence[int] | None = None,
    *,
    mode: str = "fermion",
    init_state=None,
    decompose_multicontrol: bool = False,
    trotter: bool = False,
):
    ops: List[Tuple] = []
    # init (HF) or user-provided Circuit
    if init_state is None:
        ops.extend(_hf_init_ops(n_qubits, n_elec_s, mode))
    elif isinstance(init_state, Circuit):
        # start from provided circuit's ops
        ops = list(init_state.ops)
    else:
        # 严格：仅支持 None 或 Circuit。数值路径的向量请在 runtime 侧消费。
        raise NotImplementedError("init_state only supports None or Circuit in circuits_library.ucc")

    # parameterized excitations
    if param_ids is None:
        param_ids = list(range(len(ex_ops)))
    for pid, f_idx in zip(param_ids, ex_ops):
        theta = float(params[pid])
        if trotter:
            # trotter: Pauli-evolution per string
            fop = ex_op_to_fop(f_idx, with_conjugation=True)
            qop = reverse_qop_idx(jordan_wigner(fop), n_qubits)
            for pauli_string, v in qop.terms.items():
                if mode in ["qubit", "hcb"]:
                    pauli_string = [(idx, symbol) for idx, symbol in pauli_string if symbol != "Z"]
                ops.extend(evolve_pauli_ops(tuple(pauli_string), -2 * v.imag * theta))
        else:
            fop = ex_op_to_fop(f_idx, with_conjugation=True)
            qop = reverse_qop_idx(jordan_wigner(fop), n_qubits)
            # Gate-level follows TenCirChem: use 2*theta
            ops.extend(_evolve_excitation_ops(n_qubits, f_idx, qop, 2 * theta, mode, decompose_multicontrol))

    return Circuit(n_qubits, ops=ops)


__all__ = ["build_ucc_circuit"]


