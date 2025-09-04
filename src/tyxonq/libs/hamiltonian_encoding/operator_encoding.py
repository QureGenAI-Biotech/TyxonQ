"""
Operator encoding (problem → qubit operators)
=============================================

Purpose
-------
- Provide problem-to-qubit encoding/transforms: fermionic/bosonic operator mappings,
  basis encodings (binary/gray/unary), and integration with optional renormalizer models.
- Live at the "compile-before-gates" layer: produce qubit operators/Hamiltonians and
  encoded bases that later feed into gate-level construction and measurement grouping.

Non-goals
---------
- This module does NOT assume access to full pure quantum states, nor does it implement
  device execution or IR simulation. Numeric baselines (statevectors/CI) live in
  chem/quantum_chem_library or dedicated libs/quantum_library files.

Notes & TODO
------------
- [TODO] Split optional renormalizer integration as a submodule (e.g. operator_encoding.renorm)
  to decouple base encodings from heavy deps.
- [TODO] Add measurement reduction / term grouping helpers close to encoding output, but keep
  provider-specific passes in a separate compiler layer.
"""
from typing import Any, List, Union, Tuple
import numpy as np

from renormalizer.model import Op, OpSum, Model
from renormalizer import BasisHalfSpin, BasisSimpleElectron, BasisMultiElectron, BasisMultiElectronVac, Mps
from renormalizer.model.basis import BasisSet

import tyxonq as tq

from .gray_binary_encoding import get_gray_codes, get_binary_codes, get_encoding

from renormalizer import Mpo  # type: ignore


DOF_SALT = "TCCQUBIT"

# NumPy 2.0 compatibility: renormalizer expects np.product
if not hasattr(np, "product"):
    np.product = np.prod  # type: ignore[attr-defined]


def _require_renormalizer():
    if Op is None:
        raise ImportError("renormalizer is required for renormalizer_encoding utilities")


def check_basis_type(basis: List[Any]):
    _require_renormalizer()
    for b in basis:
        if isinstance(b, (BasisMultiElectronVac,)):
            raise TypeError(f"Unsupported basis: {type(b)}")
        if isinstance(b, BasisMultiElectron) and len(b.dofs) != 2:
            raise ValueError(f"For only two DOFs are allowed in BasisMultiElectron. Got {b}")


def qubit_encode_op(
    terms: Union[List[Any], Any], basis: List[Any], boson_encoding: Union[str, None] = None
) -> Tuple[Any, float]:
    _require_renormalizer()
    check_basis_type(basis)
    if isinstance(terms, Op):
        terms = [terms]
    model = Model(basis, [])
    new_terms = []
    for op in terms:
        terms_e, factor = op.split_elementary(model.dof_to_siteidx)
        opsum_list = []
        for op_e in terms_e:
            opsum = transform_op(op_e, model.dof_to_basis[op_e.dofs[0]], boson_encoding)
            opsum_list.append(opsum)
        new_term = 1
        for opsum in opsum_list:
            new_term = new_term * opsum
        new_term = new_term * factor
        new_terms.extend(new_term)
    identity_terms: List[Any] = []
    non_identity_terms = OpSum()
    for op in new_terms:
        if op.is_identity:
            identity_terms.append(op)
        else:
            non_identity_terms.append(op.squeeze_identity())
    constant = sum([op.factor for op in identity_terms])
    return non_identity_terms.simplify(atol=float(np.finfo(np.float64).eps)), constant


def qubit_encode_op_grouped(
    terms: List[Union[List[Any], Any]], basis: List[Any], boson_encoding: Union[str, None] = None
) -> Tuple[List[Any], float]:
    new_terms = []
    constant_sum = 0.0
    for op in terms:
        opsum, constant = qubit_encode_op(op, basis, boson_encoding)
        new_terms.append(opsum)
        constant_sum += constant
    return new_terms, constant_sum


def qubit_encode_basis(basis: List[Any], boson_encoding: Union[str, None] = None):
    _require_renormalizer()
    spin_basis = []
    for b in basis:
        if isinstance(b, BasisMultiElectron):
            assert b.nbas == 2
            spin_basis.append(BasisHalfSpin(b.dofs))
        elif b.is_phonon:
            if boson_encoding is None:
                new_dofs = [b.dof]
            elif boson_encoding == "unary":
                new_dofs = [(b.dof, f"{DOF_SALT}-{i}") for i in range(b.nbas)]
            else:
                assert boson_encoding.lower() in ["binary", "gray"]
                n_qubits = int(np.ceil(np.log2(b.nbas)))
                new_dofs = [(b.dof, f"{DOF_SALT}-{i}") for i in range(n_qubits)]
            new_basis = [BasisHalfSpin(dof) for dof in new_dofs]
            spin_basis.extend(new_basis)
        else:
            spin_basis.append(BasisHalfSpin(b.dof))
    return spin_basis


def transform_op(op: Any, basis: Any, boson_encoding: Union[str, None] = None):
    _require_renormalizer()
    assert op.factor == 1
    if set(op.split_symbol) == {"I"}:
        return OpSum([op])
    if isinstance(basis, (BasisHalfSpin, BasisSimpleElectron, BasisMultiElectron)):
        if isinstance(basis, BasisMultiElectron):
            assert len(basis.dof) == 2
            new_dof = basis.dofs
        else:
            new_dof = op.dofs[0]
        return transform_op_direct(op, new_dof, basis)
    assert basis.is_phonon
    return transform_op_boson(op, basis, boson_encoding)


def get_elem_qubit_op_direct(row_idx: int, col_idx: int, dof: Any):
    _require_renormalizer()
    if (row_idx, col_idx) == (0, 0):
        return 1 / 2 * (Op("I", dof) + Op("Z", dof))
    elif (row_idx, col_idx) == (0, 1):
        return 1 / 2 * (Op("X", dof) + 1j * Op("Y", dof))
    elif (row_idx, col_idx) == (1, 0):
        return 1 / 2 * (Op("X", dof) - 1j * Op("Y", dof))
    else:
        assert (row_idx, col_idx) == (1, 1)
        return 1 / 2 * (Op("I", dof) - Op("Z", dof))


def transform_op_direct(op: Any, dof: Any, basis: Any):
    _require_renormalizer()
    if basis.nbas != 2:
        raise ValueError("Direct encoding only support two level basis")
    mat = basis.op_mat(op)
    ret = OpSum()
    for row_idx, col_idx in zip(*np.nonzero(mat)):
        ret += mat[row_idx, col_idx] * get_elem_qubit_op_direct(row_idx, col_idx, dof)
    return ret.simplify(atol=float(np.finfo(np.float64).eps))


def get_elem_qubit_op_unary(row_idx: int, col_idx: int, new_dofs: List[Any]):
    _require_renormalizer()
    if row_idx == col_idx:
        dof_list = [new_dofs[row_idx]]
        return 1 / 2 * (Op("I", dof_list) - Op("Z", dof_list))
    else:
        des = 1 / 2 * (Op("X", new_dofs[col_idx]) + 1j * Op("Y", new_dofs[col_idx]))
        cre = 1 / 2 * (Op("X", new_dofs[row_idx]) - 1j * Op("Y", new_dofs[row_idx]))
        if new_dofs[row_idx] < new_dofs[col_idx]:
            return cre * des
        else:
            return des * cre


def transform_op_boson_unary(op: Any, dof: Any, basis: Any):
    _require_renormalizer()
    new_dofs = [(dof, f"{DOF_SALT}-{i}") for i in range(basis.nbas - 1, -1, -1)]
    mat = basis.op_mat(op)
    ret = OpSum()
    for row_idx, col_idx in zip(*np.nonzero(mat)):
        ret += mat[row_idx, col_idx] * get_elem_qubit_op_unary(row_idx, col_idx, new_dofs)
    return ret.simplify(atol=float(np.finfo(np.float64).eps))


def get_elem_qubit_op_binary(row_idx: int, col_idx: int, new_dofs: List[Any], code_strs: List[str]):
    _require_renormalizer()
    n_qubits = len(new_dofs)
    if row_idx == col_idx:
        op_list = []
        for i in range(n_qubits):
            dof = new_dofs[i]
            if code_strs[row_idx][i] == "0":
                new_op = 1 / 2 * (Op("I", dof) + Op("Z", dof))
            else:
                new_op = 1 / 2 * (Op("I", dof) - Op("Z", dof))
            op_list.append(new_op)
        return OpSum.product(op_list)
    else:
        code1 = code_strs[row_idx]
        code2 = code_strs[col_idx]
        op_list = []
        for i in range(n_qubits):
            dof = new_dofs[i]
            if code1[i] == code2[i]:
                if code1[i] == "0":
                    new_op = 1 / 2 * (Op("I", dof) + Op("Z", dof))
                else:
                    new_op = 1 / 2 * (Op("I", dof) - Op("Z", dof))
            else:
                if code1[i] + code2[i] == "01":
                    new_op = 1 / 2 * (Op("X", dof) + 1j * Op("Y", dof))
                else:
                    new_op = 1 / 2 * (Op("X", dof) - 1j * Op("Y", dof))
            op_list.append(new_op)
        return OpSum.product(op_list)


def transform_op_boson_binary(op: Any, dof: Any, basis: Any, encoding: str):
    _require_renormalizer()
    n_qubits = (basis.nbas - 1).bit_length()
    new_dofs = [(dof, f"{DOF_SALT}-{i}") for i in range(n_qubits)]
    code_strs = get_gray_codes(n_qubits) if encoding == "gray" else get_binary_codes(n_qubits)
    mat = basis.op_mat(op)
    ret = OpSum()
    for row_idx, col_idx in zip(*np.nonzero(mat)):
        ret += mat[row_idx, col_idx] * get_elem_qubit_op_binary(row_idx, col_idx, new_dofs, code_strs)
    return ret.simplify(atol=float(np.finfo(np.float64).eps))


def transform_op_boson(op, basis, encoding=None):
    _require_renormalizer()
    assert op.factor == 1
    if encoding is None:
        return transform_op_direct(op, op.dofs[0], basis)
    elif encoding == "unary":
        return transform_op_boson_unary(op, op.dofs[0], basis)
    elif encoding and encoding.lower() in ["binary", "gray"]:
        return transform_op_boson_binary(op, op.dofs[0], basis, encoding.lower())
    else:
        raise ValueError(f"Encoding '{encoding}' not supported")


def get_init_circuit(model_ref, model, boson_encoding, init_condition):
    _require_renormalizer()
    for k, v in init_condition.items():
        basis = model_ref.dof_to_basis[k]
        if not isinstance(v, int):
            if (
                isinstance(basis, BasisHalfSpin)
                and v.shape == (2, 2)
                and np.allclose(np.eye(2), v @ v.T.conj)
                and np.allclose(np.eye(2), v.T.conj @ v)
            ):
                continue
            else:
                return get_init_circuit_general(model_ref, model, boson_encoding, init_condition)
    circuit = tq.Circuit(len(model.basis))
    for k, v in init_condition.items():
        basis = model_ref.dof_to_basis[k]
        if isinstance(basis, BasisHalfSpin):
            if v == 1:
                circuit.X(model.dof_to_siteidx[k])
            elif getattr(v, "shape", None) == (2, 2):
                circuit.ANY(idx, unitary=v)
            else:
                assert v == 0
        elif isinstance(basis, BasisMultiElectron):
            if v == 1:
                idx = model.dof_to_siteidx[basis.dofs]
                circuit.X(idx)
            else:
                assert v == 0
        else:
            assert basis.is_phonon
            if boson_encoding is None:
                assert v in [0, 1]
                target = [v]
            elif boson_encoding == "unary":
                target = [0] * len(basis.nbas)
                target[len(basis.nbas) - v - 1] = 1
            elif boson_encoding == "binary":
                target = get_binary_codes((basis.nbas - 1).bit_length())[v]
            else:
                assert boson_encoding == "gray"
                target = get_gray_codes((basis.nbas - 1).bit_length())[v]
            for i, t in enumerate(target):
                if t == 1:
                    circuit.X(model.dof_to_siteidx[(k, f"{DOF_SALT}-{i}")])
    return circuit


def get_init_circuit_general(model_ref, model, boson_encoding, init_condition):
    _require_renormalizer()
    mps = Mps.hartree_product_state(model_ref, init_condition)
    mps_state = mps.todense()
    subspace_idx = get_subspace_idx(model_ref.basis, boson_encoding)
    assert len(mps_state) == len(subspace_idx)
    n_qubits = len(model.basis)
    state = np.zeros(1 << n_qubits, dtype=np.complex128)
    state[subspace_idx] = mps_state
    return state


def get_subspace_idx(basis_list, boson_encoding):
    _require_renormalizer()
    subspace_idx = [""]
    for basis in basis_list:
        if isinstance(basis, (BasisSimpleElectron, BasisMultiElectron, BasisHalfSpin)):
            new_idx = "01"
        else:
            new_idx = get_encoding(basis.nbas, boson_encoding)
        new_subspace_idx = []
        for idx1 in subspace_idx:
            for idx2 in new_idx:
                new_subspace_idx.append(idx1 + idx2)
        subspace_idx = new_subspace_idx
    return [int(i, base=2) for i in subspace_idx]


def get_dense_operator(basis: List[Any], terms: List[Any]):
    _require_renormalizer()
    if Mpo is None:
        raise ImportError("renormalizer.Mpo is required for get_dense_operator")
    return Mpo(Model(basis, []), terms).todense()


def get_init_statevector(model_ref, model, boson_encoding, init_condition):
    return get_init_circuit_general(model_ref, model, boson_encoding, init_condition)

__all__ = [
    "qubit_encode_op",
    "qubit_encode_op_grouped",
    "qubit_encode_basis",
    "get_init_circuit",
    "get_init_statevector",
    "transform_op",
    "get_dense_operator",
]


