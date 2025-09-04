from typing import Tuple, Union

from openfermion import (
    jordan_wigner,
    bravyi_kitaev,
    binary_code_transform,
    checksum_code,
    parity_code,
    QubitOperator,
    FermionOperator,
)

from tyxonq.libs.hamiltonian_encoding.pauli_io import reverse_fop_idx, reverse_qop_idx


def binary(fermion_operator: FermionOperator, n_modes: int, n_elec: int) -> QubitOperator:
    """Binary transformation with checksum code parity saving.

    Parameters
    ----------
    fermion_operator: FermionOperator
        Fermionic operator in OpenFermion form.
    n_modes: int
        Number of spatial orbitals (spin-orbitals = 2*n_modes in parity mapping contexts).
    n_elec: int
        Total number of electrons (assumed even, split equally by spin).
    """
    return binary_code_transform(fermion_operator, 2 * checksum_code(n_modes // 2, (n_elec // 2) % 2))


def _parity_core(fermion_operator: FermionOperator, n_modes: int) -> QubitOperator:
    return binary_code_transform(fermion_operator, parity_code(n_modes))


def parity(
    fermion_operator: FermionOperator, n_modes: int, n_elec: Union[int, Tuple[int, int]]
) -> QubitOperator:
    """Parity mapping with two-qubit reduction under electron-number conservation.

    Parameters
    ----------
    fermion_operator: FermionOperator
    n_modes: int
        Number of spin-orbitals.
    n_elec: int | (int,int)
        Total electrons or (alpha, beta) split.
    """
    qubit_operator = _parity_core(reverse_fop_idx(fermion_operator, n_modes), n_modes)
    assert n_modes % 2 == 0
    reduction_indices = [n_modes // 2 - 1, n_modes - 1]
    if isinstance(n_elec, int):
        if n_elec % 2 != 0:
            raise ValueError("Specify (alpha,beta) as tuple when total electrons is odd")
        n_elec_s = [n_elec // 2, n_elec // 2]
    else:
        n_elec_s = n_elec
    phase_alpha = (-1) ** n_elec_s[0]
    phase_beta = (-1) ** sum(n_elec_s)
    res = QubitOperator()
    for qop in qubit_operator:
        pauli_string, coeff = next(iter(qop.terms.items()))
        new_pauli_string = []
        for idx, symbol in pauli_string:
            is_alpha = idx <= reduction_indices[0]
            if idx in reduction_indices:
                if symbol in ["X", "Y"]:
                    continue
                else:
                    assert symbol == "Z"
                    coeff *= phase_alpha if is_alpha else phase_beta
                    continue
            if not is_alpha:
                idx -= 1
            new_pauli_string.append((idx, symbol))
        res += QubitOperator(tuple(new_pauli_string), coeff)
    return res


def fop_to_qop(fop: FermionOperator, mapping: str, n_sorb: int, n_elec: Union[int, Tuple[int, int]]) -> QubitOperator:
    mapping = mapping.lower()
    if mapping == "parity":
        qop = parity(fop, n_sorb, n_elec)
    elif mapping in ["jordan-wigner", "jordan_wigner"]:
        qop = reverse_qop_idx(jordan_wigner(fop), n_sorb)
    elif mapping in ["bravyi-kitaev", "bravyi_kitaev"]:
        qop = reverse_qop_idx(bravyi_kitaev(fop, n_sorb), n_sorb)
    else:
        raise ValueError(f"Unknown mapping: {mapping}")
    return qop


__all__ = [
    "binary",
    "parity",
    "fop_to_qop",
]


