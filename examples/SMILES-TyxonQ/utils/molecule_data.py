import numpy as np
from tyxonq.applications.chem import molecule
from tyxonq.applications.chem.algorithms.uccsd import UCCSD
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import (
    get_integral_from_hf,
    get_h_sparse_from_integral,
)

# Mapping from user-facing molecule names to TyxonQ molecule objects.
# Callables (functions) are stored as-is and invoked in _get_mol().
MOLECULE_MAP = {
    "H2":   molecule.h2,
    "H3+":  molecule.h3p,
    "H4":   molecule.h4,
    "H5+":  molecule.h5p,
    "H6":   molecule.h6,
    "H8":   molecule.h8,
    "LiH":  molecule.lih,
    "H2O":  molecule.water,
    "HeH+": molecule.hehp,
    "BeH2": molecule.beh2,
    "NH3":  molecule.nh3,
    "BH3":  molecule.bh3,
    "N2":   molecule.n2,
    "CO":   molecule.co,
    "CH4":  molecule.ch4,
}


def _get_mol(name):
    """Resolve a molecule name to a PySCF Mole object."""
    obj = MOLECULE_MAP.get(name)
    if obj is None:
        raise ValueError(
            f"Unknown molecule '{name}'. Available: {list(MOLECULE_MAP.keys())}"
        )
    # Some entries are pre-built Mole objects, others are factory functions.
    if callable(obj) and not hasattr(obj, "atom"):
        return obj()
    return obj


def generate_ucc_operator_pool(ex_ops, op_times):
    """Generate the UCC operator pool by combining excitation ops with time parameters.

    Returns a list of (ex_op_tuple, time) pairs.
    """
    operator_pool = []
    for ex_op in ex_ops:
        for time in op_times:
            operator_pool.append((ex_op, float(time)))
    return operator_pool


def generate_molecule_data(molecules="H2", use_ucc=True):
    """Generate molecule data using TyxonQ.

    Parameters
    ----------
    molecules : str
        Molecule name (e.g. "H2", "LiH", "H4").
    use_ucc : bool
        If True, build a UCC operator pool from excitation operators.

    Returns
    -------
    dict : keyed by molecule name, each value is a dict with:
        - op_pool: list of (ex_op_tuple, time) pairs
        - num_qubits: int
        - hf_state: (na, nb) tuple of alpha/beta electron counts
        - hamiltonian: scipy sparse matrix (without e_core)
        - e_core: float, core energy offset
        - expected_ground_state_E: float, FCI energy
        - ex_ops_raw: list of raw excitation tuples (for FeatureMapper)
    """
    op_times = np.sort(np.logspace(-2, 0, num=8)) / 160

    mol = _get_mol(molecules)
    uccsd = UCCSD(mol, init_method="zeros", run_fci=True)

    # Build sparse Hamiltonian
    int1e, int2e, e_core = get_integral_from_hf(uccsd.hf, uccsd.active_space)
    h_sparse = get_h_sparse_from_integral(int1e, int2e)

    if use_ucc:
        operator_pool = generate_ucc_operator_pool(uccsd.ex_ops, op_times)
    else:
        # Fallback: identity-like pool (each excitation with zero time)
        operator_pool = [(ex_op, 0.0) for ex_op in uccsd.ex_ops]

    molecule_data = {
        molecules: {
            "op_pool": operator_pool,
            "num_qubits": uccsd.n_qubits,
            "hf_state": uccsd.n_elec_s,
            "hamiltonian": h_sparse,
            "e_core": float(e_core),
            "expected_ground_state_E": float(uccsd.e_fci),
            "ex_ops_raw": list(uccsd.ex_ops),
        }
    }

    return molecule_data
