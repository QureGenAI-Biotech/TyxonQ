"""原生 SKQD / SqDRIFT：线路生成、样本还原和经典对角化。"""

from .circuit_builder import prepare_slater_reference, build_trotter_krylov_circuits
from .ensemble import CircuitRecord, SKQDEnsemble, merge_skqd_samples, sample_statevector_ensemble
from .driver import SKQDResult, postprocess_skqd
from .qdrift import FermionicGroup, group_fermionic_hamiltonian, build_sqdrift_circuits
from .basis import build_local_natural_orbitals, rotate_integrals, correlations_from_rdms, rdms_to_original_basis
from .gaussian import GaussianFit, fit_gaussian_evolution
from .layout import LayoutResult, optimize_mode_order

__all__ = [
    "prepare_slater_reference", "build_trotter_krylov_circuits",
    "CircuitRecord", "SKQDEnsemble", "merge_skqd_samples", "sample_statevector_ensemble",
    "SKQDResult", "postprocess_skqd",
    "FermionicGroup", "group_fermionic_hamiltonian", "build_sqdrift_circuits",
    "build_local_natural_orbitals", "rotate_integrals", "correlations_from_rdms",
    "rdms_to_original_basis",
    "GaussianFit", "fit_gaussian_evolution", "LayoutResult", "optimize_mode_order",
]
