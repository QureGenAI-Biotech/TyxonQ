"""TyxonQ 费米子 SQD 对外 API。"""

from .fermion import (
    SCIResult,
    SCIState,
    diagonalize_fermionic_hamiltonian,
    run_sqd_fermion,
    solve_sci,
    solve_sci_batch,
)
from .pyscf_solver import SQDFCISolver, as_pyscf_solver, lucj_sampler
from .recovery import recover_configurations
from .samples import (
    bitstring_matrix_to_integers,
    counts_to_arrays,
    generate_counts_bipartite_hamming,
    generate_counts_uniform,
    normalize_counts_dict,
    reverse_bitstring_halves,
    samples_to_arrays,
)
from .subsampling import postselect_by_hamming_right_and_left, subsample

__all__ = [
    "SCIResult",
    "SCIState",
    "SQDFCISolver",
    "as_pyscf_solver",
    "bitstring_matrix_to_integers",
    "counts_to_arrays",
    "diagonalize_fermionic_hamiltonian",
    "generate_counts_bipartite_hamming",
    "generate_counts_uniform",
    "lucj_sampler",
    "normalize_counts_dict",
    "postselect_by_hamming_right_and_left",
    "recover_configurations",
    "reverse_bitstring_halves",
    "run_sqd_fermion",
    "samples_to_arrays",
    "solve_sci",
    "solve_sci_batch",
    "subsample",
]
