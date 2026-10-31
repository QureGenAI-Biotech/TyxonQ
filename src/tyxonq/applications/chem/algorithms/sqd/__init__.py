"""TyxonQ 费米子 SQD 对外 API。"""

from .excitations import enlarge_batch_from_single_excitations, enlarge_batch_from_transitions
from .determinants import (
    DeterminantCIResult, expand_determinants, projected_hamiltonian,
    solve_determinant_subspace, solve_expanded_subspace,
)
from .fermion import (
    SCIResult,
    SCIState,
    spin_square_from_rdms,
    diagonalize_fermionic_hamiltonian,
    run_sqd_fermion,
    solve_sci,
    solve_sci_batch,
)
from .pcm import (PCMActiveSpaceProblem, PCMSCRFCheckpoint, PCMSCRFResult,
                  PCMSCRFSettings, PCMSQDSettings, run_pcm_sqd_scrf)
from .ewf import sci_result_to_solver_result, sci_state_to_cisd
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
    "PCMActiveSpaceProblem", "PCMSCRFCheckpoint", "PCMSCRFResult",
    "PCMSCRFSettings", "PCMSQDSettings", "run_pcm_sqd_scrf",
    "sci_result_to_solver_result", "sci_state_to_cisd",
    "spin_square_from_rdms",
    "enlarge_batch_from_single_excitations",
    "enlarge_batch_from_transitions",
    "DeterminantCIResult",
    "expand_determinants",
    "projected_hamiltonian",
    "solve_determinant_subspace",
    "solve_expanded_subspace",
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
