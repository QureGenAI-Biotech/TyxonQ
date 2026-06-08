"""H2O 初始化 LUCJ 采样 + SQD 示例。"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tyxonq_matplotlib"))

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from openfermion import FermionOperator  # noqa: E402
from openfermion.transforms import jordan_wigner  # noqa: E402
from pyscf import ao2mo, cc, gto, mcscf, mp, scf  # noqa: E402

from tyxonq.applications.chem.algorithms.lucj import (  # noqa: E402
    LUCJ,
    initialize_lucj_parameters_from_ccsd,
)
from tyxonq.applications.chem.algorithms.sqd import run_sqd_fermion  # noqa: E402
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine  # noqa: E402


DEFAULT_GEOMETRY = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
DEFAULT_BASIS = "6-31g(d,p)"
DEFAULT_ACTIVE_SPACE = (4, 4)
DEFAULT_TOPOLOGY = "square"
N_LAYERS = 1
INIT_MAXITER = 100
DEFAULT_SHOTS = 4096
DEFAULT_NOISE_P = 0.05
DEFAULT_SAMPLES_PER_BATCH = 8
DEFAULT_NUM_BATCHES = 4
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_DIM = 4
DEFAULT_SEED = 7


@dataclass(frozen=True)
class H2OSQDResult:
    hf_energy: float
    fci_energy: float
    lucj_noiseless_energy: float
    lucj_noisy_energy: float
    sqd_energy: float


@dataclass(frozen=True)
class ActiveSpaceData:
    hf_energy: float
    fci_energy: float
    one_body_integrals: np.ndarray
    two_body_integrals: np.ndarray
    core_energy: float
    t1_amplitudes: np.ndarray
    t2_amplitudes: np.ndarray
    h_qubit_op: object
    active_natural_occupations: tuple[float, ...]


def torch_is_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def resolve_initialization_mode(no_optimize: bool) -> bool:
    return False if no_optimize else torch_is_available()


def build_h2o_molecule():
    return gto.M(
        atom=DEFAULT_GEOMETRY,
        basis=DEFAULT_BASIS,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )


def compute_ump2_natural_orbitals(uhf) -> tuple[np.ndarray, np.ndarray]:
    """用 UMP2 one-particle density matrix 选择 H2O active orbitals。"""
    ump2 = mp.UMP2(uhf)
    ump2.kernel()
    rdm1_alpha, rdm1_beta = ump2.make_rdm1()
    mo_alpha, mo_beta = uhf.mo_coeff
    density_ao = mo_alpha @ rdm1_alpha @ mo_alpha.T + mo_beta @ rdm1_beta @ mo_beta.T
    overlap = uhf.mol.intor("int1e_ovlp")

    eigvals, eigvecs = np.linalg.eigh(overlap)
    s_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    s_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    occupations, natural_vectors = np.linalg.eigh(s_sqrt @ density_ao @ s_sqrt)
    order = np.argsort(occupations)[::-1]
    return s_inv_sqrt @ natural_vectors[:, order], occupations[order]


def make_rhf_like_reference_from_natural_orbitals(uhf, natural_orbitals: np.ndarray):
    """让 active-space CCSD 的轨道和 CASCI/FCI 保持一致。"""
    mol = uhf.mol
    reference = scf.RHF(mol)
    reference.kernel()
    reference.mo_coeff = np.asarray(natural_orbitals, dtype=float)
    reference.mo_occ = np.zeros(mol.nao, dtype=float)
    reference.mo_occ[: mol.nelectron // 2] = 2.0
    fock_ao = reference.get_fock()
    reference.mo_energy = np.diag(reference.mo_coeff.T @ fock_ao @ reference.mo_coeff)
    reference._eri = mol.intor("int2e", aosym="s8")
    return reference


def prepare_active_space_data() -> ActiveSpaceData:
    """准备 H2O CAS(4e,4o) 的积分、FCI 和 CCSD 初始化振幅。"""
    n_active_electrons, n_active_orbitals = DEFAULT_ACTIVE_SPACE
    mol = build_h2o_molecule()
    uhf = scf.UHF(mol)
    uhf.kernel()
    natural_orbitals, occupations = compute_ump2_natural_orbitals(uhf)

    inactive_occupied = (mol.nelectron - n_active_electrons) // 2
    active_start = inactive_occupied
    active_stop = active_start + n_active_orbitals
    active_positions = tuple(range(active_start, active_stop))

    casci = mcscf.CASCI(uhf, n_active_orbitals, n_active_electrons)
    fci_energy = float(casci.kernel(natural_orbitals)[0])
    int1e, core_energy = casci.get_h1eff(natural_orbitals)
    int2e = ao2mo.restore("s1", casci.get_h2eff(natural_orbitals), n_active_orbitals)

    rhf_like = make_rhf_like_reference_from_natural_orbitals(uhf, natural_orbitals)
    ccsd = cc.CCSD(rhf_like)
    ccsd.frozen = [idx for idx in range(mol.nao) if idx not in active_positions]
    _, t1_amplitudes, t2_amplitudes = ccsd.kernel()

    h_fermion_op = build_fermion_hamiltonian_from_integrals(int1e, int2e, float(core_energy))
    return ActiveSpaceData(
        hf_energy=float(uhf.e_tot),
        fci_energy=fci_energy,
        one_body_integrals=np.asarray(int1e, dtype=float),
        two_body_integrals=np.asarray(int2e, dtype=float),
        core_energy=float(core_energy),
        t1_amplitudes=np.asarray(t1_amplitudes, dtype=float),
        t2_amplitudes=np.asarray(t2_amplitudes, dtype=float),
        h_qubit_op=jordan_wigner(h_fermion_op),
        active_natural_occupations=tuple(float(x) for x in occupations[active_start:active_stop]),
    )


def build_fermion_hamiltonian_from_integrals(
    int1e: np.ndarray,
    int2e: np.ndarray,
    core_energy: float,
) -> FermionOperator:
    """从 active-space 积分构造 `[alpha | beta]` 顺序的 FermionOperator。"""
    n_orbitals = int1e.shape[0]
    int2e = ao2mo.restore(1, int2e, n_orbitals)
    n_spin_orbitals = 2 * n_orbitals

    h1e = np.zeros((n_spin_orbitals, n_spin_orbitals), dtype=float)
    h2e = np.zeros(
        (n_spin_orbitals, n_spin_orbitals, n_spin_orbitals, n_spin_orbitals),
        dtype=float,
    )
    h1e[:n_orbitals, :n_orbitals] = int1e
    h1e[n_orbitals:, n_orbitals:] = int1e

    for p, q, r, s in product(range(n_spin_orbitals), repeat=4):
        same_spin_pr = (p < n_orbitals) == (s < n_orbitals)
        same_spin_qs = (q < n_orbitals) == (r < n_orbitals)
        if same_spin_pr and same_spin_qs:
            h2e[p, q, r, s] = int2e[
                p % n_orbitals,
                s % n_orbitals,
                q % n_orbitals,
                r % n_orbitals,
            ]

    hamiltonian = FermionOperator.identity() * float(core_energy)
    for p, q in product(range(n_spin_orbitals), repeat=2):
        value = h1e[p, q]
        if abs(value) > 1e-12:
            hamiltonian += FermionOperator(f"{p}^ {q}", value)
    for q, s in product(range(n_spin_orbitals), repeat=2):
        for p, r in product(range(q), range(s)):
            value = h2e[p, q, r, s] - h2e[q, p, r, s]
            if abs(value) > 1e-12:
                hamiltonian += FermionOperator(f"{p}^ {q}^ {r} {s}", value)
    return hamiltonian


def build_lucj_circuit(params):
    """用初始化参数构建 1 层 H2O matrix LUCJ 线路。"""
    n_electrons, n_orbitals = DEFAULT_ACTIVE_SPACE
    return LUCJ(
        n_orbitals,
        n_electrons,
        N_LAYERS,
        DEFAULT_TOPOLOGY,
    ).get_circuit(params)


def sample_noisy_lucj_counts(circuit, *, shots: int, noise_p: float, seed: int) -> dict[str, int]:
    """从 LUCJ statevector 概率层 depolarizing mixture 中抽样。"""
    if shots <= 0:
        raise ValueError("shots must be positive.")
    if not 0.0 <= noise_p <= 1.0:
        raise ValueError("noise_p must be in [0, 1].")

    probabilities = np.asarray(StatevectorEngine().probability(circuit), dtype=float).reshape(-1)
    probabilities = probabilities / np.sum(probabilities)
    mixed_probabilities = mix_depolarizing_probabilities(probabilities, noise_p)

    rng = np.random.default_rng(seed)
    samples = rng.choice(mixed_probabilities.size, size=int(shots), p=mixed_probabilities)
    unique, counts = np.unique(samples, return_counts=True)
    n_qubits = int(np.log2(mixed_probabilities.size))
    pairs = [
        (_index_to_bitstring(int(index), n_qubits), int(count))
        for index, count in zip(unique, counts, strict=True)
    ]
    return dict(sorted(pairs, key=lambda item: (-item[1], item[0])))


def mix_depolarizing_probabilities(probabilities: np.ndarray, noise_p: float) -> np.ndarray:
    """复现旧示例使用的 probability-level depolarizing mixture。"""
    alpha = max(0.0, min(1.0, 4.0 * float(noise_p) / 3.0))
    dim = probabilities.size
    mixed = (1.0 - alpha) * probabilities + alpha * (1.0 / dim)
    return mixed / np.sum(mixed)


def noisy_mixed_energy(noiseless_energy: float, h_qubit_op, noise_p: float) -> float:
    """用 maximally mixed state 能量给出 depolarizing mixture 期望。"""
    alpha = max(0.0, min(1.0, 4.0 * float(noise_p) / 3.0))
    maximally_mixed_energy = float(np.real(h_qubit_op.terms.get((), 0.0)))
    return (1.0 - alpha) * float(noiseless_energy) + alpha * maximally_mixed_energy


def build_initial_occupancies(active_natural_occupations: object) -> tuple[np.ndarray, np.ndarray]:
    """把 spin-summed natural occupation 转成 SQD recovery 的 alpha/beta 占据数。"""
    spin_occupancies = np.clip(np.asarray(active_natural_occupations, dtype=float) / 2.0, 0.0, 1.0)
    return spin_occupancies.copy(), spin_occupancies.copy()


def _index_to_bitstring(index: int, n_qubits: int) -> str:
    return "".join("1" if (index >> (n_qubits - 1 - qubit)) & 1 else "0" for qubit in range(n_qubits))


def run_h2o_sqd(
    *,
    shots: int = DEFAULT_SHOTS,
    noise_p: float = DEFAULT_NOISE_P,
    samples_per_batch: int = DEFAULT_SAMPLES_PER_BATCH,
    num_batches: int = DEFAULT_NUM_BATCHES,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_dim: int = DEFAULT_MAX_DIM,
    seed: int = DEFAULT_SEED,
    init_optimize: bool = True,
    init_maxiter: int | None = None,
) -> H2OSQDResult:
    """运行 H2O 初始化 LUCJ noisy sampling 和 SQD recovery。"""
    n_electrons, n_orbitals = DEFAULT_ACTIVE_SPACE
    nelec = (n_electrons // 2, n_electrons // 2)
    init_maxiter = INIT_MAXITER if init_maxiter is None else int(init_maxiter)

    data = prepare_active_space_data()
    params = initialize_lucj_parameters_from_ccsd(
        data.t2_amplitudes,
        t1=data.t1_amplitudes,
        n_spatial_orbitals=n_orbitals,
        n_layers=N_LAYERS,
        topology=DEFAULT_TOPOLOGY,
        optimize=init_optimize,
        maxiter=init_maxiter,
    )

    circuit = build_lucj_circuit(params)
    lucj_noiseless_energy = float(StatevectorEngine().expval(circuit, data.h_qubit_op))
    lucj_noisy_energy = noisy_mixed_energy(lucj_noiseless_energy, data.h_qubit_op, noise_p)
    noisy_counts = sample_noisy_lucj_counts(circuit, shots=shots, noise_p=noise_p, seed=seed)
    initial_occupancies = build_initial_occupancies(data.active_natural_occupations)

    sqd_result = run_sqd_fermion(
        data.one_body_integrals,
        data.two_body_integrals,
        noisy_counts,
        samples_per_batch=samples_per_batch,
        norb=n_orbitals,
        nelec=nelec,
        nuclear_repulsion_energy=data.core_energy,
        num_batches=num_batches,
        max_iterations=max_iterations,
        symmetrize_spin=True,
        max_dim=max_dim,
        initial_occupancies=initial_occupancies,
        seed=seed,
    )

    energies = np.asarray(
        [
            data.hf_energy,
            data.fci_energy,
            lucj_noiseless_energy,
            lucj_noisy_energy,
            sqd_result.total_energy,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(energies)):
        raise RuntimeError(f"Non-finite energy detected: {energies!r}")

    return H2OSQDResult(
        hf_energy=float(data.hf_energy),
        fci_energy=float(data.fci_energy),
        lucj_noiseless_energy=lucj_noiseless_energy,
        lucj_noisy_energy=lucj_noisy_energy,
        sqd_energy=float(sqd_result.total_energy),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H2O initialized LUCJ sampling + SQD demo")
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--noise-p", type=float, default=DEFAULT_NOISE_P)
    parser.add_argument("--samples-per-batch", type=int, default=DEFAULT_SAMPLES_PER_BATCH)
    parser.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--maxiter", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    init_optimize = resolve_initialization_mode(args.no_optimize)
    result = run_h2o_sqd(
        shots=args.shots,
        noise_p=args.noise_p,
        samples_per_batch=args.samples_per_batch,
        num_batches=args.num_batches,
        max_iterations=args.max_iterations,
        max_dim=args.max_dim,
        seed=args.seed,
        init_optimize=init_optimize,
        init_maxiter=args.maxiter,
    )
    print(f"HF energy: {result.hf_energy:.12f} Ha")
    print(f"FCI energy: {result.fci_energy:.12f} Ha")
    print(f"Initialized LUCJ noiseless energy: {result.lucj_noiseless_energy:.12f} Ha")
    print(f"Initialized LUCJ noisy mixed energy: {result.lucj_noisy_energy:.12f} Ha")
    print(f"SQD energy: {result.sqd_energy:.12f} Ha")


if __name__ == "__main__":
    main()
