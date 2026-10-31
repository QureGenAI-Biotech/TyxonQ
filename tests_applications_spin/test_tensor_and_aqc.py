"""DMRG、模拟器、保真度梯度和分段 AQC 的集成验证。"""

import numpy as np
import pytest

pytest.importorskip("quimb")

from tyxonq.applications.spin import (AQCOptions, GroundStateOptions, SpinChainModel,
                                     prepare_ground_state, run_spin_dynamics)
from tyxonq.applications.spin.aqc import FidelityObjective, compress_dynamics
from tyxonq.applications.spin.circuits import (blocks_to_circuit, build_evolution, evolution_blocks,
                                              hva_blocks, initial_blocks, make_spin_ansatz)
from tyxonq.applications.spin.ground_state import dmrg_ground_state
from tyxonq.applications.spin.simulation import mps_state, simulate_observables, statevector


@pytest.mark.parametrize("n", [4, 6, 8, 10])
@pytest.mark.parametrize("jxy,jz", [(1, 0), (1, 1), (.5, 1)])
def test_dmrg_mpo_and_exact_energy(n, jxy, jz):
    from scipy.sparse.linalg import eigsh

    model = SpinChainModel(n, jxy, jz)
    matrix = model.sparse_matrix()
    np.testing.assert_allclose(model.mpo().to_dense(), matrix.toarray(), atol=1e-13)
    psi, energy, converged = dmrg_ground_state(model)
    exact = eigsh(matrix, k=1, which="SA", return_eigenvectors=False,
                  v0=np.random.default_rng(5).normal(size=2**n), tol=1e-12)[0]
    assert abs(energy - exact) < 1e-8
    assert abs(psi.norm() - 1) < 1e-10
    assert converged


def test_degenerate_ground_space():
    model = SpinChainModel(4, 0, 1)
    psi, energy, _ = dmrg_ground_state(model)
    e, vectors = np.linalg.eigh(model.sparse_matrix().toarray())
    ground_space = vectors[:, np.isclose(e, e[0], atol=1e-10)]
    assert ground_space.shape[1] == 2
    assert abs(energy - e[0]) < 1e-8
    assert np.linalg.norm(ground_space.conj().T @ psi.to_dense().reshape(-1)) ** 2 > 1 - 1e-8


@pytest.fixture(scope="module")
def small_ground():
    return prepare_ground_state(SpinChainModel(4), GroundStateOptions(layers=2, maxiter=100))


def test_hva_ground_preparation(small_ground):
    assert small_ground.fidelity >= .999
    assert abs(small_ground.prepared_energy - small_ground.reference_energy) < 1e-5


def test_neel_preparation():
    result = prepare_ground_state(SpinChainModel(4, .5, 1),
                                  GroundStateOptions(layers=2, initial_state="neel", maxiter=100))
    assert np.isfinite(result.fidelity) and 0 <= result.fidelity <= 1 + 1e-10
    assert result.prepared_energy >= result.reference_energy - 1e-10


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_simulators_and_qiskit_match(n):
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    model = SpinChainModel(n, .7, 1)
    blocks = hva_blocks(model, [.13, .42, -.1, .31]) + [("ry", 0, .37), ("rz", 1, np.pi / 2)] + evolution_blocks(model, .2, 3)
    ansatz = make_spin_ansatz(n, blocks)
    circuit = ansatz.bind(ansatz.initial_parameters + np.random.default_rng(20).normal(0, .01, len(ansatz.initial_parameters)))
    native = statevector(circuit)
    mps = mps_state(circuit).to_dense().reshape(-1)
    qc = qiskit.QuantumCircuit(n)
    for op in circuit.ops:
        if op[0] in ("rx", "ry", "rz"):
            getattr(qc, op[0])(op[2], op[1])
        else:
            getattr(qc, op[0])(*op[1:])
    # Qiskit 的 q0 是最低有效位；显式反转张量轴后才能和 TyxonQ 比较。
    other = Statevector(qc).data.reshape((2,) * n).transpose(tuple(reversed(range(n)))).reshape(-1)
    assert abs(np.vdot(native, mps)) ** 2 >= 1 - 1e-8
    assert abs(np.vdot(native, other)) ** 2 >= 1 - 1e-8
    for axis in "XYZ":
        # 单独关闭 SVD 截断检查算符收缩；上面的保真度检查保留默认 cutoff。
        np.testing.assert_allclose(simulate_observables(circuit, axis),
                                   simulate_observables(circuit, axis, "quimb_mps", cutoff=0), atol=1e-10)


def test_ansatz_seed_and_gradient():
    model = SpinChainModel(4, .7, 1)
    blocks = initial_blocks(4) + [("rz", 1, np.pi / 2)] + evolution_blocks(model, .23, 1)
    ansatz = make_spin_ansatz(4, blocks)
    seed = blocks_to_circuit(4, blocks)
    assert abs(np.vdot(statevector(seed), statevector(ansatz.bind(ansatz.initial_parameters)))) ** 2 > 1 - 1e-10
    objective = FidelityObjective(ansatz, mps_state(seed))
    x = ansatz.initial_parameters + np.random.default_rng(1).normal(0, .03, len(ansatz.initial_parameters))
    value, gradient = objective.value_and_grad(x)
    dense_fidelity = abs(np.vdot(statevector(seed), statevector(ansatz.bind(x)))) ** 2
    assert abs(value - (1 - dense_fidelity)) < 1e-10
    numerical = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += 1e-6
        xm[i] -= 1e-6
        numerical[i] = (objective.value_and_grad(xp)[0] - objective.value_and_grad(xm)[0]) / 2e-6
    assert np.linalg.norm(gradient - numerical) / np.linalg.norm(numerical) < 1e-5


def test_segments_warm_start_and_tail(small_ground):
    model = SpinChainModel(4)
    initial = small_ground.blocks + [("rz", 1, np.pi / 2)]
    opts = AQCOptions(segments=((2, 1), (1, 2)), maxiter=40, fidelity_target=.999)
    result = compress_dynamics(model, initial, .2, 4, opts)
    assert len(result.circuits) == 5 and result.compressed_steps == 3
    assert min(record["fidelity"] for record in result.records) >= .99
    np.testing.assert_allclose(result.records[1]["initial_parameters"], result.parameters[1])
    new_ansatz = make_spin_ansatz(4, initial + evolution_blocks(model, .2, 2))
    np.testing.assert_allclose(result.records[2]["initial_parameters"], new_ansatz.initial_parameters)
    expected_tail = result.circuits[3].compose(build_evolution(model, .2, 1))
    assert abs(np.vdot(statevector(result.circuits[4]), statevector(expected_tail))) ** 2 > 1 - 1e-10


def test_unmet_target_is_reported(small_ground):
    result = compress_dynamics(SpinChainModel(4), small_ground.blocks + [("rx", 1, .8)], .6, 2,
                              AQCOptions(segments=((2, 1),), maxiter=1, fidelity_target=1.0))
    assert not result.records[-1]["target_met"]
    assert result.records[-1]["iterations"] <= 1
    assert result.circuits[-1].ops != blocks_to_circuit(4, small_ground.blocks + [("rx", 1, .8)]
                                                       + evolution_blocks(SpinChainModel(4), .6, 2)).ops


def test_complete_small_workflow_without_aqc():
    result = run_spin_dynamics(SpinChainModel(4), steps=3, dt=.2,
                               ground_state_options=GroundStateOptions(layers=2, maxiter=100))
    assert result.arrays["green"].shape == (4, 4)
    assert result.arrays["dsf"].shape == (100, 100)
    assert result.arrays["times"][0] == 0
    assert result.summary["errors"]["aqc"]["green"]["max_abs"] == 0
    assert result.summary["acceptance_passed"]
