"""独立小矩阵验证 SKQD，不依赖 Qiskit/ffsim 的线路或模拟器。"""

import numpy as np
import pytest
from openfermion import get_sparse_operator
from scipy.linalg import expm

from tyxonq.applications.chem.algorithms.skqd import (
    CircuitRecord, SKQDEnsemble, prepare_slater_reference,
    build_trotter_krylov_circuits, merge_skqd_samples, sample_statevector_ensemble,
    postprocess_skqd,
)
from tyxonq.applications.chem.chem_libs.fermionic_operators import electronic_fermion_operator
from tyxonq.core.ir.circuit import Circuit
from tyxonq.applications.chem.algorithms.skqd import (
    group_fermionic_hamiltonian, build_sqdrift_circuits, rotate_integrals,
    build_local_natural_orbitals, fit_gaussian_evolution, optimize_mode_order,
    correlations_from_rdms,
)
from tyxonq.applications.chem.algorithms.skqd.gaussian import _network_and_gradient
from tyxonq.applications.chem.algorithms.sqd import (
    expand_determinants, solve_determinant_subspace, solve_expanded_subspace,
)


def model(n=2):
    """两个相互耦合轨道，其中一个存在 onsite 相互作用。"""
    h = np.diag(np.linspace(-0.8, 0.6, n))
    for i in range(n - 1):
        h[i, i + 1] = h[i + 1, i] = -0.7
    g = np.zeros((n,) * 4)
    g[0, 0, 0, 0] = 1.9
    return h, g


def state(circuit):
    return np.asarray(circuit.state(engine="statevector", backend="numpy", form="numpy"))


def aligned_error(actual, expected):
    """消去允许的整体相位，保留所有相对相位误差。"""
    overlap = np.vdot(expected, actual)
    return np.linalg.norm(actual - expected * (overlap / abs(overlap)))


def test_reference_and_trotter_against_expm():
    h, g = model()
    initial = prepare_slater_reference(2, (1, 1))
    psi0 = state(initial)
    assert np.argmax(abs(psi0)) == int("1010", 2)
    hmat = get_sparse_operator(electronic_fermion_operator(h, g), n_qubits=4).toarray()
    dt = 0.3
    exact = expm(-1j * dt * hmat) @ psi0
    errors = []
    for substeps in (1, 2, 4):
        batch = build_trotter_krylov_circuits(h, g, 2, (1, 1), time_step=dt,
                                            krylov_indices=[0, 1], trotter_substeps=substeps)
        assert aligned_error(state(batch.circuits[0]), psi0) < 1e-12
        errors.append(aligned_error(state(batch.circuits[1]), exact))
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2
    assert len(initial.ops) == 2


def test_mapping_before_merge():
    # old q0 和 q2 占据；不同线路拥有不同的模式和测量顺序。
    a = CircuitRecord("a", 0, 0, mode_permutation=(0, 1, 2, 3), measurement_map=(0, 1, 2, 3))
    b = CircuitRecord("b", 1, 1, mode_permutation=(2, 0, 3, 1), measurement_map=(3, None, 1, 0, 2))
    batch = SKQDEnsemble([Circuit(4), Circuit(4)], [a, b], 2, (1, 1), "test")
    assert merge_skqd_samples({"a": {"0101": 2}, "b": {"10001": 5}}, batch) == {"0101": 7}
    b.basis_id = "rotated"
    with pytest.raises(ValueError, match="different orbital bases"):
        merge_skqd_samples({"a": {"0101": 2}, "b": {"10001": 5}}, batch)


def test_sqd_closed_loop_energy_offset():
    h, g = model()
    ensemble = build_trotter_krylov_circuits(h, g, 2, (1, 1), time_step=0.3)
    samples = sample_statevector_ensemble(ensemble, shots=300, seed=12)
    result = postprocess_skqd(h, g, samples, ensemble, samples_per_batch=100,
                             max_iterations=2, energy_offset=1.23)
    from pyscf.fci import direct_spin1
    energy, _ = direct_spin1.kernel(h, g, 2, (1, 1))
    assert result.energy == pytest.approx(energy, abs=1e-10)
    assert result.total_energy == pytest.approx(energy + 1.23, abs=1e-10)
    r = result.sci_result
    recomputed = np.einsum("pq,qp", h, r.rdm1) + 0.5 * np.einsum("pqrs,pqrs", g, r.rdm2)
    assert recomputed == pytest.approx(result.energy, abs=1e-10)
    assert result.statistics["shots"] == 8 * 300


def random_integrals(n, seed=43):
    """独立生成实数 eightfold 积分，不仅覆盖 onsite 特例。"""
    from pyscf import ao2mo
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, n))
    pair = n * (n + 1) // 2
    g = rng.normal(size=(pair, pair))
    return (h + h.T) / 2, ao2mo.restore(1, (g + g.T) / 2, n)


def sector_determinants(n, nelec):
    from pyscf.fci import cistring
    return tuple(sorted(int(a) + (int(b) << n)
                        for a in cistring.make_strings(range(n), nelec[0])
                        for b in cistring.make_strings(range(n), nelec[1])))


@pytest.mark.parametrize("n,nelec", [(2, (1, 1)), (3, (2, 1)), (4, (2, 2))])
def test_groups_and_strict_solver_against_pyscf(n, nelec):
    from openfermion import FermionOperator
    from pyscf.fci import direct_spin1, cistring
    h, g = random_integrals(n)
    groups = group_fermionic_hamiltonian(h, g)
    restored = sum((item.weight * item.operator for item in groups), FermionOperator())
    delta = get_sparse_operator(restored - electronic_fermion_operator(h, g), n_qubits=2 * n)
    assert not delta.nnz or np.max(abs(delta.data)) < 1e-10
    # 全固定电子数空间验证能量，与 PySCF 独立积分路径对照。
    dets = sector_determinants(n, nelec)
    result = solve_determinant_subspace(h, g, dets, n, nelec)
    exact, _ = direct_spin1.kernel(h, g, n, nelec)
    assert result.energy == pytest.approx(exact, abs=1e-9)
    assert result.residual_norm < 1e-8
    alpha = {int(x): i for i, x in enumerate(cistring.make_strings(range(n), nelec[0]))}
    beta = {int(x): i for i, x in enumerate(cistring.make_strings(range(n), nelec[1]))}
    ci = np.zeros((len(alpha), len(beta)))
    for determinant, coefficient in zip(result.determinants, result.amplitudes):
        ci[alpha[determinant & ((1 << n) - 1)], beta[determinant >> n]] = coefficient
    one, two = direct_spin1.make_rdm12s(ci, n, nelec)
    for actual, expected in zip(result.rdm1s + result.rdm2s, tuple(one) + tuple(two)):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    recomputed = np.einsum("pq,qp", h, result.rdm1) + 0.5 * np.einsum("pqrs,pqrs", g, result.rdm2)
    assert recomputed == pytest.approx(result.energy, abs=1e-9)
    corr = correlations_from_rdms(result.rdm1s, result.rdm2s)
    assert corr["density"].sum() == pytest.approx(sum(nelec) ** 2, abs=1e-10)
    assert corr["spin_dot"].sum() == pytest.approx(result.spin_square(), abs=1e-10)


def test_qdrift_native_sequence_and_number_conservation():
    h, g = random_integrals(3)
    kwargs = dict(time_step=0.08, krylov_indices=[1, 2], num_sampled_groups=5,
                  num_randomizations=2, seed=61)
    ensemble = build_sqdrift_circuits(h, g, 3, (2, 1), **kwargs)
    replay = build_sqdrift_circuits(h, g, 3, (2, 1), **kwargs)
    groups = group_fermionic_hamiltonian(h, g)
    initial = state(prepare_slater_reference(3, (2, 1)))
    allowed = [int(format(d, "06b")[::-1], 2) for d in sector_determinants(3, (2, 1))]
    for circuit, record, repeated in zip(ensemble.circuits, ensemble.records, replay.records):
        assert record.sampled_group_ids == repeated.sampled_group_ids
        expected = initial.copy()
        for group_id in record.sampled_group_ids:
            matrix = get_sparse_operator(groups[group_id].operator, n_qubits=6).toarray()
            expected = expm(-1j * record.diagnostics["delta"] * matrix) @ expected
        actual = state(circuit)
        assert aligned_error(actual, expected) < 1e-10
        assert np.sum(np.abs(actual[allowed]) ** 2) == pytest.approx(1, abs=1e-10)
    assert ensemble.records[0].sampled_group_ids != ensemble.records[1].sampled_group_ids


def test_qdrift_average_channel_converges():
    h, g = model()
    groups = group_fermionic_hamiltonian(h, g)
    weights = np.array([item.weight for item in groups])
    matrices = [get_sparse_operator(item.operator, n_qubits=4).toarray() for item in groups]
    hmat = sum(w * a for w, a in zip(weights, matrices))
    psi = state(prepare_slater_reference(2, (1, 1)))
    exact = expm(-0.1j * hmat) @ psi
    target = np.outer(exact, exact.conjugate())
    errors = []
    # 精确累加单步随机通道并迭代 M 次，避免 Monte Carlo 波动制造伪回归。
    for length in (2, 4, 8):
        rho = np.outer(psi, psi.conjugate())
        unitaries = [expm(-1j * 0.1 * weights.sum() / length * a) for a in matrices]
        for _ in range(length):
            rho = sum(w / weights.sum() * (u @ rho @ u.conj().T) for w, u in zip(weights, unitaries))
        errors.append(np.linalg.norm(rho - target))
    assert errors[2] < errors[1] < errors[0]


def test_fermionic_permutation_on_superposed_reference():
    h, g = random_integrals(3)
    rotation = expm(-0.27j * h)
    reference = prepare_slater_reference(3, (2, 1), orbital_rotation=rotation)
    kwargs = dict(time_step=0.1, krylov_indices=[1], num_sampled_groups=7,
                  num_randomizations=1, initial_state=reference)
    plain = build_sqdrift_circuits(h, g, 3, (2, 1), **kwargs)
    # 独立验证自定义 reference 的 unitary 缓存确实被复制，而非两条线路都遗漏旋转。
    expected = state(reference)
    groups = group_fermionic_hamiltonian(h, g)
    for group_id in plain.records[0].sampled_group_ids:
        matrix = get_sparse_operator(groups[group_id].operator, n_qubits=6).toarray()
        expected = expm(-1j * plain.records[0].diagnostics["delta"] * matrix) @ expected
    assert aligned_error(state(plain.circuits[0]), expected) < 1e-10
    permutation = (4, 1, 5, 0, 3, 2)
    mapped = build_sqdrift_circuits(h, g, 3, (2, 1), mode_permutation=permutation, **kwargs)
    old = abs(state(plain.circuits[0])) ** 2
    new = abs(state(mapped.circuits[0])) ** 2
    remapped = np.zeros_like(old)
    for i, probability in enumerate(new):
        text = format(i, "06b")
        logical = "".join(text[permutation[p]] for p in range(6))
        remapped[int(logical, 2)] = probability
    np.testing.assert_allclose(remapped, old, atol=1e-10)


def test_local_natural_orbitals_and_interaction_rotation():
    h, g = model(3)
    rho = np.array([[1.3, 0.1, 0.2], [0.1, 0.8, 0.15], [0.2, 0.15, 0.9]])
    c = build_local_natural_orbitals(rho, [[0, 1, 2]])
    np.testing.assert_allclose(c.T @ c, np.eye(3), atol=1e-12)
    transformed = c.T @ rho @ c
    np.testing.assert_allclose(transformed, np.diag(np.diag(transformed)), atol=1e-12)
    hr, gr = rotate_integrals(h, g, c)
    from pyscf.fci import direct_spin1
    assert direct_spin1.kernel(hr, gr, 3, (1, 1))[0] == pytest.approx(
        direct_spin1.kernel(h, g, 3, (1, 1))[0], abs=1e-10)
    with pytest.raises(ValueError, match="not density-density"):
        build_trotter_krylov_circuits(hr, gr, 3, (1, 1), time_step=0.1)
    ensemble = build_trotter_krylov_circuits(hr, gr, 3, (1, 1), time_step=0.1,
                                           interaction_rotation=c, krylov_indices=[1], trotter_substeps=4)
    matrix = get_sparse_operator(electronic_fermion_operator(hr, gr), n_qubits=6).toarray()
    original_matrix = get_sparse_operator(electronic_fermion_operator(h, g), n_qubits=6).toarray()
    np.testing.assert_allclose(np.linalg.eigvalsh(matrix), np.linalg.eigvalsh(original_matrix), atol=1e-10)
    exact = expm(-0.1j * matrix) @ state(prepare_slater_reference(3, (1, 1)))
    # 先对照相同 Trotter 乘积，分开检查线路实现误差与物理分裂误差。
    h_one = get_sparse_operator(electronic_fermion_operator(hr, np.zeros_like(gr)), n_qubits=6).toarray()
    half = expm(-0.0125j * (matrix - h_one))
    step = half @ expm(-0.025j * h_one) @ half
    trotter_exact = np.linalg.matrix_power(step, 4) @ state(prepare_slater_reference(3, (1, 1)))
    assert aligned_error(state(ensemble.circuits[0]), trotter_exact) < 1e-10
    finer = build_trotter_krylov_circuits(hr, gr, 3, (1, 1), time_step=0.1,
                                        interaction_rotation=c, krylov_indices=[1], trotter_substeps=8)
    error = aligned_error(state(ensemble.circuits[0]), exact)
    fine_error = aligned_error(state(finer.circuits[0]), exact)
    assert 3.9 < error / fine_error < 4.1
    assert fine_error < 1e-5
    # 用独立 FCI 的 RDM 检查变回原轨道后的物理关联，不能直接比较两个基底的对角线。
    from tyxonq.applications.chem.algorithms.skqd import rdms_to_original_basis
    _, original_ci = direct_spin1.kernel(h, g, 3, (1, 1))
    _, rotated_ci = direct_spin1.kernel(hr, gr, 3, (1, 1))
    one, two = direct_spin1.make_rdm12s(original_ci, 3, (1, 1))
    rotated_one, rotated_two = direct_spin1.make_rdm12s(rotated_ci, 3, (1, 1))
    restored_one, restored_two = rdms_to_original_basis(rotated_one, rotated_two, c)
    np.testing.assert_allclose(restored_one, one, atol=1e-9)
    np.testing.assert_allclose(restored_two, two, atol=1e-9)


def test_gaussian_analytic_gradient_and_fixed_depth_emission():
    from scipy.optimize._numdiff import approx_derivative
    h, g = model(3)
    target = expm(-0.2j * h)
    pairs = ((0, 1), (1, 2), (0, 1))
    parameters = np.random.default_rng(4).normal(size=3 + 2 * len(pairs))
    _, gradient, _ = _network_and_gradient(parameters, target, pairs)
    numeric = approx_derivative(lambda x: _network_and_gradient(x, target, pairs)[0], parameters).ravel()
    np.testing.assert_allclose(gradient, numeric, atol=1e-8)
    fit = fit_gaussian_evolution(target, depth=6)
    assert fit.error < 1e-9 and fit.error < fit.initial_error
    fitted = build_trotter_krylov_circuits(h, g, 3, (1, 1), time_step=0.2,
                                        krylov_indices=[1], gaussian_fit=fit)
    exact = build_trotter_krylov_circuits(h, g, 3, (1, 1), time_step=0.2, krylov_indices=[1])
    assert aligned_error(state(fitted.circuits[0]), state(exact.circuits[0])) < 1e-4


def test_layout_and_strict_excitation_rank():
    result = optimize_mode_order([(0, 5), (1, 4)], 6, time_limit=2)
    assert sorted(result.permutation) == list(range(6))
    assert result.objective < 3
    seed = 3 + (3 << 4)
    sizes = []
    spaces = []
    for rank in (1, 2, 3):
        expanded = expand_determinants([seed], 4, (2, 2), max_rank=rank)
        assert all((d ^ seed).bit_count() <= 2 * rank for d in expanded)
        sizes.append(len(expanded))
        spaces.append(set(expanded))
    assert sizes == [9, 27, 35]
    assert spaces[0] < spaces[1] < spaces[2]
    with pytest.raises(ValueError, match="no truncation"):
        expand_determinants([seed], 4, (2, 2), max_rank=3, max_determinants=10)
    h, g = random_integrals(4)
    energies = [solve_expanded_subspace(h, g, [seed], 4, (2, 2), max_rank=r).energy for r in (1, 2, 3)]
    assert energies[0] >= energies[1] >= energies[2]


def test_no_qiskit_runtime_and_invalid_hamiltonian(monkeypatch):
    import builtins
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"qiskit", "ffsim", "qiskit_fermions"}:
            raise AssertionError(f"unexpected runtime dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    h, g = model()
    batch = build_sqdrift_circuits(h, g, 2, (1, 1), time_step=0.2, num_randomizations=2)
    samples = sample_statevector_ensemble(batch, shots=100)
    assert postprocess_skqd(h, g, samples, batch, max_iterations=1).statistics["shots"] == 800
    with pytest.raises(ValueError, match="real restricted"):
        build_sqdrift_circuits(h.astype(complex), g, 2, (1, 1), time_step=0.2)


def test_ensemble_roundtrip_keeps_custom_unitary_matrices(tmp_path):
    h, g = model()
    reference = prepare_slater_reference(2, (1, 1), orbital_rotation=expm(-0.2j * h))
    ensemble = build_trotter_krylov_circuits(h, g, 2, (1, 1), time_step=0.1,
                                           krylov_indices=[0, 1], initial_state=reference)
    path = tmp_path / "ensemble.json"
    ensemble.save(path)
    loaded = SKQDEnsemble.load(path)
    for before, after in zip(ensemble.circuits, loaded.circuits):
        np.testing.assert_allclose(state(before), state(after), atol=1e-12)
    assert sample_statevector_ensemble(ensemble, seed=5) == sample_statevector_ensemble(loaded, seed=5)


def test_f2q_integrated_and_filters_are_explicit(monkeypatch):
    h, g = model(3)
    options = dict(time_step=0.2, krylov_indices=[1], num_randomizations=1,
                   num_sampled_groups=6, seed=3)
    normal = build_sqdrift_circuits(h, g, 3, (2, 1), **options)
    optimized = build_sqdrift_circuits(h, g, 3, (2, 1), optimize_layout=True, **options)
    perm = optimized.records[0].mode_permutation
    old = np.abs(state(normal.circuits[0])) ** 2
    new = np.abs(state(optimized.circuits[0])) ** 2
    for i, probability in enumerate(new):
        bits = format(i, "06b")
        restored = int("".join(bits[perm[p]] for p in range(6)), 2)
        assert probability == pytest.approx(old[restored], abs=1e-10)
    # 无可行排列时保持原模式，不输出半成品排列。
    from types import SimpleNamespace
    from tyxonq.applications.chem.algorithms.skqd import layout
    monkeypatch.setattr(layout, "milp", lambda *a, **k: SimpleNamespace(x=None, success=False))
    fallback = optimize_mode_order([(0, 5)], 6)
    assert fallback.permutation == tuple(range(6))
    assert fallback.status == "identity_no_feasible_solution"
    filtered = build_sqdrift_circuits(h, g, 3, (1, 1), filter_diagonal=True,
                                      filter_trivial=True, **options)
    assert filtered.records[0].diagnostics["filter_diagonal"]
    assert filtered.records[0].diagnostics["filter_trivial"]


def test_native_compiler_preserves_gaussian_matrices(monkeypatch):
    import builtins
    from tyxonq.compiler.api import compile
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"qiskit", "ffsim", "qiskit_fermions"}:
            raise AssertionError(f"unexpected native compiler dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    h, g = model()
    for builder in (build_trotter_krylov_circuits, build_sqdrift_circuits):
        circuit = builder(h, g, 2, (1, 1), time_step=0.2, krylov_indices=[1]).circuits[0]
        circuit = circuit.with_metadata(test="retain matrices")
        before = state(circuit)
        with pytest.warns(UserWarning, match="自动补 Z 测量"):
            compiled = compile(circuit, platform="tyxonq", output="ir")["compiled_source"]
        assert aligned_error(state(compiled), before) < 1e-10


def test_example_models_and_three_stage_workflow(tmp_path):
    import importlib.util
    from pathlib import Path
    import sys
    # CLI 所在目录只用于加载示例，核心算法无需把 examples 加入 Python 路径。
    directory = Path(__file__).resolve().parents[1] / "examples/SKQD"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("skqd_example_workflow", directory / "workflow.py")
        workflow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow)
        config = workflow.read_json(directory / "impurity.json")
        config["evolution"]["krylov_indices"] = [0, 1, 2]
        config["sqd"]["max_iterations"] = 2
        path = tmp_path / "config.json"
        workflow.write_json(path, config)
        first, second = tmp_path / "first", tmp_path / "second"
        workflow.prepare(path, first)
        workflow.sample(first, shots=100)
        result = workflow.postprocess(first, exact=True)
        assert result["energy_error"] >= -1e-9
        assert result["rdm_energy_error"] < 1e-8
        workflow.rotate(first, second)
        assert not (second / "counts.json").exists()
        workflow.sample(second, shots=100)
        other = workflow.postprocess(second, exact=True)
        assert other["statistics"]["basis_id"] != result["statistics"]["basis_id"]
        assert other["exact_energy_active"] == pytest.approx(result["exact_energy_active"], abs=1e-9)
        # 不依赖历史论文模板，直接验证缺失输入会在构造线路前被拒绝。
        incomplete = workflow.read_json(directory / "impurity.json")
        incomplete["evolution"]["time_step"] = None
        incomplete_path = tmp_path / "incomplete.json"
        workflow.write_json(incomplete_path, incomplete)
        assert workflow.estimate(incomplete)["missing_inputs"] == ["evolution.time_step"]
        with pytest.raises(ValueError, match="complete these config fields"):
            workflow.prepare(incomplete_path, tmp_path / "missing")
        from models import near_fermi_reference, build_problem
        reference = near_fermi_reference(8, (4, 4))
        assert sum(op[0] == "unitary" for op in reference.ops) == 14
        four = workflow.read_json(directory / "four_impurity.json")
        problem, blocks = build_problem(four)
        assert len(blocks) == 4 and all(len(block) == 2 for block in blocks)
        assert len(set(p for block in blocks for p in block)) == 8
        assert np.count_nonzero(problem["eri"]) == 4
    finally:
        sys.path.remove(str(directory))


def test_notebook_reference_optional_qiskit_crosscheck():
    """第三方只用于此可选独立对照，原生构建/采样/求解测试会禁止其导入。"""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate
    from qiskit.quantum_info import Statevector
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "examples/SKQD/models.py"
    spec = importlib.util.spec_from_file_location("skqd_reference_models", path)
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    original = QuantumCircuit(16)
    # 原 notebook 的门参数及排列逐项保留，专门检查矩阵大小端转换的符号。
    for p in range(4):
        original.x(p)
        original.x(8 + p)
    gate = XXPlusYYGate(np.pi / 2, -np.pi / 2)
    for layer in range(3):
        for p in range(3 - layer, 4 + layer, 2):
            original.append(gate, [p, p + 1])
            original.append(gate, [8 + p, 9 + p])
    original.append(gate, [p + 1, p + 2])
    original.append(gate, [9 + p, 10 + p])
    expected = np.asarray(Statevector(original)).reshape((2,) * 16).T.reshape(-1)
    assert aligned_error(state(models.near_fermi_reference(8, (4, 4))), expected) < 1e-12
