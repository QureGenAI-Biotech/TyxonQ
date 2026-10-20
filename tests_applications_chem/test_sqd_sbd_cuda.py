"""可选 A100 验收：随机算符像、密度及跨调用积分更新。"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _problem(norb=4, nelec=(2, 1), labels=None):
    from pyscf.fci import cistring

    rng = np.random.default_rng(20261013)
    h = rng.normal(size=(norb, norb))
    h = (h + h.T) / 2
    factors = rng.normal(size=(3, norb, norb)) * 0.1
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    eri = np.einsum("lpq,lrs->pqrs", factors, factors)
    if labels is None:
        labels = tuple(cistring.make_strings(range(norb), n) for n in nelec)
    labels = tuple(np.asarray(x, dtype=np.int64) for x in labels)
    c = rng.normal(size=tuple(map(len, labels)))
    c /= np.linalg.norm(c)
    return labels, h, eri, c


def _cpu_image(labels, h, eri, c, norb, nelec):
    from pyscf.fci import cistring, direct_spin1, selected_ci

    ci = selected_ci._as_SCIvector(c, labels)
    if 0 in nelec:
        # PySCF selected_ci 的 Hc 对空自旋扇区会除零；小测试用完整 FCI 独立复核。
        full_labels = tuple(cistring.make_strings(range(norb), n) for n in nelec)
        indices = tuple(np.searchsorted(full, part) for full, part in zip(full_labels, labels))
        full = np.zeros(tuple(map(len, full_labels)))
        full[np.ix_(*indices)] = c
        return np.asarray(direct_spin1.contract_2e(
            direct_spin1.absorb_h1e(h, eri, norb, nelec, 0.5), full, norb, nelec))[np.ix_(*indices)]
    return np.asarray(selected_ci.contract_2e(
        direct_spin1.absorb_h1e(h, eri, norb, nelec, 0.5), ci, norb, nelec))


@pytest.mark.parametrize("norb,nelec,labels", [
    (4, (2, 2), None),
    (4, (2, 1), None),
    (4, (1, 1), ([4], [8])),
    (4, (1, 0), ([1, 4], [0])),
    # 每个自旋标签跨越 SBD 的 20-bit 存储字，同时不含 HF。
    (22, (2, 1), ([1 + (1 << 20), 2 + (1 << 21)], [2, 1 << 20])),
])
def test_random_hc_spin_rdms_and_updated_integrals(norb, nelec, labels):
    from pyscf.fci import selected_ci
    from tyxonq.applications.chem.algorithms.sqd.gpu.sbd_solver import SBDContractor
    from tyxonq.applications.chem.algorithms.sqd.gpu.torch_spin_square import TorchSelectedS2Contractor

    labels, h, eri, c = _problem(norb, nelec, labels)
    contractor = SBDContractor(h, eri, labels, norb, nelec, device="cuda:0")
    vector = torch.tensor(c, dtype=torch.float64, device="cuda:0")
    expected = _cpu_image(labels, h, eri, c, norb, nelec)
    actual = contractor.contract(vector).cpu().numpy()
    assert np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-15) < 1e-10
    np.testing.assert_allclose(contractor._raw(contractor._raw(vector)).cpu(), c, atol=0, rtol=0)
    ci = selected_ci._as_SCIvector(c, labels)
    expected_rdms = (selected_ci.make_rdm1s(ci, norb, nelec),
                     selected_ci.make_rdm2s(ci, norb, nelec))
    actual_rdms = contractor.rdms(vector)
    for actual, expected in zip((*actual_rdms[0], *actual_rdms[1]),
                                (*expected_rdms[0], *expected_rdms[1])):
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
    spin = TorchSelectedS2Contractor(labels, norb, nelec,
                                     device=torch.device("cuda:0"), dtype=torch.float64)
    np.testing.assert_allclose(spin.contract(vector).cpu(),
                               selected_ci.contract_ss(ci, norb, nelec), atol=1e-10, rtol=0)
    changed_h = h + np.diag(np.linspace(-0.2, 0.3, norb))
    contractor.update_integrals(changed_h, 1.07 * eri)
    np.testing.assert_allclose(contractor.contract(vector).cpu(),
                               _cpu_image(labels, changed_h, 1.07 * eri, c, norb, nelec),
                               atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("backend,spin_sq", [("native", None), ("hybrid", None),
                                             ("hybrid", 0.75), ("hybrid", 3.75)])
def test_solver_dense_reference_and_failure(backend, spin_sq):
    from pyscf.fci import addons, direct_spin1, selected_ci
    from tyxonq.applications.chem.algorithms.sqd.gpu.sbd_solver import solve_sci_sbd
    from tyxonq.applications.chem.algorithms.sqd.gpu.torch_sci_solver import solve_sci_cuda

    norb, nelec = 4, (2, 1)
    labels, h, eri, c = _problem(norb, nelec)
    # 输入故意乱序并重复；输出必须与规范标签逐项对齐。
    dirty = tuple(np.concatenate([x[::-1], x[:1]]) for x in labels)
    solve = solve_sci_sbd if backend == "native" else solve_sci_cuda
    options = {} if backend == "native" else {"hamiltonian_backend": "sbd", "rdm_backend": "sbd"}
    options.update(threads=2, chunk_size=256, operator_residual_tolerance=1e-7,
                   rdm_energy_tolerance=1e-8)
    result, report = solve(dirty, h, eri, norb, nelec, spin_sq=spin_sq, spin_shift=2,
                           tolerance=1e-10, max_cycles=300, max_space=24, **options)
    assert result.converged and report["independent_cpu_validation"]
    solver = selected_ci.SelectedCI()
    if spin_sq is not None:
        solver = addons.fix_spin_(solver, shift=2, ss=spin_sq)
    absorbed = direct_spin1.absorb_h1e(h, eri, norb, nelec, 0.5)
    dim = c.size
    eye = np.eye(dim).reshape(dim, *c.shape)
    matrix = np.column_stack([np.asarray(solver.contract_2e(
        absorbed, selected_ci._as_SCIvector(v, labels), norb, nelec)).ravel() for v in eye])
    expected = np.linalg.eigvalsh(matrix)[0]
    assert abs(report["gpu_penalized_eigenvalue_hartree"] - expected) <= 1e-8
    for a, b in zip((result.sci_state.ci_strs_a, result.sci_state.ci_strs_b), labels):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(RuntimeError, match="converge|residual|Residual"):
        solve(labels, h, eri, norb, nelec, max_cycles=1, max_space=24,
              tolerance=1e-12, **options)


def test_live_contexts_different_word_widths():
    from tyxonq.applications.chem.algorithms.sqd.gpu.sbd_solver import SBDContractor

    labels, h, eri, c = _problem()
    first = SBDContractor(h, eri, labels, 4, (2, 1), device="cuda:0")
    labels2, h2, eri2, unused = _problem(22, (1, 1), ([1, 1 << 21], [2, 1 << 20]))
    second = SBDContractor(h2, eri2, labels2, 22, (1, 1), device="cuda:0")
    vector = torch.tensor(c, dtype=torch.float64, device="cuda:0")
    np.testing.assert_allclose(first.contract(vector).cpu(),
                               _cpu_image(labels, h, eri, c, 4, (2, 1)), atol=1e-10, rtol=1e-10)


def test_cuda_device_alias_remains_bound_after_switching_current_device():
    """仅两卡环境运行：无编号 cuda 上下文不随外部当前设备切换而漂移。"""
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices required")
    from tyxonq.applications.chem.algorithms.sqd.gpu.sbd_solver import SBDContractor

    previous_device = torch.cuda.current_device()
    contractor = None
    try:
        torch.cuda.set_device(0)
        labels, h, eri, c = _problem()
        contractor = SBDContractor(h, eri, labels, 4, (2, 1), device="cuda")
        original_device = torch.device("cuda:0")
        assert contractor.device == contractor.phase.device == original_device
        vector = torch.tensor(c, dtype=torch.float64, device=original_device)
        expected_diagonal = contractor.diagonal().cpu().numpy()

        torch.cuda.set_device(1)
        diagonal = contractor.diagonal()
        assert diagonal.device == original_device
        np.testing.assert_allclose(diagonal.cpu(), expected_diagonal, atol=0, rtol=0)
        torch.cuda.set_device(1)
        image = contractor.contract(vector)
        assert image.device == original_device
        np.testing.assert_allclose(image.cpu(), _cpu_image(labels, h, eri, c, 4, (2, 1)),
                                   atol=1e-10, rtol=1e-10)
    finally:
        # C++ 析构也会选择其所属卡，须先释放上下文再恢复测试前的当前设备。
        if contractor is not None:
            del contractor
        torch.cuda.set_device(previous_device)


def test_same_seed_sqd_recovery_and_batch_results():
    """固定样本和随机种子，检查后端切换不改变三轮恢复及批次顺序。"""
    from tyxonq.applications.chem.algorithms.sqd.fermion import run_sqd_fermion
    from tyxonq.applications.chem.algorithms.sqd.gpu import make_cuda_sci_solver, make_sbd_sci_solver

    labels, h, eri, unused = _problem()
    counts = {format(i, "08b"): 1 + i % 7 for i in range(256)}
    options = dict(threads=2, chunk_size=256, tolerance=1e-10, max_cycles=100,
                   max_space=24, operator_residual_tolerance=1e-7, rdm_energy_tolerance=1e-8)
    histories, energies = [], []
    for mode in ("torch", "native", "hybrid"):
        solver = (make_cuda_sci_solver(**options) if mode == "torch"
                  else make_sbd_sci_solver(mode=mode, **options))
        history = []

        def record(results):
            history.append([(r.sci_state.ci_strs_a.copy(), r.sci_state.ci_strs_b.copy(), r.energy)
                            for r in results])

        try:
            result = run_sqd_fermion(
                h, eri, counts, 8, 4, (2, 1), num_batches=2, max_iterations=3,
                energy_tol=0, occupancies_tol=0, seed=20261013, sci_solver=solver,
                initial_occupancies=(np.array([1., 1., 0., 0.]), np.array([1., 0., 0., 0.])),
                callback=record,
            )
        finally:
            solver.close()
        histories.append(history)
        energies.append(result.energy)
    np.testing.assert_allclose(energies, energies[0], atol=1e-8, rtol=0)
    assert len(histories[0]) == 3
    for actual in histories[1:]:
        assert len(actual) == len(histories[0])
        for one, ref in zip(actual, histories[0]):
            assert len(one) == len(ref) == 2
            for (a, b, energy), (ra, rb, re) in zip(one, ref):
                np.testing.assert_array_equal(a, ra)
                np.testing.assert_array_equal(b, rb)
                assert abs(energy - re) <= 1e-8


@pytest.mark.parametrize("backend", ["torch", "native", "hybrid"])
@pytest.mark.parametrize("nelec", [(1, 0), (0, 1)])
def test_empty_spin_sector_full_solver(backend, nelec):
    """单电子极限的能量直接等于一体矩阵最低本征值，二体作用严格为零。"""
    from tyxonq.applications.chem.algorithms.sqd.gpu.sbd_solver import solve_sci_sbd
    from tyxonq.applications.chem.algorithms.sqd.gpu.torch_sci_solver import solve_sci_cuda

    labels, h, eri, unused = _problem(4, nelec)
    options = (dict(hamiltonian_backend="sbd", rdm_backend="sbd", spin_backend="cuda_fused")
               if backend == "hybrid" else {})
    solver = solve_sci_sbd if backend == "native" else solve_sci_cuda
    result, report = solver(labels, h, eri, 4, nelec, tolerance=1e-10, **options)
    assert result.converged and report["independent_cpu_validation"]
    assert abs(result.energy - np.linalg.eigvalsh(h)[0]) <= 1e-8
    assert np.max(np.abs(result.rdm2)) <= 1e-10
