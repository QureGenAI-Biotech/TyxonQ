"""可选 CUDA 扩展的原生迭代边界；无扩展或 GPU 时跳过。"""

import numpy as np
import pytest


@pytest.fixture
def extension():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return pytest.importorskip("_sbd_tpb_cuda")


def test_single_determinant_native_space(extension):
    """只有一个组态时内部子空间为 1，零残差不能导致除零。"""
    context = extension.TPBContext(
        np.diag([-1.0, 0.5]), np.zeros((2,) * 4),
        np.array([1], dtype=np.int64), np.array([1], dtype=np.int64), 2, (1, 1),
    )
    result = context.solve(1e-12, 3, 20)
    assert result["converged"]
    assert result["iterations"] == 1
    assert result["energy"] == pytest.approx(-2.0, abs=1e-12)
    assert result["residual"] < 1e-12
    np.testing.assert_allclose(result["coefficients"], [1.0], atol=1e-12)
    with pytest.raises(ValueError, match="convergence settings"):
        context.solve(1e-12, 3, 1)


def test_native_total_matvec_budget(extension):
    """一轮严格截断不能被解释成一次包含多次扩张的重启。"""
    context = extension.TPBContext(
        np.array([[0.0, 0.3], [0.3, 1.0]]), np.zeros((2,) * 4),
        np.array([1, 2], dtype=np.int64), np.array([0], dtype=np.int64), 2, (1, 0),
    )
    limited = context.solve(1e-12, 1, 20, compute_rdms=False)
    assert limited["iterations"] == 1
    assert not limited["converged"]
    assert limited["residual"] == pytest.approx(0.3, abs=1e-12)
    assert limited["one_p_rdm"] is None
    result = context.solve(1e-12, 2, 20, compute_rdms=False)
    assert result["iterations"] == 2
    assert result["converged"]
    assert result["energy"] == pytest.approx((1.0 - np.sqrt(1.36)) / 2, abs=1e-12)
    assert result["residual"] < 1e-12


def test_native_nearly_dependent_directions(extension):
    """弱耦合且对角能量接近时，以完整矩阵检查伪低能根和正交化失稳。"""
    norb = 12
    rng = np.random.default_rng(0)
    diagonal = -160.0 + rng.uniform(0.0, 2.0, size=norb)
    coupling = rng.normal(size=(norb, norb))
    h1e = np.diag(diagonal) + 0.001 * (coupling + coupling.T) / 2
    context = extension.TPBContext(
        h1e, np.zeros((norb,) * 4), 1 << np.arange(norb, dtype=np.int64),
        np.array([0], dtype=np.int64), norb, (1, 0),
    )
    result = context.solve(1e-10, 200, 20, compute_rdms=False)
    coefficients = np.asarray(result["coefficients"])
    coefficients /= np.linalg.norm(coefficients)
    reference = np.linalg.eigvalsh(h1e)[0]
    direct_residual = np.linalg.norm(h1e @ coefficients - reference * coefficients)
    assert result["converged"]
    assert result["iterations"] <= 200
    assert result["energy"] == pytest.approx(reference, abs=1e-10)
    assert result["residual"] <= 1e-10
    assert direct_residual <= 1e-10


def test_native_diagonal_hamiltonian_with_dense_initial(extension):
    """纯对角 H 的精确预条件校正落回当前态，仍须找到独立残差方向。"""
    diagonal = np.array([-4.0, -1.0, 2.0, 7.0])
    initial = np.array([1.0, 2.0, 3.0, 4.0])
    initial /= np.linalg.norm(initial)
    context = extension.TPBContext(
        np.diag(diagonal), np.zeros((4,) * 4), 1 << np.arange(4, dtype=np.int64),
        np.array([0], dtype=np.int64), 4, (1, 0),
    )
    result = context.solve(1e-11, 20, 20, initial, compute_rdms=False)
    coefficients = np.asarray(result["coefficients"])
    coefficients /= np.linalg.norm(coefficients)
    assert result["converged"]
    assert 1 < result["iterations"] <= 20
    assert result["energy"] == pytest.approx(diagonal.min(), abs=1e-11)
    assert result["residual"] <= 1e-11
    assert np.linalg.norm(diagonal * coefficients - diagonal.min() * coefficients) <= 1e-11


def test_context_memory_budget_rejects_before_allocation(extension):
    with pytest.raises(RuntimeError, match="memory budget exceeded"):
        extension.TPBContext(
            np.diag([-1.0, 0.5]), np.zeros((2,) * 4),
            np.array([1], dtype=np.int64), np.array([1], dtype=np.int64), 2, (1, 1),
            memory_fraction=1e-12,
        )


def test_crossed_spin_excitation_keeps_ab_block(extension):
    """自旋求和会掩盖 ab/ba 写反，使用可手算的两个组态单独验收。"""
    import torch

    context = extension.TPBContext(
        np.zeros((4, 4)), np.zeros((4,) * 4),
        np.array([1, 4], dtype=np.int64), np.array([2, 8], dtype=np.int64), 4, (1, 1),
    )
    # PySCF 态为 (|α0 β3>+|α2 β1>)/sqrt(2)；第二项转换交错位序时带负号。
    c = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64, device="cuda") / np.sqrt(2.0)
    raw = context.rdms_device(c.data_ptr(), c.numel())
    blocks = np.asarray(raw["two_p_rdm"])
    ab = blocks[2].reshape((4,) * 4, order="F").transpose(0, 2, 1, 3)
    expected = np.zeros((4,) * 4)
    expected[0, 0, 3, 3] = expected[2, 2, 1, 1] = 0.5
    expected[0, 2, 3, 1] = expected[2, 0, 1, 3] = 0.5
    np.testing.assert_allclose(ab, expected, atol=1e-12)
    np.testing.assert_allclose(blocks[[0, 3]], 0, atol=1e-12)


@pytest.mark.parametrize("nelec", [(2, 2), (2, 1), (1, 0)])
def test_fused_spin_retains_two_step_projection(extension, nelec):
    """用不完整的行列式集合检查融合没有改动升降算符投影。"""
    import torch
    from pyscf.fci import cistring, selected_ci
    from tyxonq.applications.chem.algorithms.sqd.gpu.torch_spin_square import TorchSelectedS2Contractor

    strings = tuple(cistring.make_strings(range(4), number)[::2] for number in nelec)
    shape = tuple(len(side) for side in strings)
    c = np.random.default_rng(20261013).normal(size=shape)
    c /= np.linalg.norm(c)
    state = selected_ci._as_SCIvector(c, strings)
    reference = selected_ci.contract_ss(state, 4, nelec)
    kwargs = dict(ci_strings=strings, norb=4, nelec=nelec, device=torch.device("cuda"), dtype=torch.float64)
    baseline = TorchSelectedS2Contractor(**kwargs)
    fused = TorchSelectedS2Contractor(**kwargs, backend="cuda_fused")
    coefficients = torch.tensor(c, dtype=torch.float64, device="cuda")
    # 同一对象再调用一次，覆盖中间态清零与复用。
    for _ in range(2):
        actual = fused.contract(coefficients).cpu().numpy()
        np.testing.assert_allclose(actual, reference, atol=1e-12)
        np.testing.assert_allclose(actual, baseline.contract(coefficients).cpu().numpy(), atol=1e-12)
    assert fused.external_memory_bytes() >= 0
