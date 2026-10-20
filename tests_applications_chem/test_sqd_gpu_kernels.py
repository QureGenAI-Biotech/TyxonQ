"""在 CPU 上将迁移后的 Torch 算符与独立 PySCF 实现对照。"""

import numpy as np
import pytest
import torch
from pyscf.fci import cistring, direct_spin1, selected_ci

from tyxonq.applications.chem.algorithms.sqd.gpu.torch_davidson import davidson_lowest
from tyxonq.applications.chem.algorithms.sqd.gpu.torch_selected_ci import (
    TorchSelectedCIContractor, absorb_h1e_torch,
)
from tyxonq.applications.chem.algorithms.sqd.gpu.torch_spin_square import TorchSelectedS2Contractor


@pytest.mark.parametrize("nelec", [(2, 2), (3, 1)])
@pytest.mark.parametrize("partial", [False, True])
def test_torch_hamiltonian_and_spin_match_pyscf(nelec, partial):
    rng = np.random.default_rng(123)
    norb = 4
    h = rng.normal(size=(norb, norb))
    h = (h + h.T) / 2
    factors = rng.normal(size=(5, norb, norb))
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    eri = np.einsum("lpq,lrs->pqrs", factors, factors)
    strings = tuple(cistring.make_strings(range(norb), n) for n in nelec)
    if partial:
        strings = tuple(s[::2] for s in strings)
    coefficients = rng.normal(size=tuple(len(s) for s in strings))
    coefficients /= np.linalg.norm(coefficients)
    ci = selected_ci._as_SCIvector(coefficients, strings)
    expected = selected_ci.contract_2e(
        direct_spin1.absorb_h1e(h, eri, norb, nelec, 0.5), ci, norb, nelec
    )
    links = selected_ci._all_linkstr_index(strings, norb, nelec)
    actual = TorchSelectedCIContractor(
        absorb_h1e_torch(torch.tensor(h), torch.tensor(eri), nelec),
        links, nelec, chunk_size=2,
    ).contract(torch.tensor(coefficients)).numpy()
    actual_spin = TorchSelectedS2Contractor(
        strings, norb, nelec, device=torch.device("cpu"), dtype=torch.float64,
    ).contract(torch.tensor(coefficients)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        actual_spin, selected_ci.contract_ss(ci, norb, nelec), atol=1e-10, rtol=0
    )


def test_davidson_against_dense_eigensolver():
    rng = np.random.default_rng(24)
    matrix = rng.normal(size=(8, 8))
    matrix = torch.tensor(matrix + matrix.T, dtype=torch.float64)
    progress = []
    result = davidson_lowest(
        lambda vector: matrix @ vector, matrix.diag(),
        tolerance=1e-10, max_space=8, callback=progress.append,
    )
    assert result.converged
    assert progress
    assert abs(result.energy - float(torch.linalg.eigvalsh(matrix)[0])) < 1e-10
    assert float(torch.linalg.vector_norm(matrix @ result.coefficients - result.energy * result.coefficients)) < 1e-10


@pytest.mark.parametrize("max_space", [2, 4])
def test_davidson_thick_restart_respects_capacity(max_space):
    """有限子空间必须真实重启，并在不越过容量的条件下得到正确最低根。"""
    rng = np.random.default_rng(6)
    coupling = rng.normal(size=(10, 10)) * 0.05
    matrix = torch.tensor(
        np.diag(np.arange(10)) + (coupling + coupling.T) / 2,
        dtype=torch.float64,
    )
    result = davidson_lowest(
        lambda vector: matrix @ vector, matrix.diag(),
        tolerance=1e-10, max_cycles=300, max_space=max_space,
    )
    dimensions = [entry["subspace_dimension"] for entry in result.history]
    assert result.converged
    assert max(dimensions) <= max_space
    assert result.iterations > max_space
    assert abs(result.energy - float(torch.linalg.eigvalsh(matrix)[0])) < 1e-10
    residual = matrix @ result.coefficients - result.energy * result.coefficients
    assert float(torch.linalg.vector_norm(residual)) < 1e-10


def test_davidson_initial_vector_is_used_without_mutation():
    """已有本征态作为非单位初猜时应一轮收敛，并保留调用方输入。"""
    matrix = torch.tensor([[2.0, -0.3, 0.2], [-0.3, 1.0, 0.4], [0.2, 0.4, 3.0]])
    matrix = matrix.to(torch.float64)
    values, vectors = torch.linalg.eigh(matrix)
    initial = 7 * vectors[:, 0]
    saved = initial.clone()
    result = davidson_lowest(
        lambda vector: matrix @ vector, matrix.diag(),
        initial_vector=initial, tolerance=1e-12, max_space=2,
    )
    assert result.converged and result.iterations == result.h_matvecs == 1
    assert abs(result.energy - float(values[0])) < 1e-12
    torch.testing.assert_close(initial, saved, atol=0, rtol=0)
    assert abs(float(torch.linalg.vector_norm(result.coefficients)) - 1.0) < 1e-12


@pytest.mark.parametrize("max_space", [3, 10])
def test_davidson_diagonal_preconditioner_uses_residual_fallback(max_space):
    """纯对角算符的预条件方向共线时仍能收敛，有限容量下真实经历厚重启。"""
    diagonal = torch.tensor([-3.5, -1., .2, 1.3, 2.1, 4., 5.6, 9.], dtype=torch.float64)
    initial = torch.tensor(np.random.default_rng(24).normal(size=8), dtype=torch.float64)
    saved = initial.clone()
    assert torch.all(initial != 0)
    result = davidson_lowest(
        lambda vector: diagonal * vector, diagonal, initial_vector=initial,
        tolerance=1e-10, max_cycles=300, max_space=max_space,
    )
    assert result.converged
    assert abs(result.energy - float(diagonal.min())) < 1e-10
    fresh_residual = diagonal * result.coefficients - result.energy * result.coefficients
    assert float(torch.linalg.vector_norm(fresh_residual)) < 1e-10
    assert max(record["subspace_dimension"] for record in result.history) <= min(max_space, 8)
    if max_space < diagonal.numel():
        assert result.iterations > max_space
    torch.testing.assert_close(initial, saved, atol=0, rtol=0)


def test_davidson_iteration_limit_remains_nonconverged():
    """达到迭代上限不能把有限的 Ritz 能量误报为收敛。"""
    matrix = torch.tensor([[1.0, 0.3], [0.3, 2.0]], dtype=torch.float64)
    result = davidson_lowest(
        lambda vector: matrix @ vector, matrix.diag(),
        tolerance=1e-12, max_cycles=1, max_space=2,
    )
    assert not result.converged
    assert result.iterations == result.h_matvecs == 1
    assert result.residual_norm > 1e-12
    residual = matrix @ result.coefficients - result.energy * result.coefficients
    assert abs(float(torch.linalg.vector_norm(residual)) - result.residual_norm) < 1e-12


def test_davidson_single_determinant_space():
    """单行列式无需重启或额外方向，容量可小于 max_space。"""
    diagonal = torch.tensor([-1.25], dtype=torch.float64)
    result = davidson_lowest(lambda vector: diagonal * vector, diagonal, max_space=12)
    assert result.converged and result.iterations == 1
    assert result.energy == -1.25
    assert result.history[0]["subspace_dimension"] == 1


def test_cuda_batch_adapter_preserves_options_and_reports(monkeypatch):
    """批接口只负责转发，实际 GPU 数值验收由独立 CUDA 环境完成。"""
    import sys
    from types import SimpleNamespace
    from tyxonq.applications.chem.algorithms.sqd.gpu import make_cuda_sci_solver

    calls, reports = [], []

    def solve(strings, h1, eri, norb, nelec, **options):
        calls.append((strings, h1, eri, norb, nelec, options))
        return strings, {"verified": True}

    module = "tyxonq.applications.chem.algorithms.sqd.gpu.torch_sci_solver"
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(solve_sci_cuda=solve))
    solver = make_cuda_sci_solver(threads=2, tolerance=1e-10, report_callback=reports.append)
    batches = [(np.array([1]), np.array([1])), (np.array([2]), np.array([1]))]
    h1, eri = np.eye(2), np.zeros((2,) * 4)
    results = solver(batches, h1, eri, 2, (1, 1))
    assert results == batches
    assert [report["batch"] for report in reports] == [0, 1]
    assert all(call[-1] == {"threads": 2, "tolerance": 1e-10} for call in calls)
    assert all(call[1] is h1 and call[2] is eri for call in calls)
