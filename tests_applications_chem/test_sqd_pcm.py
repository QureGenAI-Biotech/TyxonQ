from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tyxonq.applications.chem.algorithms import sqd
from tyxonq.applications.chem.algorithms.sqd import pcm


class _DummyMol:
    """只提供 PCM 代数测试需要的 AO 接口。"""

    def nao_nr(self) -> int:
        return 1

    def intor_symmetric(self, name: str) -> np.ndarray:
        assert name == "int1e_ovlp"
        return np.eye(1)


class _TwoAOMol:
    """提供两个正交 AO，用于区分两个 SQD batch 的密度。"""

    def nao_nr(self) -> int:
        return 2

    def intor_symmetric(self, name: str) -> np.ndarray:
        assert name == "int1e_ovlp"
        return np.eye(2)


class _LinearPCM:
    """使用 V=0.1D 的确定性响应，便于检查外循环和续跑。"""

    method = ""
    eps = 0.0
    max_memory = 0

    def kernel(self, density: np.ndarray) -> tuple[float, np.ndarray]:
        potential = 0.1 * np.asarray(density)
        energy = 0.5 * float(np.einsum("ij,ji->", density, potential))
        return energy, potential


class _FixedSCIState:
    """固定闭壳层态，只实现 PCM driver 实际读取的接口。"""

    ci_strs_a = np.asarray([1], dtype=np.int64)
    ci_strs_b = np.asarray([1], dtype=np.int64)

    def spin_square(self) -> float:
        return 0.0


def _problem() -> pcm.PCMActiveSpaceProblem:
    return pcm.PCMActiveSpaceProblem(
        mol=_DummyMol(),  # type: ignore[arg-type]
        h1e_vacuum=np.asarray([[-1.0]]),
        eri_chemist=np.zeros((1, 1, 1, 1)),
        ecore_vacuum=0.0,
        active_mo_coeff=np.eye(1),
        inactive_core_mo_coeff=np.empty((1, 0)),
        nelec=(1, 1),
    )


def _two_ao_problem() -> pcm.PCMActiveSpaceProblem:
    """构造两个 batch 可产生不同 1-RDM 的最小活性空间问题。"""

    return pcm.PCMActiveSpaceProblem(
        mol=_TwoAOMol(),  # type: ignore[arg-type]
        h1e_vacuum=np.zeros((2, 2)),
        eri_chemist=np.zeros((2, 2, 2, 2)),
        ecore_vacuum=0.0,
        active_mo_coeff=np.eye(2),
        inactive_core_mo_coeff=np.empty((2, 0)),
        nelec=(1, 1),
    )


def _sqd_settings() -> pcm.PCMSQDSettings:
    return pcm.PCMSQDSettings(
        samples_per_batch=1,
        num_batches=1,
        max_iterations=1,
        max_dim=None,
        spin_sq=0.0,
    )


def _fake_run_sqd(
    one_body_tensor,
    _two_body_tensor,
    _samples,
    *,
    nuclear_repulsion_energy,
    **_kwargs,
):
    density = np.asarray([[2.0]])
    rdm1s = (np.asarray([[1.0]]), np.asarray([[1.0]]))
    return SimpleNamespace(
        energy=2.0 * float(np.asarray(one_body_tensor)[0, 0]),
        total_energy=(
            2.0 * float(np.asarray(one_body_tensor)[0, 0])
            + float(nuclear_repulsion_energy)
        ),
        sci_state=_FixedSCIState(),
        orbital_occupancies=(np.asarray([1.0]), np.asarray([1.0])),
        rdm1=density,
        rdm1s=rdm1s,
    )


def _scientific_trajectory(trajectory):
    timing_keys = {
        "pcm_input_seconds",
        "sqd_seconds",
        "pcm_output_seconds",
        "iteration_seconds",
    }
    return [
        {key: value for key, value in item.items() if key not in timing_keys}
        for item in trajectory
    ]


def test_pcm_public_api_and_method_aliases():
    assert sqd.PCMActiveSpaceProblem is pcm.PCMActiveSpaceProblem
    assert sqd.run_pcm_sqd_scrf is pcm.run_pcm_sqd_scrf
    assert pcm.canonical_pcm_method("CPCM") == "C-PCM"
    assert pcm.canonical_pcm_method("ief-pcm") == "IEF-PCM"

    with pytest.raises(ValueError, match="C-PCM/CPCM"):
        pcm.canonical_pcm_method("COSMO")


def test_pcm_density_projection_and_double_count_bookkeeping():
    problem = _problem()
    active_rdm1 = np.asarray([[2.0]])
    ao_rdm1 = pcm.build_ao_density(
        active_rdm1,
        problem.active_mo_coeff,
        problem.inactive_core_mo_coeff,
    )
    projected = pcm.project_ao_density_to_active(
        problem.mol,
        ao_rdm1,
        problem.active_mo_coeff,
        problem.inactive_core_mo_coeff,
    )
    bookkeeping = pcm.pcm_response_bookkeeping(
        problem,
        ao_rdm1,
        pcm_energy=0.2,
        vpcm_ao=np.asarray([[0.2]]),
    )

    np.testing.assert_allclose(projected, active_rdm1, atol=1.0e-14)
    assert bookkeeping["vpcm_full"] == pytest.approx(0.4)
    assert bookkeeping["double_count_correction"] == pytest.approx(-0.2)
    assert bookkeeping["dynamic_ecore"] == pytest.approx(-0.2)


def test_pcm_problem_and_settings_reject_invalid_inputs(monkeypatch):
    problem = _problem()
    invalid_problem = replace(problem, eri_chemist=np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="eri_chemist shape"):
        invalid_problem.validate()

    monkeypatch.setattr(pcm, "run_sqd_fermion", _fake_run_sqd)
    samples = {"11": 1}
    with pytest.raises(ValueError, match="电子数迹"):
        pcm.run_pcm_sqd_scrf(
            problem,
            samples,
            np.asarray([[1.0]]),
            _sqd_settings(),
            pcm.PCMSCRFSettings(pcm_method="C-PCM", max_iterations=2),
            pcm_model=_LinearPCM(),
        )
    with pytest.raises(ValueError, match="mixing"):
        pcm.run_pcm_sqd_scrf(
            problem,
            samples,
            np.asarray([[2.0]]),
            _sqd_settings(),
            pcm.PCMSCRFSettings(pcm_method="C-PCM", max_iterations=2, mixing=0.0),
            pcm_model=_LinearPCM(),
        )


@pytest.mark.parametrize("method", ["C-PCM", "IEF-PCM"])
def test_pcm_checkpoint_resume_matches_uninterrupted_run(monkeypatch, method):
    monkeypatch.setattr(pcm, "run_sqd_fermion", _fake_run_sqd)
    problem = _problem()
    samples = {"11": 7}
    initial_rdm1 = np.asarray([[2.0]])
    occupancies = (np.asarray([1.0]), np.asarray([1.0]))
    full_settings = pcm.PCMSCRFSettings(pcm_method=method, max_iterations=3)

    full = pcm.run_pcm_sqd_scrf(
        problem,
        samples,
        initial_rdm1,
        _sqd_settings(),
        full_settings,
        initial_occupancies=occupancies,
        pcm_model=_LinearPCM(),
    )
    checkpoints: list[pcm.PCMSCRFCheckpoint] = []
    first_part = pcm.run_pcm_sqd_scrf(
        problem,
        samples,
        initial_rdm1,
        _sqd_settings(),
        replace(full_settings, max_iterations=1),
        initial_occupancies=occupancies,
        pcm_model=_LinearPCM(),
        iteration_callback=checkpoints.append,
    )
    resumed = pcm.run_pcm_sqd_scrf(
        problem,
        samples,
        None,
        _sqd_settings(),
        full_settings,
        pcm_model=_LinearPCM(),
        resume_from=first_part.checkpoint,
    )

    assert full.converged and resumed.converged
    assert len(checkpoints) == 1
    assert checkpoints[0].next_outer_iteration == 2
    assert samples == {"11": 7}
    assert resumed.total_energy == pytest.approx(full.total_energy, abs=1.0e-12)
    np.testing.assert_allclose(resumed.active_rdm1, full.active_rdm1, atol=1.0e-12)
    assert _scientific_trajectory(resumed.trajectory) == _scientific_trajectory(
        full.trajectory
    )

    with pytest.raises(ValueError, match="initial_active_rdm1 必须为 None"):
        pcm.run_pcm_sqd_scrf(
            problem,
            samples,
            initial_rdm1,
            _sqd_settings(),
            full_settings,
            pcm_model=_LinearPCM(),
            resume_from=first_part.checkpoint,
        )


def test_pcm_uses_batch_average_density_and_reports_lowest_batch(monkeypatch):
    """论文式外循环用 batch 平均密度更新反应场，但报告最低能 batch。"""

    density_first = np.diag([2.0, 0.0])
    density_second = np.diag([0.0, 2.0])
    results = [
        SimpleNamespace(
            energy=-1.0,
            total_energy=-0.8,
            sci_state=_FixedSCIState(),
            orbital_occupancies=(np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])),
            rdm1=density_first,
            rdm1s=(np.diag([1.0, 0.0]), np.diag([1.0, 0.0])),
        ),
        SimpleNamespace(
            energy=-0.9,
            total_energy=-0.7,
            sci_state=_FixedSCIState(),
            orbital_occupancies=(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])),
            rdm1=density_second,
            rdm1s=(np.diag([0.0, 1.0]), np.diag([0.0, 1.0])),
        ),
    ]

    def fake_run_sqd(*_args, **kwargs):
        kwargs["callback"](results)
        return results[0]

    monkeypatch.setattr(pcm, "run_sqd_fermion", fake_run_sqd)
    callbacks: list[tuple[int, int, int]] = []
    result = pcm.run_pcm_sqd_scrf(
        _two_ao_problem(),
        {"1010": 2},
        np.eye(2),
        pcm.PCMSQDSettings(
            samples_per_batch=1,
            num_batches=2,
            max_iterations=1,
            max_dim=None,
            spin_sq=0.0,
        ),
        pcm.PCMSCRFSettings(pcm_method="C-PCM", max_iterations=1),
        pcm_model=_LinearPCM(),
        inner_iteration_callback=lambda outer, inner, batches: callbacks.append(
            (outer, inner, len(batches))
        ),
    )

    np.testing.assert_allclose(result.active_rdm1, np.eye(2), atol=1.0e-14)
    np.testing.assert_allclose(
        result.checkpoint.occupancies[0], np.asarray([0.5, 0.5]), atol=1.0e-14
    )
    assert result.sci_result is results[0]
    assert result.total_energy == pytest.approx(-1.0, abs=1.0e-14)
    assert result.trajectory[0]["batch_total_energies_hartree"] == pytest.approx(
        [-1.0, -0.9], abs=1.0e-14
    )
    assert callbacks == [(1, 1, 2)]
