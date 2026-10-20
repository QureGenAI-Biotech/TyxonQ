"""SBD 相位、指标约定、独立验收及显式初猜的 CPU 回归。"""

import numpy as np
import pytest

pytest.importorskip("pyscf")
from pyscf import fci, lib

from tyxonq.applications.chem.algorithms.sqd.fermion import SCIState
from tyxonq.applications.chem.algorithms.sqd.gpu.sbd_solver import (
    SBDContractor, convert_raw_rdms, interleaved_phase, solve_sci_sbd,
)
from tyxonq.applications.chem.algorithms.sqd.gpu.validation import (
    assemble_result, compare_cpu, independent_cpu_validation, normalize_problem,
    project_initial_state, rdms_from_cpu, validate_rdms, validate_spin_square,
)


def test_normalization_preserves_labels_and_rejects_wrong_electron_counts():
    labels, _, _, _, _ = normalize_problem(
        (np.array([4, 1, 4, 2]), np.array([3, 6, 3])),
        np.eye(3), np.zeros((3,) * 4), 3, (1, 2))
    assert labels[0].tolist() == [1, 2, 4]
    assert labels[1].tolist() == [3, 6]
    with pytest.raises(ValueError, match="norb/nelec"):
        normalize_problem((np.array([3]), np.array([1])), np.eye(3), np.zeros((3,) * 4), 3, (1, 1))
    with pytest.raises(NotImplementedError, match="real"):
        normalize_problem((np.array([1]), np.array([1])), np.eye(3, dtype=complex), np.zeros((3,) * 4), 3, (1, 1))


@pytest.mark.parametrize("index", [(0, 1, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1)])
def test_restricted_integral_compression_cannot_hide_broken_symmetry(index):
    strings = (np.array([1]), np.array([1]))
    eri = np.zeros((2,) * 4)
    eri[index] = .01
    with pytest.raises(ValueError, match="eightfold"):
        normalize_problem(strings, np.eye(2), eri, 2, (1, 1))
    h1 = np.eye(2)
    h1[0, 1] = .01
    with pytest.raises(ValueError, match="symmetric"):
        normalize_problem(strings, h1, np.zeros((2,) * 4), 2, (1, 1))


def test_restricted_integrals_allow_roundoff_without_symmetrizing():
    strings = (np.array([1]), np.array([1]))
    h1 = np.eye(2)
    h1[0, 1] = 1e-14
    eri = np.zeros((2,) * 4)
    eri[0, 0, 1, 1] = 1e-14
    _, accepted_h1, accepted_eri, _, _ = normalize_problem(strings, h1, eri, 2, (1, 1))
    np.testing.assert_array_equal(accepted_h1, h1)
    np.testing.assert_array_equal(accepted_eri, eri)


def test_initial_state_projects_using_labels_not_array_position():
    source = SCIState(np.array([[1., 2.], [3., 4.]]), np.array([2, 1]), np.array([4, 1]), 3, (1, 1))
    target = (np.array([1, 2, 4]), np.array([1, 2]))
    projected = project_initial_state(source, target, 3, (1, 1))
    expected = np.array([[4., 0.], [2., 0.], [0., 0.]])
    np.testing.assert_allclose(projected, expected / np.linalg.norm(expected))
    assert project_initial_state(source, (np.array([4]), np.array([2])), 3, (1, 1)) is None
    with pytest.raises(ValueError, match="incompatible"):
        project_initial_state(source, target, 4, (1, 1))


def test_interleaved_phase_matches_explicit_fermion_permutation():
    pytest.importorskip("torch")
    norb = 5
    labels = (fci.cistring.make_strings(range(norb), 3), fci.cistring.make_strings(range(norb), 2))
    phases = interleaved_phase(labels, norb, device="cpu").numpy()
    expected = np.empty_like(phases)
    for ia, a in enumerate(labels[0]):
        for ib, b in enumerate(labels[1]):
            # 显式构造创建算符排列，统计把它排序为交错顺序的逆序数。
            occupied = [2 * p for p in range(norb) if int(a) & (1 << p)]
            occupied += [2 * p + 1 for p in range(norb) if int(b) & (1 << p)]
            inversions = sum(x > y for i, x in enumerate(occupied) for y in occupied[i + 1:])
            expected[ia, ib] = (-1) ** inversions
    np.testing.assert_array_equal(phases, expected)
    np.testing.assert_array_equal(phases * phases, np.ones_like(phases))


def test_raw_rdms_keep_asymmetric_alpha_beta_blocks():
    lib.num_threads(2)
    norb, nelec = 3, (2, 1)
    coeff = np.random.default_rng(17).normal(size=(3, 3))
    coeff /= np.linalg.norm(coeff)
    dm1 = fci.direct_spin1.make_rdm1s(coeff, norb, nelec)
    dm2 = fci.direct_spin1.make_rdm12s(coeff, norb, nelec)[1]
    raw = {
        "one_p_rdm": [x.reshape(-1, order="F") for x in dm1],
        "two_p_rdm": [x.transpose(0, 2, 1, 3).reshape(-1, order="F")
                      for x in (dm2[0], dm2[1].transpose(2, 3, 0, 1), dm2[1], dm2[2])],
    }
    converted = convert_raw_rdms(raw, norb)
    for actual, expected in zip((*converted[0], *converted[1]), (*dm1, *dm2)):
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert not np.allclose(dm2[1], dm2[1].transpose(2, 3, 0, 1))


def test_independent_cpu_validation_and_rdm_invariants():
    norb, nelec = 3, (2, 1)
    strings = tuple(fci.cistring.make_strings(range(norb), n) for n in nelec)
    h1 = np.diag([-2., -1., 1.])
    eri = np.zeros((norb,) * 4)
    coeff = np.zeros((3, 3))
    coeff[0, 0] = 1.
    cpu = independent_cpu_validation(coeff, strings, h1, eri, norb, nelec,
                                     spin_sq=None, spin_shift=.1, threads=2)
    result = assemble_result(coeff, strings, h1, eri, norb, nelec, *rdms_from_cpu(cpu))
    assert result.energy == pytest.approx(-5.)
    report = validate_rdms(result, -5., energy_tolerance=1e-8)
    assert report["rdm_contraction_max_abs_error"] < 1e-14
    checked = compare_cpu(result, cpu, cpu["cpu_image"], residual_tolerance=1e-7, energy_tolerance=1e-8)
    assert checked["rdm_cpu_max_abs_error"] == 0
    # 构造 ab/ba 间相互抵消的错误：总 RDM、能量和粒子数均看不出来，
    # full_cpu 必须逐个自旋块比较，避免重现上游 mixed-spin 累加错误。
    wrong_ab = result.rdm2s[1].copy()
    for index, value in [((0, 1, 1, 2), .1), ((1, 2, 0, 1), -.1),
                         ((1, 0, 2, 1), .1), ((2, 1, 1, 0), -.1)]:
        wrong_ab[index] += value
    hidden_error = assemble_result(coeff, strings, h1, eri, norb, nelec,
                                   result.rdm1s, (result.rdm2s[0], wrong_ab, result.rdm2s[2]))
    np.testing.assert_allclose(hidden_error.rdm2, result.rdm2, atol=1e-14)
    assert hidden_error.energy == result.energy
    validate_rdms(hidden_error, -5., energy_tolerance=1e-8)
    with pytest.raises(RuntimeError, match="rdm_cpu_max_abs_error"):
        compare_cpu(hidden_error, cpu, cpu["cpu_image"], residual_tolerance=1e-7, energy_tolerance=1e-8)
    wrong_image = cpu["cpu_image"].copy()
    wrong_image[1] += 0.01
    with pytest.raises(RuntimeError, match="Independent PySCF"):
        compare_cpu(result, cpu, wrong_image, residual_tolerance=1e-7, energy_tolerance=1e-8)
    result.rdm2s[1][0, 0, 0, 0] += 0.1
    with pytest.raises(RuntimeError, match="RDM validation"):
        validate_rdms(result, -5., energy_tolerance=1e-8)


def test_native_spin_constraint_rejected_before_cuda_import():
    with pytest.raises(ValueError, match="hybrid"):
        solve_sci_sbd(None, None, None, 2, (1, 1), spin_sq=0.)


def test_gpu_validation_rejects_inconsistent_energy_and_nonfinite_data():
    strings = (np.array([1]), np.array([0]))
    a = np.diag([1., 0.])
    zeros = np.zeros((2,) * 4)
    result = assemble_result(np.ones((1, 1)), strings, np.diag([1., 2.]), zeros,
                             2, (1, 0), (a, np.zeros((2, 2))), (zeros.copy(), zeros.copy(), zeros.copy()))
    validate_rdms(result, 1., energy_tolerance=1e-8)
    with pytest.raises(RuntimeError, match="RDM energy"):
        validate_rdms(result, 1.1, energy_tolerance=1e-8)
    with pytest.raises(RuntimeError, match="nonfinite"):
        validate_rdms(result, float("nan"), energy_tolerance=1e-8)
    result.sci_state.amplitudes[0, 0] = 0.5
    with pytest.raises(RuntimeError, match="normalization"):
        validate_rdms(result, 1., energy_tolerance=1e-8)


def test_cpu_rdm_only_mode_does_not_claim_operator_validation():
    strings = (np.array([1]), np.array([1]))
    cpu = independent_cpu_validation(np.ones((1, 1)), strings, np.diag([-1., 1.]),
                                     np.zeros((2,) * 4), 2, (1, 1), spin_sq=None,
                                     spin_shift=.1, threads=2, include_operator=False)
    assert cpu["cpu_image"].size == 0
    assert np.isnan(cpu["bare_cpu_rayleigh"])
    assert float(cpu["physical_energy"]) == pytest.approx(-2.)


def test_cuda_backend_name_rejected_before_device_check():
    pytest.importorskip("torch")
    from tyxonq.applications.chem.algorithms.sqd.gpu.torch_sci_solver import solve_sci_cuda

    with pytest.raises(ValueError, match="Unsupported"):
        solve_sci_cuda(None, None, None, 2, (1, 1), threads=2, chunk_size=16, tolerance=1e-8,
                       max_cycles=100, max_space=20, operator_residual_tolerance=1e-7,
                       rdm_energy_tolerance=1e-8, hamiltonian_backend="invalid")


def test_reused_context_requires_exact_space_device_and_memory_budget():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    updates = []
    context = SBDContractor.__new__(SBDContractor)
    context.norb, context.nelec = 3, (2, 1)
    context.ci_strings = (np.array([3, 5]), np.array([1, 4]))
    context.phase = SimpleNamespace(device=torch.device("cuda:0"))
    context.memory_fraction = .8
    context.context = SimpleNamespace(update_integrals=lambda h1, eri: updates.append((h1.copy(), eri.copy())))
    context.assert_compatible(context.ci_strings, 3, (2, 1), "cuda:0", memory_fraction=.8)
    cases = [((np.array([3, 6]), np.array([1, 4])), 3, (2, 1), "cuda:0"),
             (context.ci_strings, 4, (2, 1), "cuda:0"),
             (context.ci_strings, 3, (1, 2), "cuda:0"),
             (context.ci_strings, 3, (2, 1), "cuda:1")]
    for args in cases:
        with pytest.raises(ValueError, match="does not match"):
            context.assert_compatible(*args)
    with pytest.raises(ValueError, match="budget"):
        context.assert_compatible(context.ci_strings, 3, (2, 1), "cuda:0", memory_fraction=.5)
    context.update_integrals(np.eye(3), np.zeros((3,) * 4))
    context.update_integrals(2 * np.eye(3), np.zeros((3,) * 4))
    assert len(updates) == 2
    np.testing.assert_array_equal(updates[1][0], 2 * updates[0][0])


def test_device_spin_expectation_matches_rdms_for_truncated_space():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    lib.num_threads(2)
    strings = (np.array([3, 5]), np.array([1, 4]))
    coefficients = np.array([[1., 2.], [-3., 4.]])
    coefficients /= np.linalg.norm(coefficients)
    state = SCIState(coefficients, *strings, 3, (2, 1))
    result = assemble_result(coefficients, strings, np.eye(3), np.zeros((3,) * 4),
                             3, (2, 1), state.rdm(1), state.rdm(2))
    tensor = torch.as_tensor(coefficients, dtype=torch.float64)
    report = validate_spin_square(tensor, result)
    assert report["spin_square_rdm_abs_error"] < 1e-12
    with pytest.raises(RuntimeError, match="S\\^2"):
        validate_spin_square(tensor, result, contractor=SimpleNamespace(contract=torch.zeros_like))


@pytest.mark.parametrize("nelec,spin_target", [((1, 0), 0.), ((0, 1), 2.), ((2, 0), 4.), ((0, 2), 2.)])
def test_empty_spin_sector_cpu_reference_preserves_space_and_spin(nelec, spin_target):
    from pyscf import ao2mo

    lib.num_threads(2)
    norb = 4
    rng = np.random.default_rng(14)
    h1 = rng.normal(size=(norb, norb))
    h1 = (h1 + h1.T) / 2
    pair = rng.normal(size=(norb * (norb + 1) // 2,) * 2)
    eri = ao2mo.restore(1, (pair + pair.T) / 2, norb)
    full_strings = tuple(fci.cistring.make_strings(range(norb), n) for n in nelec)
    # 截断非空 spin 组态，验证旁观电子方法没有扩充求解波函数空间。
    strings = tuple(x[::2] for x in full_strings)
    coefficients = rng.normal(size=tuple(map(len, strings)))
    coefficients /= np.linalg.norm(coefficients)
    indexes = tuple(np.searchsorted(full, part) for full, part in zip(full_strings, strings))
    full = np.zeros(tuple(map(len, full_strings)))
    full[np.ix_(*indexes)] = coefficients
    h2 = fci.direct_spin1.absorb_h1e(h1, eri, norb, nelec, .5)
    bare_image = fci.direct_spin1.contract_2e(h2, full, norb, nelec)[np.ix_(*indexes)]
    cpu = independent_cpu_validation(coefficients, strings, h1, eri, norb, nelec,
                                     spin_sq=spin_target, spin_shift=.3, threads=2)
    s = .5 * sum(nelec)
    delta = s * (s + 1.) - spin_target
    if spin_target >= s * (s + 1.) + .1:
        delta **= 2
    expected = bare_image + .3 * delta * coefficients
    np.testing.assert_allclose(cpu["cpu_image"], expected.reshape(-1), atol=1e-12)
    np.testing.assert_allclose(cpu["bare_cpu_rayleigh"], np.sum(coefficients * bare_image), atol=1e-12)
    reference_rdms = fci.direct_spin1.make_rdm12s(full, norb, nelec)
    for actual, reference in zip((*rdms_from_cpu(cpu)[0], *rdms_from_cpu(cpu)[1]),
                                  (*reference_rdms[0], *reference_rdms[1])):
        np.testing.assert_allclose(actual, reference, atol=1e-12)
    torch = pytest.importorskip("torch")
    from tyxonq.applications.chem.algorithms.sqd.gpu.cpu_validate_sci import add_zero_coupling_spectator
    from tyxonq.applications.chem.algorithms.sqd.gpu.torch_selected_ci import (
        TorchSelectedCIContractor, absorb_h1e_torch,
    )

    ext_strings, ext_h1, ext_eri, ext_norb, ext_nelec = add_zero_coupling_spectator(
        strings, h1, eri, norb, nelec)
    links = fci.selected_ci._all_linkstr_index(ext_strings, ext_norb, ext_nelec)
    ext_h2 = absorb_h1e_torch(torch.as_tensor(ext_h1), torch.as_tensor(ext_eri), ext_nelec)
    contractor = TorchSelectedCIContractor(ext_h2, links, ext_nelec, chunk_size=2)
    actual = contractor.contract(torch.as_tensor(coefficients)).numpy()
    np.testing.assert_allclose(actual, bare_image, atol=1e-12)
    with pytest.raises(ValueError, match="spectator"):
        TorchSelectedCIContractor(torch.as_tensor(eri), None, nelec)


@pytest.mark.parametrize("failure_stage", ["kernel_rdms", "residual", "rdms", "cpu"])
def test_native_validation_failure_snapshot(monkeypatch, tmp_path, failure_stage):
    """不需要 CUDA：模拟三个验收失败点，验证输入和当前算符像可恢复。"""
    import json
    from types import SimpleNamespace
    from tyxonq.applications.chem.algorithms.sqd.gpu import sbd_solver, torch_sci_solver

    torch = pytest.importorskip("torch")
    coefficients = np.array([[1.], [0.]])
    image = np.array([[0.], [1. if failure_stage == "residual" else 0.]])
    raw = {"coefficients": coefficients, "hamiltonian_image": image,
           "converged": True, "iterations": 1,
           "timings": {"davidson_seconds": 0., "rdm_seconds": 0.}}
    def native_solve(**kwargs):
        if failure_stage == "kernel_rdms":
            raise RuntimeError("mock native RDM kernel failure")
        return raw

    contractor = SimpleNamespace(shape=(2, 1), phase=torch.ones((2, 1), dtype=torch.float64),
                                 context=SimpleNamespace(solve=native_solve),
                                 _raw=lambda value: value)
    monkeypatch.setattr(sbd_solver, "SBDContractor", lambda *args, **kwargs: contractor)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("set_device", "reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (10**12, 10**12))
    real_as_tensor = torch.as_tensor
    monkeypatch.setattr(torch, "as_tensor", lambda data, **kwargs:
                        real_as_tensor(data, **{k: v for k, v in kwargs.items() if k != "device"}))
    monkeypatch.setattr(torch_sci_solver, "wait_for_davidson_memory", lambda *args, **kwargs: (2, 10**12))
    monkeypatch.setattr(sbd_solver, "independent_cpu_validation", lambda *args, **kwargs:
                        {"cpu_image": image.reshape(-1), "seconds": 0.})
    monkeypatch.setattr(sbd_solver, "convert_raw_rdms", lambda *args: (None, None))
    monkeypatch.setattr(sbd_solver, "assemble_result", lambda *args: object())

    def validate(*args, **kwargs):
        if failure_stage == "rdms":
            raise RuntimeError("mock RDM validation failure")
        return {}

    def compare(*args, **kwargs):
        raise RuntimeError("mock CPU validation failure")

    monkeypatch.setattr(sbd_solver, "validate_rdms", validate)
    monkeypatch.setattr(sbd_solver, "compare_cpu", compare)
    strings = (np.array([1, 2]), np.array([1]))
    initial = SCIState(coefficients, *strings, 2, (1, 1))
    h1, eri = np.zeros((2, 2)), np.zeros((2,) * 4)
    debug_path = tmp_path / "failure.npz"
    events = []
    for path in (None, debug_path):
        with pytest.raises(RuntimeError, match="converge|mock"):
            sbd_solver.solve_sci_sbd(strings, h1, eri, 2, (1, 1), initial_state=initial,
                                    failure_debug_path=path, progress_callback=events.append)
        if path is None:
            assert not list(tmp_path.iterdir())
    assert not any(event["status"] == "complete" for event in events)
    with np.load(debug_path, allow_pickle=False) as data:
        assert {"coefficients", "ci_strs_a", "ci_strs_b", "h1e", "eri", "gpu_image",
                "cpu_image", "metadata_json"} <= set(data.files)
        metadata = json.loads(str(data["metadata_json"]))
        assert metadata["norb"] == 2 and metadata["nelec"] == [1, 1]
        assert metadata["coefficients_normalized"]
        restored = SCIState(data["coefficients"].reshape(2, 1), data["ci_strs_a"],
                            data["ci_strs_b"], metadata["norb"], tuple(metadata["nelec"]))
        np.testing.assert_array_equal(restored.amplitudes, coefficients)
        if failure_stage == "kernel_rdms":
            assert metadata["coefficients_role"] == "initial_guess"
            assert data["gpu_image"].size == 0
        else:
            assert metadata["coefficients_role"] == "solver_result"
            np.testing.assert_array_equal(data["gpu_image"], image)
        assert data["cpu_image"].size == (0 if failure_stage in ("kernel_rdms", "residual") else 2)
    if failure_stage == "kernel_rdms":
        from tyxonq.applications.chem.algorithms.sqd.gpu.validation import validation_failure_snapshot

        success_path = tmp_path / "success.npz"
        with validation_failure_snapshot(success_path, strings, h1, eri, 2, (1, 1)) as snapshot:
            snapshot["coefficients"] = coefficients
        assert not success_path.exists()


def _blocking_cpu_validation_worker(wrapper, pid_path):
    """真实启动验证 CLI；仅测试包装器将读取输入延后以模拟长任务。"""
    from tyxonq.applications.chem.algorithms.sqd.gpu import validation

    real_run = validation.subprocess.run

    def run(command, **kwargs):
        return real_run([command[0], wrapper, pid_path, *command[1:]], **kwargs)

    validation.subprocess.run = run
    validation.independent_cpu_validation(
        np.ones((1, 1)), (np.array([1]), np.array([1])), np.ones((1, 1)),
        np.zeros((1, 1, 1, 1)), 1, (1, 1), spin_sq=None, spin_shift=.1, threads=1)


def _validation_process_running(pid):
    """仅查询测试创建的 PID；Windows 不使用有终止语义的 os.kill(pid, 0)。"""
    import os
    import sys
    from pathlib import Path

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:
                return False
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            exit_code = wintypes.DWORD()
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return exit_code.value == 259
        finally:
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle(handle)
    if sys.platform.startswith("linux"):
        try:
            if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return False
        except FileNotFoundError:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.parametrize("stop", ["terminate", "kill"])
def test_cpu_validation_exits_when_spawn_worker_dies(tmp_path, stop):
    """检验真实 spawn/subprocess 生命周期，不用 CUDA，也不留下孤立 CPU 进程。"""
    import multiprocessing as mp
    import os
    import signal
    import time
    from tyxonq.applications.chem.algorithms.sqd.gpu.batch import _DeviceBatchPool

    wrapper = tmp_path / "blocking_validator.py"
    wrapper.write_text('''import os, pathlib, runpy, sys, time
import numpy as np
pid_path, command = sys.argv[1], sys.argv[2:]
real_load = np.load
def delayed_load(*args, **kwargs):
    pathlib.Path(pid_path).write_text(str(os.getpid()))
    time.sleep(60)
    return real_load(*args, **kwargs)
np.load = delayed_load
sys.argv = command
runpy.run_path(command[0], run_name="__main__")
''')
    pid_path = tmp_path / "validator.pid"
    worker = mp.get_context("spawn").Process(
        target=_blocking_cpu_validation_worker, args=(str(wrapper), str(pid_path)))
    # 复用真实 close 实现，构造时不启动 GPU worker，也不注册额外 atexit。
    pool = _DeviceBatchPool.__new__(_DeviceBatchPool)
    pool.processes = [worker]
    child_pid = None
    worker.start()
    try:
        deadline = time.monotonic() + 30
        while not pid_path.exists():
            assert worker.is_alive(), "validation worker exited before the child became ready"
            assert time.monotonic() < deadline, "CPU validation child failed to start"
            time.sleep(.05)
        child_pid = int(pid_path.read_text())
        assert _validation_process_running(child_pid)
        if stop == "kill":
            worker.kill()
            worker.join(timeout=5)
        pool.close()
        deadline = time.monotonic() + 5
        while _validation_process_running(child_pid) and time.monotonic() < deadline:
            time.sleep(.05)
        assert not _validation_process_running(child_pid), "orphaned CPU validation process remains"
        assert not worker.is_alive()
    finally:
        pool.close()
        # 失败时也只清理本测试登记的子进程，避免缺陷回归导致长时间残留。
        if child_pid is not None and _validation_process_running(child_pid):
            os.kill(child_pid, signal.SIGTERM)


def test_cpu_validator_manual_cli_keeps_normal_stdin_behavior(tmp_path):
    """默认 import 不启线程；未选守护参数时，关闭 stdin 不会提前退出 CLI。"""
    import json
    import subprocess
    import sys
    import threading
    from tyxonq.applications.chem.algorithms.sqd.gpu import cpu_validate_sci

    assert not any(getattr(thread, "_target", None) is cpu_validate_sci._watch_parent_stdin
                   for thread in threading.enumerate())
    input_path, output_path = tmp_path / "input.npz", tmp_path / "output.npz"
    np.savez(input_path, coefficients=np.ones(1), ci_strs_a=np.array([1]), ci_strs_b=np.array([1]),
             h1e=np.ones((1, 1)), eri=np.zeros((1, 1, 1, 1)),
             metadata_json=np.asarray(json.dumps({"norb": 1, "nelec": [1, 1],
                                                  "spin_sq": None, "spin_shift": .1})))
    completed = subprocess.run([sys.executable, cpu_validate_sci.__file__, "--input", str(input_path),
                                "--output", str(output_path), "--threads", "1"],
                               stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    with np.load(output_path, allow_pickle=False) as data:
        assert float(data["physical_energy"]) == pytest.approx(2.)
