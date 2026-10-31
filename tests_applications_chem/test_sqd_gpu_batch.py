"""不依赖 CUDA 的批次顺序、显式后端选择及进程故障回归。"""

import os
import time

import pytest

from tyxonq.applications.chem.algorithms.sqd.gpu import batch


def _fake_worker(device, tasks, messages, kind, threads):
    """真实 spawn 进程，使用简单整数任务代替 GPU 求解。"""
    while True:
        item = tasks.get()
        if item is None:
            return
        index, value, *unused = item
        messages.put(("started", index, device, None))
        if value == "exit":
            os._exit(3)
        if value == "error":
            messages.put(("error", index, device, "synthetic solver error"))
            continue
        if value == "timeout":
            time.sleep(20)
            continue
        time.sleep(0.05 if index == 0 else 0.001)
        messages.put(("progress", index, device, {"cycle": 1}))
        messages.put(("done", index, device, (value * 2, {"value": value})))


def test_factory_preserves_serial_contract_and_explicit_hybrid(monkeypatch):
    calls, reports = [], []

    def fake(kind, strings, h1e, eri, norb, nelec, options):
        calls.append((kind, strings, h1e, eri, norb, nelec, options))
        return strings * 2, {"converged": True}

    monkeypatch.setattr(batch, "_solve_one", fake)
    solve = batch.make_cuda_sci_solver(report_callback=reports.append, spin_sq=2)
    assert solve([3, 1], "h", "g", 4, (2, 1)) == [6, 2]
    assert calls[0][2:6] == ("h", "g", 4, (2, 1))
    assert calls[0][-1] == {"spin_sq": 2}
    assert [item["batch"] for item in reports] == [0, 1]
    hybrid = batch.make_sbd_sci_solver(mode="hybrid", spin_sq=2)
    hybrid([1], None, None, 4, (2, 1))
    assert calls[-1][-1] == {
        "spin_sq": 2, "hamiltonian_backend": "sbd", "rdm_backend": "sbd"
    }
    with pytest.raises(ValueError, match="spin_sq"):
        batch.make_sbd_sci_solver(spin_sq=0)
    with pytest.raises(ValueError, match="mode"):
        batch.make_sbd_sci_solver(mode="unknown")


def test_spawn_batches_keep_input_order_and_reuse_workers(monkeypatch):
    monkeypatch.setattr(batch, "_device_worker", _fake_worker)
    reports, progress = [], []
    pool = batch._DeviceBatchPool([4, 5], "hybrid", 2, 10)
    try:
        assert pool.solve([3, 2, 1], None, None, 2, (1, 1),
                          {"progress_callback": progress.append}, reports.append) == [6, 4, 2]
        pids = [process.pid for process in pool.processes]
        assert pool.solve([4], None, None, 2, (1, 1), {}, None) == [8]
        assert pids == [process.pid for process in pool.processes]
        assert [item["batch"] for item in reports] == [0, 1, 2]
        assert {item["batch"] for item in progress} == {0, 1, 2}
    finally:
        pool.close()
    assert not pool.processes


@pytest.mark.parametrize("value,error,match", [
    ("error", RuntimeError, "synthetic solver error"),
    ("exit", RuntimeError, "exited"),
    ("timeout", TimeoutError, "worker_timeout"),
])
def test_worker_failure_closes_only_owned_processes(monkeypatch, value, error, match):
    monkeypatch.setattr(batch, "_device_worker", _fake_worker)
    pool = batch._DeviceBatchPool([4], "hybrid", 2, 5 if value == "timeout" else 10)
    with pytest.raises(error, match=match):
        pool.solve([value], None, None, 2, (1, 1), {}, None)
    assert not pool.processes


def test_pool_rejects_invalid_resource_requests():
    with pytest.raises(ValueError, match="unique"):
        batch._DeviceBatchPool([4, 4], "native", 2, 600)
    with pytest.raises(ValueError, match="positive"):
        batch._DeviceBatchPool([4], "native", 0, 600)
    with pytest.raises(ValueError, match="context"):
        batch.make_sbd_sci_solver(device_ids=[4], sbd_context=object())
