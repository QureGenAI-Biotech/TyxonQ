from __future__ import annotations

import pytest


def test_simulator_list_devices_imports():
    import tyxonq as tq

    devs = tq.api.list_devices(provider="simulator")
    assert isinstance(devs, list)
    assert any("simulator::matrix_product_state" in d for d in devs)


def _toy_circuit(n: int = 1):
    # Build a minimal IR Circuit: H on q0 then measure Z
    from tyxonq.core.ir import Circuit

    ops = [("h", 0), ("measure_z", 0)]
    return Circuit(num_qubits=n, ops=ops)


@pytest.mark.parametrize(
    "sim_device",
    [
        "matrix_product_state",
        "statevector",
        "density_matrix",
    ],
)
def test_simulator_submit_run_result_cancel(sim_device: str):
    import tyxonq as tq

    # Resolve simulator device
    meta = tq.api.device(provider="simulator", id=sim_device)
    device_name = meta["device"]

    c = _toy_circuit(1)

    # submit_task
    tasks = tq.api.submit_task(provider="simulator", device=device_name, circuit=c, shots=128)
    assert isinstance(tasks, list) and len(tasks) == 1
    t = tasks[0]

    # run is alias of submit_task
    tasks2 = tq.api.run(provider="simulator", device=device_name, circuit=c, shots=64)
    assert isinstance(tasks2, list) and len(tasks2) == 1

    # result is alias of get_task_details
    details = tq.api.result(t)
    assert isinstance(details, dict)
    # Accept both new and legacy result shapes
    assert (
        "result" in details or "results" in details or "expectations" in details
    )

    # cancel is no-op for simulator but should return a dict
    cancelled = tq.api.cancel(t)
    assert isinstance(cancelled, dict)
    assert cancelled.get("state") == "cancelled"


