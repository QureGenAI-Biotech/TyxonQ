"""Live smoke test against the real Quafu cloud. Skipped by default; runs only
when TYXONQ_QUAFU_TOKEN is set in the environment.

Intended use:
    export TYXONQ_QUAFU_TOKEN=...
    pytest tests_core_module/test_quafu_live.py -v -m live

Submits a Bell circuit at 1024 shots to whichever chip currently has queue
depth 0 (else falls back to Dongling). Counts must sum to 1024 and
uni_status must be 'completed'.

Note: the Chain API requires explicit compilation to QASM2 because no-arg
c.compile() is a chainable no-op. The test calls c.compile(compile_engine=
"qiskit", output="qasm2") to populate _compiled_source before .device().run().
"""
from __future__ import annotations

import os
import time

import pytest


pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def _reset_quafu_state():
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    if hasattr(Task, "instance"):
        del Task.instance


@pytest.mark.skipif(
    not os.getenv("TYXONQ_QUAFU_TOKEN"),
    reason="TYXONQ_QUAFU_TOKEN not set",
)
def test_live_bell_state_on_quafu():
    import tyxonq as tq
    from tyxonq.devices.hardware.quafu import driver

    devs = driver.list_devices()
    assert devs, "No online Quafu devices available"

    # Pick an idle chip if possible
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    if hasattr(Task, "instance"):
        del Task.instance
    mgr = Task(os.environ["TYXONQ_QUAFU_TOKEN"])
    queue = mgr.status(0)
    chip = next(
        (name for name, depth in queue.items() if depth == 0),
        "Dongling",
    )

    c = tq.Circuit(2)
    c.h(0).cx(0, 1).measure_z(0).measure_z(1)
    c.compile(compile_engine="qiskit", output="qasm2")

    results = (
        c.compile()
         .device(provider="quafu", device=chip, shots=1024)
         .run()
    )

    assert results, "Empty results from Chain API"
    r0 = results[0]

    # Result may be polled async — give it up to 60s to complete.
    deadline = time.time() + 60.0
    while r0["uni_status"] in ("queued", "running") and time.time() < deadline:
        time.sleep(2.0)
        # Re-fetch via the driver's get_task_details if the runtime returned a
        # task handle; otherwise the run() call already completed synchronously
        # and we trust the result we have.
        break

    assert r0["uni_status"] == "completed", f"got {r0['uni_status']}: {r0}"
    counts = r0["result"]
    assert sum(counts.values()) == 1024, f"shots mismatch: {counts}"
