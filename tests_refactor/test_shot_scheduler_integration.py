from tyxonq.compiler.compile_engine.native.compile_plan import build_plan
from tyxonq.compiler.stages.scheduling.shot_scheduler import schedule
from tyxonq.core.ir import Circuit
from tyxonq.core.measurements import Expectation


def test_group_based_shot_scheduling_distribution():
    circ = Circuit(num_qubits=2, ops=[])
    ms = [Expectation(obs="Z", wires=(0,)), Expectation(obs="X", wires=(1,)), Expectation(obs="ZZ", wires=(0, 1))]

    pl = build_plan(["rewrite/measurement"])
    circ = pl.execute_plan(circ, device_rule={}, measurements=ms)

    plan = schedule(circ, total_shots=1000)
    assert "segments" in plan and len(plan["segments"]) >= 2
    assert sum(seg["shots"] for seg in plan["segments"]) in (999, 1000, 1001)  # rounding tolerance
    # segments should carry basis and wires for device executor awareness
    assert all("basis" in seg and "wires" in seg for seg in plan["segments"])


