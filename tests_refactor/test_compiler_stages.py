import pytest

from tyxonq.compiler.stages.scheduling.shot_scheduler import ShotSchedulerPass, schedule
from tyxonq.core.ir import Circuit


def test_shot_scheduler_schedule_structure():
    circ = Circuit(num_qubits=1, ops=[])
    plan = schedule(circ, [100, 200])
    assert plan["circuit"] is circ
    assert [seg["shots"] for seg in plan["segments"]] == [100, 200]


def test_shot_scheduler_pass_validates():
    p = ShotSchedulerPass()
    circ = Circuit(num_qubits=1, ops=[])
    # valid
    p.run(circ, caps={}, shot_plan=[1, 2, 3])
    # invalid
    with pytest.raises(ValueError):
        p.run(circ, caps={}, shot_plan=[0, -1])


def test_shot_scheduler_respects_max_shots_per_job():
    circ = Circuit(num_qubits=1, ops=[("measure_z", 0)])
    plan = schedule(circ, shot_plan=None, total_shots=23, caps={"max_shots_per_job": 7, "supports_batch": True, "max_segments_per_batch": 3})
    assert sum(seg.get("shots", 0) for seg in plan["segments"]) == 23
    assert all(seg.get("shots", 0) <= 7 for seg in plan["segments"])  # split into <=7 per segment
    # batching annotates batch_id every up to 3 segments
    assert all("batch_id" in seg for seg in plan["segments"]) and max(seg["batch_id"] for seg in plan["segments"]) >= 0


