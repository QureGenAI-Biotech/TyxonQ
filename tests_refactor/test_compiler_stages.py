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


