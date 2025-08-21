from tyxonq.core.ir import PulseInstruction, PulseSchedule


def test_pulse_instruction_fields():
    instr = PulseInstruction(
        channel="d0",
        start=0,
        duration=160,
        waveform=[0.0, 0.5, 1.0, 0.5, 0.0],
        metadata={"label": "gaussian", "amp": 0.5},
    )
    assert instr.channel == "d0"
    assert instr.start == 0
    assert instr.duration == 160
    assert len(instr.waveform) == 5
    assert instr.metadata["label"] == "gaussian"


def test_pulse_schedule_basic_append_and_timing():
    sched = PulseSchedule(sampling_rate_hz=4e9)
    sched.append(PulseInstruction(channel="d0", start=0, duration=10, waveform=[0.0]))
    sched.append(PulseInstruction(channel="d1", start=5, duration=5, waveform=[1.0]))
    assert len(sched.instructions) == 2
    # end time should be max(start+duration)
    assert sched.end_time() == 10


