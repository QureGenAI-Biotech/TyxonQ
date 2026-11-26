"""SpinQLab 波形参数适配的纯离线测试。"""

from __future__ import annotations

import json
import math

import pytest

from tyxonq.compiler.platforms.spinq.waveform_adapter import (
    native_pulse_fields,
    program_to_spinqlab_parameters,
    spinqlab_parameters_from_jsonable,
    spinqlab_parameters_to_jsonable,
    validate_spinqlab_parameters,
)
from tyxonq.core.ir.pulse import PulseProgram


def test_constant_waveform_maps_to_native_spinqlab_pulse():
    program = PulseProgram(1).constant(0, amp=0.5, duration=80000, phase=math.pi / 2)
    draft = program_to_spinqlab_parameters(program)
    pulse = draft["pulses"][0]
    assert pulse["native"] is True
    assert pulse["width"] == pytest.approx(40.0)
    assert pulse["amplitude"] == pytest.approx(50.0)
    assert pulse["phase"] == pytest.approx(90.0)
    assert native_pulse_fields(pulse) == {
        "path": 0, "phase": 90.0, "amplitude": 50.0, "width": 40.0, "detuning": 0.0
    }
    assert draft["feasibility"]["native_pulse_only"] is True


def test_gaussian_is_sampling_draft_not_native_pulse():
    program = PulseProgram(1).gaussian(0, amp=0.2, duration=64, sigma=16.0)
    draft = program_to_spinqlab_parameters(program)
    pulse = draft["pulses"][0]
    assert pulse["native"] is False
    assert len(pulse["samples"]) == 64
    assert len(pulse["samples"][0]) == 2
    assert draft["feasibility"]["requires_shape_pulse_validation"] is True
    with pytest.raises(ValueError, match="矩形参数"):
        native_pulse_fields(pulse)


def test_json_roundtrip():
    program = PulseProgram(1).gaussian(0, amp=0.1, duration=16, sigma=4.0)
    draft = program_to_spinqlab_parameters(program)
    blob = json.dumps(spinqlab_parameters_to_jsonable(draft), allow_nan=False)
    loaded = spinqlab_parameters_from_jsonable(json.loads(blob))
    assert loaded == draft


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("pulses", 0, "amplitude"), 101, "amplitude"),
        (("pulses", 0, "path"), 2, "path"),
        (("pulses", 0, "width"), 0, "width"),
        (("pulses", 0, "phase"), math.nan, "有限"),
        (("pulses", 0, "detuning"), -1, "detuning"),
    ],
)
def test_validation_rejects_unsafe_values(path, value, message):
    draft = program_to_spinqlab_parameters(PulseProgram(1).constant(0, amp=0.5, duration=80))
    target = draft
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_spinqlab_parameters(draft)


def test_z_channel_and_empty_program_are_rejected():
    with pytest.raises(ValueError, match="为空"):
        program_to_spinqlab_parameters(PulseProgram(1))
    program = PulseProgram(1).constant(0, amp=0.5, duration=80, channel="z")
    with pytest.raises(ValueError, match="xy"):
        program_to_spinqlab_parameters(program)
