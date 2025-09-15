import pytest

from tyxonq.compiler.compile_engine.native.compile_plan import build_plan
from tyxonq.core.ir import Circuit


def test_build_and_run_pipeline_noop_stages():
    pl = build_plan([
        "decompose",
        "rewrite/measurement",
        "layout",
        "scheduling",
        "scheduling/shot_scheduler",
    ])
    circ = Circuit(num_qubits=2, ops=[("h", 0)])
    out = pl.execute_plan(circ, device_rule={})
    assert out is circ  # no-op stages should return the same instance for now


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        build_plan(["unknown_stage"])


def test_pipeline_parameter_shift_stage_populates_metadata():
    circ = Circuit(num_qubits=1, ops=[("rz", 0, 0.1), ("measure_z", 0)])
    pipe = build_plan(["gradients/parameter_shift"])  # type: ignore[list-item]
    out = pipe.execute_plan(circ, device_rule={}, grad_op="rz")
    g = out.metadata["gradients"]["rz"]
    assert g["plus"].ops[0][2] != g["minus"].ops[0][2]
    assert g["meta"]["coeff"] == 0.5


