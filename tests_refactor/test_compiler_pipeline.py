import pytest

from tyxonq.compiler.pipeline import build_pipeline
from tyxonq.core.ir import Circuit


def test_build_and_run_pipeline_noop_stages():
    pl = build_pipeline([
        "decompose",
        "rewrite/measurement",
        "layout",
        "scheduling",
        "scheduling/shot_scheduler",
    ])
    circ = Circuit(num_qubits=2, ops=[("h", 0)])
    out = pl.run(circ, caps={})
    assert out is circ  # no-op stages should return the same instance for now


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        build_pipeline(["unknown_stage"])


