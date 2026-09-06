# compiler 测试合并：编译 API 与 provider（default/qiskit）、native 编译流水线
# （build_plan 的 no-op/parameter_shift stage 与未知 stage 报错）、shot 调度 stage，
# 以及 MPO 到矩阵的翻译（contract_mpo_to_matrix）。
from __future__ import annotations

import importlib

import numpy as np
import pytest

from tyxonq.compiler.api import compile as compile_ir
from tyxonq.compiler.compile_engine.native.compile_plan import build_plan
from tyxonq.compiler.stages.scheduling.shot_scheduler import ShotSchedulerPass, schedule
from tyxonq.compiler.translation.mpo_converters import contract_mpo_to_matrix
from tyxonq.core.ir import Circuit


def _make_identity_mpo(n: int):
    # Build MPO tensors for identity: each site (1,2,2,1) with identity on physical legs
    Ts = []
    I = np.eye(2, dtype=np.complex128)
    for _ in range(n):
        T = np.zeros((1, 2, 2, 1), dtype=np.complex128)
        T[0, :, :, 0] = I
        Ts.append(T)
    return Ts


def test_compile_with_default_provider_returns_ir():
    circ = Circuit(num_qubits=1, ops=[("h", 0), ("measure_z", 0)])
    res = compile_ir(circ, compile_engine="default", output="ir")
    compiled = res["circuit"]
    # Accept either identity or structurally equivalent IR
    assert isinstance(compiled, Circuit)
    assert compiled.num_qubits == circ.num_qubits
    assert list(compiled.ops) == list(circ.ops)


@pytest.mark.skipif(importlib.util.find_spec("qiskit") is None, reason="qiskit not installed")
def test_compile_with_qiskit_provider_returns_qc():
    from qiskit import QuantumCircuit

    circ = Circuit(num_qubits=1, ops=[("h", 0), ("measure_z", 0)])
    res = compile_ir(circ, compile_engine="qiskit", output="qiskit", options={"opt_level": 1})
    assert isinstance(res["circuit"], QuantumCircuit)


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


def test_shot_scheduler_schedule_structure():
    circ = Circuit(num_qubits=1, ops=[])
    plan = schedule(circ, [100, 200])
    assert plan["circuit"] is circ
    assert [seg["shots"] for seg in plan["segments"]] == [100, 200]


def test_shot_scheduler_pass_validates():
    p = ShotSchedulerPass()
    circ = Circuit(num_qubits=1, ops=[])
    # valid
    p.execute_plan(circ, device_rule={}, shot_plan=[1, 2, 3])
    # invalid
    with pytest.raises(ValueError):
        p.execute_plan(circ, device_rule={}, shot_plan=[0, -1])


def test_shot_scheduler_respects_max_shots_per_job():
    circ = Circuit(num_qubits=1, ops=[("measure_z", 0)])
    plan = schedule(circ, shot_plan=None, total_shots=23, device_rule={"max_shots_per_job": 7, "supports_batch": True, "max_segments_per_batch": 3})
    assert sum(seg.get("shots", 0) for seg in plan["segments"]) == 23
    assert all(seg.get("shots", 0) <= 7 for seg in plan["segments"])  # split into <=7 per segment
    # batching annotates batch_id every up to 3 segments
    assert all("batch_id" in seg for seg in plan["segments"]) and max(seg["batch_id"] for seg in plan["segments"]) >= 0


def test_contract_mpo_identity_gives_identity_matrix():
    Ts = _make_identity_mpo(3)
    M = contract_mpo_to_matrix(Ts)
    dim = 2 ** 3
    np.testing.assert_allclose(M, np.eye(dim, dtype=np.complex128))
