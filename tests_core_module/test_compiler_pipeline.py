# compiler 测试合并：编译 API 与平台（tyxonq/qiskit + compile_engine 过渡别名）、
# gate_compiler 核心的类型化步骤装配（build_compile_steps/apply_steps），以及 MPO 到矩阵
# 的翻译（contract_mpo_to_matrix）。
from __future__ import annotations

import importlib

import numpy as np
import pytest

from tyxonq.compiler.api import compile as compile_ir
from tyxonq.compiler.gate_compiler.specs import DecomposeSpec
from tyxonq.compiler.gate_compiler.steps import apply_steps, build_compile_steps
from tyxonq.compiler.gate_compiler.translation.mpo_converters import contract_mpo_to_matrix
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


def _names(steps):
    return [str(getattr(s, "name", type(s).__name__)) for s in steps]


def test_compile_with_default_platform_returns_ir():
    circ = Circuit(num_qubits=1, ops=[("h", 0), ("measure_z", 0)])
    res = compile_ir(circ, platform="tyxonq", output="ir")
    compiled = res["circuit"]
    # Accept either identity or structurally equivalent IR
    assert isinstance(compiled, Circuit)
    assert compiled.num_qubits == circ.num_qubits
    assert list(compiled.ops) == list(circ.ops)


def test_compile_engine_transitional_alias_maps_to_platform():
    # compile_engine= 是 platform= 的过渡别名；default/native 归一到 tyxonq。
    circ = Circuit(num_qubits=1, ops=[("h", 0), ("measure_z", 0)])
    for alias in ("default", "native", "tyxonq"):
        res = compile_ir(circ, compile_engine=alias, output="ir")
        assert isinstance(res["circuit"], Circuit)
        # 核心默认流水线：规范化 + matchgate 分解
        assert res["metadata"]["steps"] == [
            "rewrite/auto_measure",
            "rewrite/gates_transform",
            "decompose/matchgate",
        ]


@pytest.mark.skipif(importlib.util.find_spec("qiskit") is None, reason="qiskit not installed")
def test_compile_with_qiskit_provider_returns_qc():
    from qiskit import QuantumCircuit

    circ = Circuit(num_qubits=1, ops=[("h", 0), ("measure_z", 0)])
    res = compile_ir(circ, platform="qiskit", output="qiskit", options={"opt_level": 1})
    assert isinstance(res["circuit"], QuantumCircuit)


def test_build_compile_steps_typed_assembly():
    # 默认：规范化(auto_measure+gates_transform) + matchgate 分解（核心旗舰特色）
    assert _names(build_compile_steps()) == [
        "rewrite/auto_measure",
        "rewrite/gates_transform",
        "decompose/matchgate",
    ]
    # decompose=False → 不追加任何分解步骤（不再有 NoOp 占位）
    assert _names(build_compile_steps(decompose=False)) == [
        "rewrite/auto_measure",
        "rewrite/gates_transform",
    ]
    # 细粒度：只开 rotations，不开 matchgate
    names = _names(build_compile_steps(decompose=DecomposeSpec(matchgate=False, rotations=True)))
    assert "decompose/rotations" in names and "decompose/matchgate" not in names
    # optimize>=1 追加 merge_prune；simplify 追加 lightcone
    names2 = _names(build_compile_steps(decompose=False, optimize=1, simplify=True))
    assert names2 == [
        "rewrite/auto_measure",
        "rewrite/gates_transform",
        "rewrite/merge_prune",
        "simplify/lightcone",
    ]


def test_apply_steps_runs_in_order():
    circ = Circuit(num_qubits=2, ops=[("h", 0), ("measure_z", 0)])
    steps = build_compile_steps(decompose=False)
    out = apply_steps(circ, steps, device_rule={})
    assert isinstance(out, Circuit)
    assert out.num_qubits == 2


def test_contract_mpo_identity_gives_identity_matrix():
    Ts = _make_identity_mpo(3)
    M = contract_mpo_to_matrix(Ts)
    dim = 2 ** 3
    np.testing.assert_allclose(M, np.eye(dim, dtype=np.complex128))
