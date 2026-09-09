"""国盾委托 gate_compiler 公用核心的单测（计划 §C4，mirror test_lqcloud_delegation.py）。

C 阶段起 ``GuodunCompiler.compile`` 先跑核心（强制分解 matchgate + rotations）得到编译 IR，
再 ``lower_circuit_to_gate_qcis`` 直译降门为门级 QCIS。本文件验证委托的关键契约：
  - 2-qubit matchgate **只在核心分解一次**（lowering 已删私有旋转/matchgate 副本），
    产物 QCIS 只含 {X,H,RZ,CZ,M}，不再因 unitary 而 raise（委托前国盾跑不了 SQD/LUCJ）；
  - 参数化旋转（rx/ry/rxx/ryy/rzz）同样由核心分解，lowering 不再自带这些恒等式；
  - physical_qubits 仍在**编译期**烘焙进 QCIS 文本（Q60/Q55）——这是 guodun 区别于
    lqcloud 运行期 initial_layout 的本质；
  - conversion / core_steps 元数据对齐 lqcloud。

需要可选依赖 cqlib==1.3.11（与 test_guodun_compiler.py 一致），否则整体跳过。
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from tyxonq.compiler.api import compile as compile_ir
from tyxonq.core.ir import Circuit


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cqlib") is None,
    reason="需要可选依赖 cqlib==1.3.11",
)

_ALLOWED_QCIS = {"X", "H", "RZ", "CZ", "M"}


def _compile(circuit, physical_qubits=None):
    options = {"output": "qcis"}
    if physical_qubits is not None:
        options["physical_qubits"] = physical_qubits
    return compile_ir(
        circuit, compile_engine="guodun", output="qcis", options=options
    )


def _instructions(qcis):
    """QCIS 文本 -> 每行首列的指令助记符列表。"""
    return [line.split()[0] for line in qcis.splitlines() if line.strip()]


def _two_mode_fock(v):
    """复现 LUCJ/SQD 的 two_mode_fock_matrix(v)：2x2 轨道旋转 -> 4x4 Fock 酉（matchgate）。"""
    v = np.asarray(v, dtype=complex)
    m = np.eye(4, dtype=complex)
    m[1, 1] = v[1, 1]
    m[1, 2] = v[1, 0]
    m[2, 1] = v[0, 1]
    m[2, 2] = v[0, 0]
    m[3, 3] = v[0, 0] * v[1, 1] - v[0, 1] * v[1, 0]
    return m


# --------------------------------------------------------------------------- #
# 2-qubit matchgate 只在核心分解一次
# --------------------------------------------------------------------------- #
def test_matchgate_decomposed_by_core_to_qcis():
    theta, phi = 0.4, 0.7
    v = np.array(
        [[np.cos(theta), -np.sin(theta) * np.exp(1j * phi)],
         [np.sin(theta) * np.exp(-1j * phi), np.cos(theta)]],
        dtype=complex,
    )
    circuit = (
        Circuit(2)
        .unitary(0, 1, matrix=_two_mode_fock(v))
        .measure_z(0)
        .measure_z(1)
    )
    result = _compile(circuit)
    qcis = result["compiled_source"]

    # 委托核心前，2-qubit unitary 会让 lowering 直接 raise；委托后被分解为门级 QCIS。
    assert set(_instructions(qcis)) <= _ALLOWED_QCIS
    # matchgate -> cx -> H·CZ·H，故必含 CZ；且核心确实装配了 matchgate 分解步骤。
    assert result["metadata"]["gate_stats"].get("CZ", 0) > 0
    assert "decompose/matchgate" in (result["metadata"]["core_steps"] or [])


def test_conversion_metadata_aligns_with_lqcloud():
    circuit = Circuit(2).h(0).cx(0, 1).add_measure(0, 1)
    meta = _compile(circuit)["metadata"]
    assert meta["conversion"] == "core_compile_then_ir_to_gate_qcis"
    # 核心痕迹并入：steps 含规范化 + 分解步骤，核心产物 output 为 IR。
    core_steps = meta["core_steps"] or []
    assert "rewrite/gates_transform" in core_steps
    assert "decompose/matchgate" in core_steps
    assert "decompose/rotations" in core_steps
    assert meta["core"]["output"] == "ir"


# --------------------------------------------------------------------------- #
# physical_qubits 编译期烘焙
# --------------------------------------------------------------------------- #
def test_physical_qubits_baked_at_compile_time():
    circuit = Circuit(2).h(0).cx(0, 1).add_measure(0, 1)
    result = _compile(circuit, [60, 55])
    qcis = result["compiled_source"]
    # 物理比特号直接写进 QCIS 文本（Q60/Q55），而非留作运行期 initial_layout。
    assert "Q60" in qcis and "Q55" in qcis
    assert result["metadata"]["logical_physical_mapping"] == {0: 60, 1: 55}
    assert result["metadata"]["physical_qubits"] == [60, 55]


# --------------------------------------------------------------------------- #
# 旋转由核心分解，lowering 不再自带副本
# --------------------------------------------------------------------------- #
def test_rotations_decomposed_by_core_not_lowering():
    """rx/ry/rxx/ryy/rzz 由核心 RotationsDecompose 分解；lowering 已删这些恒等式副本。"""
    circuit = (
        Circuit(2)
        .rx(0, 0.3)
        .ry(1, 0.5)
        .rxx(0, 1, 0.2)
        .measure_z(0)
        .measure_z(1)
    )
    result = _compile(circuit)
    assert set(_instructions(result["compiled_source"])) <= _ALLOWED_QCIS
    assert "decompose/rotations" in (result["metadata"]["core_steps"] or [])

    # lowering 的 _expand 不再认识旋转门（证明分解只发生在核心，无重复副本）。
    from tyxonq.compiler.platforms.guodun.lowering import _expand

    for rot in (
        ("rx", 0, 0.3),
        ("ry", 0, 0.3),
        ("rxx", 0, 1, 0.2),
        ("ryy", 0, 1, 0.2),
        ("rzz", 0, 1, 0.2),
    ):
        with pytest.raises(ValueError, match="未知门"):
            _expand(rot)
