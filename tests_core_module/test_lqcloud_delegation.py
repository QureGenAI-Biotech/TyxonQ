"""LQCloud 委托 gate_compiler 公用核心的单测（计划 §114）。

R1 起 ``LQCloudCompiler.compile`` 先跑核心（分解 matchgate/rotations + 路由）得到编译 IR，
再 ``to_lqcloud`` 直译为 SDK 对象。本文件验证委托的关键契约：
  - matchgate **只在核心分解一次**（lowering 不再有私有副本，消除交接手册 §4.7 坑）；
  - K4 少一条边的 4 比特线路能**离线路由**成功（插 swap、产物全相邻）；
  - HEA 2 比特相邻路径 route=**no-op**（既有可跑路径零回归）；
  - 委托后 ``to_lqcloud`` 直译的门集合不回归（§115）。

全部离线：用 _FakeQuantumCircuit 替身 monkeypatch ``lqcloud``，不依赖真 SDK / 网络。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import tyxonq as tq
from tyxonq.compiler.api import compile as compile_ir


class _FakeGate:
    def __init__(self, name):
        self.name = name


class _FakeQubit:
    def __init__(self, index):
        self.index = index


class _FakeQuantumCircuit:
    """最小 lqcloud.QuantumCircuit 替身：记录 (门名, 比特下标) 以便断言相邻性/测量序。"""

    def __init__(self, num_qubits, num_clbits):
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.qubits = [_FakeQubit(i) for i in range(num_qubits)]
        self.data = []

    def _append(self, name, *qubits):
        self.data.append((_FakeGate(name), [self.qubits[q] for q in qubits], []))

    def h(self, q): self._append("h", q)
    def x(self, q): self._append("x", q)
    def y(self, q): self._append("y", q)
    def z(self, q): self._append("z", q)
    def s(self, q): self._append("s", q)
    def sdg(self, q): self._append("sdg", q)
    def t(self, q): self._append("t", q)
    def tdg(self, q): self._append("tdg", q)
    def reset(self, q): self._append("reset", q)
    def rx(self, theta, q): self._append("rx", q)
    def ry(self, theta, q): self._append("ry", q)
    def rz(self, theta, q): self._append("rz", q)
    def cx(self, a, b): self._append("cx", a, b)
    def cy(self, a, b): self._append("cy", a, b)
    def cz(self, a, b): self._append("cz", a, b)
    def swap(self, a, b): self._append("swap", a, b)
    def iswap(self, a, b): self._append("iswap", a, b)
    def crx(self, t, a, b): self._append("crx", a, b)
    def cry(self, t, a, b): self._append("cry", a, b)
    def crz(self, t, a, b): self._append("crz", a, b)
    def su2(self, mat, q): self._append("su2", q)
    def barrier(self, *q): self._append("barrier", *q)
    def measure(self, qubit, clbit): self._append("measure", qubit)


@pytest.fixture
def fake_lqcloud(monkeypatch):
    module = SimpleNamespace(QuantumCircuit=_FakeQuantumCircuit)
    monkeypatch.setitem(sys.modules, "lqcloud", module)
    return module


_TWO_QUBIT_NAMES = {"cx", "cy", "cz", "swap", "iswap", "crx", "cry", "crz"}


def _names(qc):
    return [item[0].name for item in qc.data]


def _two_qubit_pairs(qc):
    return [
        (item[1][0].index, item[1][1].index)
        for item in qc.data
        if item[0].name in _TWO_QUBIT_NAMES
    ]


def _measured_qubits(qc):
    return [item[1][0].index for item in qc.data if item[0].name == "measure"]


def _two_mode_fock(v):
    """复现 LUCJ/SQD 的 two_mode_fock_matrix(v)：2x2 轨道旋转 → 4x4 Fock 酉。"""
    v = np.asarray(v, dtype=complex)
    m = np.eye(4, dtype=complex)
    m[1, 1] = v[1, 1]
    m[1, 2] = v[1, 0]
    m[2, 1] = v[0, 1]
    m[2, 2] = v[0, 0]
    m[3, 3] = v[0, 0] * v[1, 1] - v[0, 1] * v[1, 0]
    return m


# --------------------------------------------------------------------------- #
# matchgate 只在核心分解一次
# --------------------------------------------------------------------------- #
def test_matchgate_decomposed_once_by_core(fake_lqcloud):
    theta, phi = 0.4, 0.7
    v = np.array(
        [[np.cos(theta), -np.sin(theta) * np.exp(1j * phi)],
         [np.sin(theta) * np.exp(-1j * phi), np.cos(theta)]],
        dtype=complex,
    )
    circuit = tq.Circuit(2).unitary(0, 1, matrix=_two_mode_fock(v)).measure_z(0).measure_z(1)
    result = compile_ir(circuit, platform="lqcloud", output="lqcloud")
    qc = result["compiled_source"]
    names = _names(qc)

    # 2-qubit matchgate 已被核心分解为原生门：partial-iSWAP = RXX·RYY 共 4 个 cx。
    assert names.count("cx") == 4
    assert "h" in names and "rx" in names and "rz" in names
    # 未被当成单比特 su2、也未残留 unitary。
    assert "su2" not in names
    assert names[-2:] == ["measure", "measure"]
    # 核心确实装配并执行了 matchgate 分解步骤。
    assert "decompose/matchgate" in (result["metadata"]["core_steps"] or [])


def test_lowering_has_no_duplicate_decomposition():
    """交接手册 §4.7“两处重复 matchgate”坑已消除：lowering 不再有私有分解函数。"""
    from tyxonq.compiler.platforms.lqcloud import lowering

    assert not hasattr(lowering, "_decompose_two_qubit_unitary")


def test_one_qubit_unitary_still_maps_to_su2_through_delegation(fake_lqcloud):
    """委托后单比特自定义酉仍由 lowering 直译为 su2（核心只分解 2-qubit matchgate）。"""
    mat = np.array([[0, 1], [1, 0]], dtype=complex)  # 单比特 X
    circuit = tq.Circuit(1).unitary(0, matrix=mat).measure_z(0)
    result = compile_ir(circuit, platform="lqcloud", output="lqcloud")
    names = _names(result["compiled_source"])
    assert "su2" in names


# --------------------------------------------------------------------------- #
# 离线路由：K4 少一条边
# --------------------------------------------------------------------------- #
# 4 比特全连接但缺 (0,3) 这条边；cx(0,3) 非相邻，必须靠 SWAP 路由。
_K4_MINUS_EDGE = [
    [0, 1], [1, 0], [0, 2], [2, 0],
    [1, 2], [2, 1], [1, 3], [3, 1],
    [2, 3], [3, 2],
]


def test_k4_minus_one_edge_routes_offline(fake_lqcloud):
    device_rule = {"coupling_map": [list(p) for p in _K4_MINUS_EDGE], "num_qubits": 4}
    circuit = tq.Circuit(4).h(0).cx(0, 3)
    circuit = circuit.measure_z(0).measure_z(1).measure_z(2).measure_z(3)

    result = compile_ir(
        circuit, platform="lqcloud", output="lqcloud",
        route=True, device_rule=device_rule,
    )
    qc = result["compiled_source"]
    names = _names(qc)

    # 非相邻的 cx(0,3) 触发路由：插入 swap。
    assert "swap" in names
    # 产物所有两比特门都落在 K4-少一边 的耦合图上（本地下标空间）。
    edges = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in _K4_MINUS_EDGE}
    for a, b in _two_qubit_pairs(qc):
        assert (min(a, b), max(a, b)) in edges, f"非相邻两比特门 {(a, b)}"
    # 4 个逻辑比特各测一次（clbit 数 = 4，测量比特互不重复）。
    assert qc.num_clbits == 4
    measured = _measured_qubits(qc)
    assert len(measured) == 4 and len(set(measured)) == 4
    # 路由把物理布局回填进调用方的 device_rule（供 Circuit.run() 作运行期 initial_layout）。
    assert "initial_layout" in device_rule


def test_delegation_without_route_keeps_original_indices(fake_lqcloud):
    """route=False（默认）时不路由：cx(0,3) 原样保留，比特下标不变（不插 swap）。"""
    circuit = tq.Circuit(4).h(0).cx(0, 3)
    circuit = circuit.measure_z(0).measure_z(1).measure_z(2).measure_z(3)
    result = compile_ir(circuit, platform="lqcloud", output="lqcloud")
    qc = result["compiled_source"]
    assert "swap" not in _names(qc)
    assert (0, 3) in _two_qubit_pairs(qc)


# --------------------------------------------------------------------------- #
# 回归：HEA 2 比特相邻路径 route=no-op；门集合不回归
# --------------------------------------------------------------------------- #
def test_hea_two_qubit_adjacent_route_is_noop(fake_lqcloud):
    device_rule = {"coupling_map": [[0, 1], [1, 0]], "num_qubits": 2}
    circuit = tq.Circuit(2).ry(0, 0.3).ry(1, 0.5).cx(0, 1).measure_z(0).measure_z(1)
    result = compile_ir(
        circuit, platform="lqcloud", output="lqcloud",
        route=True, device_rule=device_rule,
    )
    qc = result["compiled_source"]
    names = _names(qc)
    # 已相邻 → 不插 swap；ry/cx 直译保留。
    assert "swap" not in names
    assert names.count("ry") == 2 and "cx" in names
    # 测量逻辑序保持：clbit i 测逻辑比特 i。
    assert _measured_qubits(qc) == [0, 1]


def test_delegation_preserves_native_gate_set(fake_lqcloud):
    """§115 回归：委托核心后 to_lqcloud 直译的门集合不回归（常用门仍 1:1 直译）。"""
    circuit = tq.Circuit(3)
    circuit.h(0).x(1).s(2).t(0).rx(1, 0.1).rz(2, 0.2)
    circuit.cx(0, 1).cz(1, 2).swap(0, 2).iswap(1, 2)
    circuit = circuit.add_measure(0, 1, 2)

    result = compile_ir(circuit, platform="lqcloud", output="lqcloud")
    names = _names(result["compiled_source"])
    # x 被核心 GatesTransform 重写为等价的 rx(pi)，其余受支持门直译保留。
    for g in ("h", "s", "t", "rx", "rz", "cx", "cz", "swap", "iswap", "measure"):
        assert g in names, f"门集合回归：缺 {g}"
    assert names.count("measure") == 3
