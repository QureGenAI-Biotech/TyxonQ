"""国盾 IR->QCIS 直译降门（lowering.py）的数值与结构回归测试。

纯离线：不依赖 Qiskit、不依赖 cqlib、不提交云端，因此在任何环境都应可跑。

守卫两类不变量：
  1. 数值等价——gates.py 权威词汇表 UNITARY_1Q / UNITARY_2Q 的每个门，降门到
     {x, h, rz, cz} 后的序列 unitary 与原门 unitary 相等（允许全局相位差，
     因为全局相位对测量计数无影响）。这是"直译无损耗"的核心证据。
  2. 结构契约——emit 出的门级 QCIS 只含 {X, H, RZ, CZ, M}；逻辑->物理映射正确；
     缺显式测量 / 特殊 op / 未知门 loudly raise；barrier 被丢弃；add_measure
     (instructions) 与 measure_z (ops) 两条测量写法产出一致。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tyxonq.compiler.compile_engine.guodun.lowering import (
    _lower_op,
    lower_circuit_to_gate_qcis,
)
from tyxonq.core.ir import Circuit
from tyxonq.libs.quantum_library.kernels.gates import resolve_unitary

_BASIS = {"x", "h", "rz", "cz", "measure_z"}
_ALLOWED_MNEMONIC = {"X", "H", "RZ", "CZ", "M"}
_THETAS = [math.pi / 3, -0.7, 1.234, math.pi]


# --------------------------------------------------------------------------- #
# 数值等价工具
# --------------------------------------------------------------------------- #
def _op_matrix(op, nq):
    """单个 op -> nq 比特空间 (2^nq x 2^nq) 矩阵。qubit 0 = MSB（与 gates.py 一致）。"""
    arity, qubits, mat = resolve_unitary(str(op[0]).lower(), tuple(op))
    assert arity in ("1q", "2q"), f"{op} 不是纯幺正门"
    mat = np.asarray(mat, dtype=complex)
    if arity == "1q":
        q = int(qubits[0])
        full = np.array([[1.0 + 0j]])
        for pos in range(nq):
            full = np.kron(full, mat if pos == q else np.eye(2, dtype=complex))
        return full
    return mat  # 2q：nq=2，基序 |q0 q1>


def _seq_matrix(ops, nq):
    """basis op 序列 -> 总 unitary（时间顺序：后作用的左乘）。"""
    U = np.eye(2 ** nq, dtype=complex)
    for op in ops:
        U = _op_matrix(op, nq) @ U
    return U


def _assert_equiv_up_to_global_phase(op, nq, tol=1e-8):
    """断言 op 降门后与原门幺正等价（允许全局相位差），且降门结果全落在 basis 内。"""
    orig = _op_matrix(op, nq)
    lowered = _lower_op(op)
    for b in lowered:
        assert str(b[0]).lower() in _BASIS, f"{op} 降门后仍含非 basis 门 {b}"
    dec = _seq_matrix(lowered, nq)
    M = dec @ orig.conj().T  # 等价则应为 lam * I
    lam = M[0, 0]
    assert abs(abs(lam) - 1) < 1e-6, f"{op}: |lam|={abs(lam)} 非幺正"
    assert np.allclose(M, lam * np.eye(M.shape[0]), atol=tol), (
        f"{op} 降门后与原门不等价（up to global phase）"
    )


# --------------------------------------------------------------------------- #
# 1. 数值等价：UNITARY_1Q / UNITARY_2Q 逐门
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["h", "x", "y", "z", "s", "sdg", "t", "tdg"])
def test_single_qubit_fixed_gates_equivalent(name):
    _assert_equiv_up_to_global_phase((name, 0), 1)


@pytest.mark.parametrize("name", ["rx", "ry", "rz"])
@pytest.mark.parametrize("theta", _THETAS)
def test_single_qubit_rotation_gates_equivalent(name, theta):
    _assert_equiv_up_to_global_phase((name, 0, theta), 1)


@pytest.mark.parametrize("name", ["cx", "cy", "cz", "iswap", "swap"])
def test_two_qubit_fixed_gates_equivalent(name):
    _assert_equiv_up_to_global_phase((name, 0, 1), 2)


@pytest.mark.parametrize("name", ["cry", "rxx", "ryy", "rzz"])
@pytest.mark.parametrize("theta", _THETAS)
def test_two_qubit_rotation_gates_equivalent(name, theta):
    _assert_equiv_up_to_global_phase((name, 0, 1, theta), 2)


# --------------------------------------------------------------------------- #
# 2. 结构契约
# --------------------------------------------------------------------------- #
def test_bell_emits_expected_gate_qcis_text():
    circuit = Circuit(2)
    circuit.h(0).cx(0, 1).measure_z(0).measure_z(1)
    qcis, stats, _basis_ops = lower_circuit_to_gate_qcis(circuit, [60, 55])
    assert qcis.splitlines() == [
        "H Q60",
        "H Q55",
        "CZ Q60 Q55",
        "H Q55",
        "M Q60",
        "M Q55",
    ]
    assert set(stats) <= _ALLOWED_MNEMONIC
    assert stats["M"] == 2
    assert "X2P" not in qcis  # 门级方言，不应出现脉冲原生门


def test_add_measure_and_measure_z_produce_identical_qcis():
    """instructions 里的 add_measure 与 ops 里的 measure_z 应产出完全一致的门级 QCIS。"""
    via_ops = Circuit(2)
    via_ops.h(0).cx(0, 1).measure_z(0).measure_z(1)
    via_instr = Circuit(2)
    via_instr.h(0).cx(0, 1).add_measure(0, 1)

    qcis_ops, _, _ = lower_circuit_to_gate_qcis(via_ops, [60, 55])
    qcis_instr, _, _ = lower_circuit_to_gate_qcis(via_instr, [60, 55])
    assert qcis_ops == qcis_instr


def test_two_qubit_rotations_only_use_allowed_mnemonics():
    circuit = Circuit(2)
    circuit.iswap(0, 1).rxx(0, 1, 0.2).ryy(0, 1, -0.3).rzz(0, 1, 0.4)
    circuit.measure_z(0).measure_z(1)
    _qcis, stats, basis_ops = lower_circuit_to_gate_qcis(circuit, [0, 1])
    assert set(stats) <= _ALLOWED_MNEMONIC
    assert stats["CZ"] > 0
    assert all(str(op[0]).lower() in _BASIS for op in basis_ops)


def test_physical_mapping_is_applied_per_logical_index():
    """逻辑比特 i -> physical_qubits[i]；q1 与 q10 不应混淆。"""
    mapping = list(range(11))
    mapping[1] = 60
    mapping[10] = 55
    circuit = Circuit(11).x(1).x(10).measure_z(1).measure_z(10)
    qcis, _, _ = lower_circuit_to_gate_qcis(circuit, mapping)
    assert qcis == "X Q60\nX Q55\nM Q60\nM Q55"


def test_barrier_op_is_dropped():
    assert _lower_op(("barrier", 0, 1)) == []


def test_missing_explicit_measurement_raises():
    with pytest.raises(ValueError, match="显式调用"):
        lower_circuit_to_gate_qcis(Circuit(1).h(0), [0])


@pytest.mark.parametrize(
    "special", ["unitary", "kraus", "project_z", "reset", "pulse", "pulse_inline"]
)
def test_special_ops_raise(special):
    with pytest.raises(ValueError):
        _lower_op((special, 0))


def test_unknown_gate_raises():
    with pytest.raises(ValueError, match="未知门"):
        _lower_op(("not_a_gate", 0))
