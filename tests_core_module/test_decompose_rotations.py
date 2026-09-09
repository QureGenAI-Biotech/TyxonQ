"""rotations 分解步骤（RotationsDecompose）的数值与结构回归测试。

守卫两类不变量：
  1. 结构——rx/ry/rxx/ryy/rzz 分解后只落在核心原生集 {h, rz, cx}，rz 原样透传。
  2. 数值等价——每个旋转门分解后的序列 unitary 与原门 unitary 相等（允许全局
     相位差，因全局相位对测量计数无影响）。这是“核心是旋转分解的唯一且被数值
     校验的实现”的证据；guodun/lqcloud lowering 删除各自的旋转恒等式副本后，
     旋转语义完全由本步承接，故此处数值覆盖是删除动作的安全前置。

移植自 test_guodun_lowering.py 的 up-to-global-phase 数值等价工具，但把降门入口
从 guodun 的 _lower_op 换成核心 RotationsDecompose.apply，使核心成为被测主体。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tyxonq.core.ir import Circuit
from tyxonq.compiler.gate_compiler.stages.decompose.rotations import RotationsDecompose
from tyxonq.libs.quantum_library.kernels.gates import resolve_unitary

_NATIVE = {"h", "rz", "cx"}
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
    """op 序列 -> 总 unitary（时间顺序：后作用的左乘）。"""
    U = np.eye(2 ** nq, dtype=complex)
    for op in ops:
        U = _op_matrix(op, nq) @ U
    return U


def _decompose(op, nq):
    """把单个 op 交给核心 RotationsDecompose 分解，返回分解后的 op 列表。"""
    circ = Circuit(num_qubits=nq, ops=[op])
    return RotationsDecompose().apply(circ, device_rule={}).ops


def _assert_equiv_up_to_global_phase(op, nq, tol=1e-8):
    """断言 op 经核心分解后与原门幺正等价（允许全局相位差），且结果全落在原生集内。"""
    orig = _op_matrix(op, nq)
    lowered = _decompose(op, nq)
    for b in lowered:
        assert str(b[0]).lower() in _NATIVE, f"{op} 分解后仍含非原生门 {b}"
    dec = _seq_matrix(lowered, nq)
    M = dec @ orig.conj().T  # 等价则应为 lam * I
    lam = M[0, 0]
    assert abs(abs(lam) - 1) < 1e-6, f"{op}: |lam|={abs(lam)} 非幺正"
    assert np.allclose(M, lam * np.eye(M.shape[0]), atol=tol), (
        f"{op} 分解后与原门不等价（up to global phase）"
    )


def test_decompose_rotations_lowers_to_native_gates():
    circ = Circuit(
        num_qubits=2,
        ops=[
            ("rx", 0, 0.5),
            ("ry", 1, 0.3),
            ("rxx", 0, 1, 0.2),
            ("ryy", 0, 1, 0.7),
            ("rzz", 0, 1, 1.1),
        ],
    )
    out = RotationsDecompose().apply(circ, device_rule={})
    # all ops lowered to native set {h, rz, cx}
    native = {"h", "rz", "cx"}
    assert len(out.ops) > len(circ.ops)
    assert all(op[0] in native for op in out.ops)


# --------------------------------------------------------------------------- #
# 数值等价：旋转门逐门（up-to-global-phase）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["rx", "ry"])
@pytest.mark.parametrize("theta", _THETAS)
def test_single_qubit_rotations_numerically_exact(name, theta):
    _assert_equiv_up_to_global_phase((name, 0, theta), 1)


@pytest.mark.parametrize("name", ["rxx", "ryy", "rzz"])
@pytest.mark.parametrize("theta", _THETAS)
def test_two_qubit_rotations_numerically_exact(name, theta):
    _assert_equiv_up_to_global_phase((name, 0, 1, theta), 2)


def test_rz_is_native_passthrough():
    """rz 已是核心原生门，RotationsDecompose 应原样透传（不分解为 h/rz 组合）。"""
    ops = _decompose(("rz", 0, 0.7), 1)
    assert len(ops) == 1
    assert str(ops[0][0]).lower() == "rz"
    assert abs(float(ops[0][2]) - 0.7) < 1e-12
