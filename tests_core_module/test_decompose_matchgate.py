"""核心 matchgate 分解的单测（gate_compiler/stages/decompose/matchgate.py）。

R1 起 2-qubit 数守恒 matchgate 的分解是**核心唯一实现**（lqcloud 等方言经委托复用，
lowering 不再自带副本），故本文件把原先散落在 test_lqcloud_lowering.py 的数值精确性
覆盖迁到核心函数上，并补齐 ``MatchgateDecompose`` 步骤级与 ``_unitary_cache`` 保留测试。

全部离线：用 TyxonQ statevector 引擎做数值校验，不依赖任何硬件 / SDK / 网络。
"""

from __future__ import annotations

import numpy as np
import pytest

import tyxonq as tq
from tyxonq.compiler.gate_compiler.stages.decompose.matchgate import (
    MatchgateDecompose,
    decompose_two_qubit_matchgate,
    is_number_conserving_matchgate,
)
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine


def _two_mode_fock(v):
    """复现 LUCJ/SQD 的 two_mode_fock_matrix(v)：把 2x2 轨道旋转提升为 4x4 Fock 酉。"""
    v = np.asarray(v, dtype=complex)
    m = np.eye(4, dtype=complex)
    m[1, 1] = v[1, 1]
    m[1, 2] = v[1, 0]
    m[2, 1] = v[0, 1]
    m[2, 2] = v[0, 0]
    m[3, 3] = v[0, 0] * v[1, 1] - v[0, 1] * v[1, 0]
    return m


# --------------------------------------------------------------------------- #
# is_number_conserving_matchgate：结构判定
# --------------------------------------------------------------------------- #
def test_two_mode_fock_is_number_conserving():
    v = np.array([[0.6, -0.8j], [0.8j, 0.6]], dtype=complex)
    assert is_number_conserving_matchgate(_two_mode_fock(v))


def test_identity_4x4_is_number_conserving():
    assert is_number_conserving_matchgate(np.eye(4, dtype=complex))


def test_cross_subspace_coupling_is_not_matchgate():
    # |00> <-> |11> 耦合（跨粒子数子空间泄露）→ 非数守恒 matchgate。
    mat = np.eye(4, dtype=complex)
    mat[0, 0] = mat[3, 3] = 0.0
    mat[0, 3] = mat[3, 0] = 1.0
    assert not is_number_conserving_matchgate(mat)


def test_non_4x4_is_not_matchgate():
    assert not is_number_conserving_matchgate(np.eye(2, dtype=complex))


# --------------------------------------------------------------------------- #
# decompose_two_qubit_matchgate：纯函数的入参校验
# --------------------------------------------------------------------------- #
def test_pure_fn_rejects_non_4x4():
    with pytest.raises(ValueError, match="4x4"):
        decompose_two_qubit_matchgate(np.eye(2, dtype=complex), 0, 1)


def test_pure_fn_rejects_non_matchgate():
    mat = np.eye(4, dtype=complex)
    mat[0, 0] = mat[3, 3] = 0.0
    mat[0, 3] = mat[3, 0] = 1.0
    with pytest.raises(NotImplementedError, match="matchgate"):
        decompose_two_qubit_matchgate(mat, 0, 1)


def test_pure_fn_emits_only_native_vocabulary():
    theta, phi = 0.4, 0.7
    v = np.array(
        [[np.cos(theta), -np.sin(theta) * np.exp(1j * phi)],
         [np.sin(theta) * np.exp(-1j * phi), np.cos(theta)]],
        dtype=complex,
    )
    ops = decompose_two_qubit_matchgate(_two_mode_fock(v), 0, 1)
    names = {str(op[0]).lower() for op in ops}
    # 原生门词汇：只有 rz/rx/h/cx（partial-iSWAP 用 cx，相位用 rz）。
    assert names <= {"rz", "rx", "h", "cx"}
    # partial-iSWAP = RXX·RYY 共 4 个 cx。
    assert sum(1 for op in ops if op[0] == "cx") == 4


def test_pure_fn_identity_prunes_to_empty():
    # v=I → b=0、无相位：整块为单位元，零角度块被剪枝，op 序列为空。
    ops = decompose_two_qubit_matchgate(_two_mode_fock(np.eye(2, dtype=complex)), 0, 1)
    assert ops == []


# --------------------------------------------------------------------------- #
# 数值精确性：分解门序必须精确重建 two_mode_fock(v)（忽略全局相位）
# --------------------------------------------------------------------------- #
def test_decomposition_is_numerically_exact():
    """正确性关键：用 statevector 引擎对 4 个计算基态分别作用「分解门序」与「原 unitary」，
    拼成 4x4 酉后比保真度 |Tr(U_dec† U_tgt)|/4 ≈ 1（随机 U(2) + Givens 形式 + 退化情形）。
    """
    eng = StatevectorEngine()

    def _apply_ops(c, ops):
        for op in ops:
            name = op[0]
            if name in ("h", "x"):
                getattr(c, name)(op[1])
            elif name in ("rx", "ry", "rz"):
                getattr(c, name)(op[1], op[2])
            elif name == "cx":
                c.cx(op[1], op[2])
            else:
                raise AssertionError(f"unexpected op {name}")

    def unitary_of(build):
        U = np.zeros((4, 4), dtype=complex)
        for idx in range(4):
            c = tq.Circuit(2)
            if (idx >> 1) & 1:
                c.x(0)
            if idx & 1:
                c.x(1)
            build(c)
            U[:, idx] = np.asarray(eng.state(c)).reshape(-1)
        return U

    rng = np.random.default_rng(0)
    cases = []
    for _ in range(15):  # 随机 U(2)
        A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        cases.append(np.linalg.qr(A)[0])
    for _ in range(15):  # Givens 形式（LUCJ 实际发射）
        t, p = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        cases.append(np.array([[np.cos(t), np.sin(t) * np.exp(1j * p)],
                               [-np.sin(t) * np.exp(-1j * p), np.cos(t)]], dtype=complex))
    # 退化情形：v=I（b=0）、b=π、纯对角相位
    cases += [np.eye(2, dtype=complex),
              np.array([[0, 1], [-1, 0]], dtype=complex),
              np.array([[np.exp(0.3j), 0], [0, np.exp(-0.7j)]], dtype=complex)]

    for v in cases:
        target = _two_mode_fock(v)
        ops = decompose_two_qubit_matchgate(target, 0, 1)
        U_dec = unitary_of(lambda c, _o=ops: _apply_ops(c, _o))
        U_tgt = unitary_of(lambda c, _m=target: c.unitary(0, 1, matrix=_m))
        fid = abs(np.trace(U_dec.conj().T @ U_tgt)) / 4.0
        assert fid > 1 - 1e-9, f"decomposition mismatch for v={v}, fidelity={fid}"


# --------------------------------------------------------------------------- #
# MatchgateDecompose 步骤级：线路重写 + 缓存保留
# --------------------------------------------------------------------------- #
def _matchgate_circuit():
    theta = 0.35
    v = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]], dtype=complex)
    c = tq.Circuit(2).unitary(0, 1, matrix=_two_mode_fock(v)).measure_z(0).measure_z(1)
    return c


def test_step_decomposes_two_qubit_unitary_in_circuit():
    c = _matchgate_circuit()
    out = MatchgateDecompose().apply(c)
    names = [str(op[0]).lower() for op in out.ops if isinstance(op, (list, tuple)) and op]
    # 2-qubit unitary 已展开为原生门，线路里不再有 unitary op。
    assert "unitary" not in names
    assert names.count("cx") == 4
    # 测量保持不变。
    assert names[-2:] == ["measure_z", "measure_z"]


def test_step_preserves_unitary_cache_for_residual_one_qubit_unitary():
    # 关键：MatchgateDecompose 必须把 _unitary_cache 复制到新线路（非 dataclass 字段），
    # 否则线路里残留的 1-qubit unitary 会在后续 lowering 丢失矩阵引用。
    mat1 = np.array([[0, 1], [1, 0]], dtype=complex)  # 单比特 X
    c = _matchgate_circuit().unitary(0, matrix=mat1)
    assert getattr(c, "_unitary_cache", None)  # 前置：确有缓存
    out = MatchgateDecompose().apply(c)
    assert getattr(out, "_unitary_cache", None) == c._unitary_cache
    names = [str(op[0]).lower() for op in out.ops if isinstance(op, (list, tuple)) and op]
    assert "unitary" in names  # 1-qubit unitary 原样保留


def test_step_keeps_non_matchgate_two_qubit_unitary_by_default():
    mat = np.eye(4, dtype=complex)
    mat[0, 0] = mat[3, 3] = 0.0
    mat[0, 3] = mat[3, 0] = 1.0
    c = tq.Circuit(2).unitary(0, 1, matrix=mat).measure_z(0).measure_z(1)
    out = MatchgateDecompose().apply(c)  # 默认非 strict：原样保留
    names = [str(op[0]).lower() for op in out.ops if isinstance(op, (list, tuple)) and op]
    assert "unitary" in names


def test_step_strict_mode_raises_on_non_matchgate():
    mat = np.eye(4, dtype=complex)
    mat[0, 0] = mat[3, 3] = 0.0
    mat[0, 3] = mat[3, 0] = 1.0
    c = tq.Circuit(2).unitary(0, 1, matrix=mat).measure_z(0).measure_z(1)
    with pytest.raises(NotImplementedError, match="matchgate"):
        MatchgateDecompose().apply(c, matchgate_strict=True)


def test_step_raises_when_cache_missing():
    # 人为构造带 unitary op 但无缓存的线路 → 清晰报错（缺矩阵缓存）。
    c = tq.Circuit(2, ops=[("unitary", 0, 1, "nope"), ("measure_z", 0), ("measure_z", 1)])
    with pytest.raises(ValueError, match="缺少矩阵缓存"):
        MatchgateDecompose().apply(c)
