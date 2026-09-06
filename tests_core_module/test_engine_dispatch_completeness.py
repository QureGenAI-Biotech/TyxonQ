"""模拟器引擎 op 分发单一真相源的护栏测试。

背景（backlog 7 的系统性根因）：三个模拟器引擎（statevector / density_matrix /
matrix_product_state）各自维护 run() 与 state() 两套 op 分发循环，且对未知 op
一律静默 ``continue``。结果 ``cry`` 曾在 state() 缺失被静默丢弃，而 ``y`` / ``z``
/ ``t`` / ``tdg`` / ``cy`` 至今在**全部三引擎的所有分发循环里都没有分支**——任何
使用这些门的电路都会被静默算错，且引擎层零覆盖。

本文件用**完全独立的 numpy 稠密参考**（显式基态循环，不复用引擎的 einsum/kron，
门矩阵用内联字面量）作为唯一真相源，锁死四类隐患：
  1. 分发完备性 / 无静默丢弃：每个幺正门的末态必须等于独立参考（漏门 => 不等）。
  2. run()/state() 逐门 parity：同一引擎两条路径必须给出同一概率分布。
  3. 跨引擎 parity：三引擎在公共门集上必须一致。
  4. 未知 op 必须 loudly raise，绝不静默跳过。

判据全部使用 ≥3 比特非回文体系（最小对称体系会让位序/漏门自抵消）。
"""

from __future__ import annotations

import numpy as np
import pytest

from tyxonq.core.ir.circuit import Circuit
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
from tyxonq.devices.simulators.density_matrix.engine import DensityMatrixEngine
from tyxonq.devices.simulators.matrix_product_state.engine import (
    MatrixProductStateEngine,
)

N = 3  # ≥3 比特非回文，避免位序/漏门自抵消


# ---------------------------------------------------------------------------
# 独立稠密参考实现（qubit 0 = MSB = 左端，与引擎约定一致）
# ---------------------------------------------------------------------------
_SQ2 = 1.0 / np.sqrt(2.0)

_REF_1Q = {
    "h": lambda: np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=complex),
    "x": lambda: np.array([[0, 1], [1, 0]], dtype=complex),
    "y": lambda: np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": lambda: np.array([[1, 0], [0, -1]], dtype=complex),
    "s": lambda: np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": lambda: np.array([[1, 0], [0, -1j]], dtype=complex),
    "t": lambda: np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
    "tdg": lambda: np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
    "rx": lambda th: np.array(
        [[np.cos(th / 2), -1j * np.sin(th / 2)],
         [-1j * np.sin(th / 2), np.cos(th / 2)]], dtype=complex),
    "ry": lambda th: np.array(
        [[np.cos(th / 2), -np.sin(th / 2)],
         [np.sin(th / 2), np.cos(th / 2)]], dtype=complex),
    "rz": lambda th: np.array(
        [[np.exp(-1j * th / 2), 0], [0, np.exp(1j * th / 2)]], dtype=complex),
}

_I2 = np.eye(2, dtype=complex)
_X = _REF_1Q["x"]()
_Y = _REF_1Q["y"]()
_Z = _REF_1Q["z"]()

_REF_2Q = {
    "cx": lambda: np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
    "cy": lambda: np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex),
    "cz": lambda: np.diag([1, 1, 1, -1]).astype(complex),
    "cry": lambda th: np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0],
         [0, 0, np.cos(th / 2), -np.sin(th / 2)],
         [0, 0, np.sin(th / 2), np.cos(th / 2)]], dtype=complex),
    "swap": lambda: np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex),
    "iswap": lambda: np.array(
        [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex),
    "rxx": lambda th: np.cos(th / 2) * np.eye(4, dtype=complex)
        - 1j * np.sin(th / 2) * np.kron(_X, _X),
    "ryy": lambda th: np.cos(th / 2) * np.eye(4, dtype=complex)
        - 1j * np.sin(th / 2) * np.kron(_Y, _Y),
    "rzz": lambda th: np.cos(th / 2) * np.eye(4, dtype=complex)
        - 1j * np.sin(th / 2) * np.kron(_Z, _Z),
}

# 各门的参数个数（op 元组里 theta 的位置）
_PARAM_1Q = {"rx", "ry", "rz"}
_PARAM_2Q = {"cry", "rxx", "ryy", "rzz"}


def _apply_1q_ref(psi: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    """独立 1q 应用：显式基态循环，qubit q 对应 bit (n-1-q)（q0=MSB）。"""
    dim = 1 << n
    out = np.zeros(dim, dtype=complex)
    bit = 1 << (n - 1 - q)
    for j in range(dim):
        a = psi[j]
        if a == 0:
            continue
        b0 = U[0, 0] * a if (j & bit) == 0 else U[0, 1] * a
        b1 = U[1, 0] * a if (j & bit) == 0 else U[1, 1] * a
        if (j & bit) == 0:
            out[j] += b0
            out[j | bit] += b1
        else:
            out[j & ~bit] += b0
            out[j] += b1
    return out


def _apply_2q_ref(psi: np.ndarray, U4: np.ndarray, qa: int, qb: int, n: int) -> np.ndarray:
    """独立 2q 应用：显式基态循环。U4 基序 |qa qb>（qa 为高位），与引擎一致。"""
    dim = 1 << n
    U = np.asarray(U4, dtype=complex).reshape(4, 4)
    out = np.zeros(dim, dtype=complex)
    bita = 1 << (n - 1 - qa)
    bitb = 1 << (n - 1 - qb)
    for j in range(dim):
        a = psi[j]
        if a == 0:
            continue
        ba = 1 if (j & bita) else 0
        bb = 1 if (j & bitb) else 0
        col = ba * 2 + bb
        base = j & ~bita & ~bitb
        for oa in range(2):
            for ob in range(2):
                amp = U[oa * 2 + ob, col]
                if amp == 0:
                    continue
                k = base | (oa * bita) | (ob * bitb)
                out[k] += amp * a
    return out


def ref_statevector(ops, n: int = N) -> np.ndarray:
    """按 ops 列表逐个应用，返回独立参考末态（|0..0> 起）。"""
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0
    for op in ops:
        name = op[0]
        if name in _REF_1Q:
            th = float(op[2]) if name in _PARAM_1Q else None
            U = _REF_1Q[name](th) if th is not None else _REF_1Q[name]()
            psi = _apply_1q_ref(psi, U, int(op[1]), n)
        elif name in _REF_2Q:
            th = float(op[3]) if name in _PARAM_2Q else None
            U4 = _REF_2Q[name](th) if th is not None else _REF_2Q[name]()
            psi = _apply_2q_ref(psi, U4, int(op[1]), int(op[2]), n)
        elif name in ("measure_z", "barrier"):
            continue
        else:
            raise AssertionError(f"reference has no matrix for op '{name}'")
    return psi


# 通用非平凡前奏：让每个门都作用在一个通用（非基矢）态上，确保可观测
def _prep_ops(target_qubits) -> list:
    ops = []
    for q in target_qubits:
        ops.append(("rx", q, 1.1))
        ops.append(("ry", q, 0.7))
    return ops


def _probs(psi: np.ndarray) -> np.ndarray:
    p = np.abs(np.asarray(psi).reshape(-1)) ** 2
    s = p.sum()
    return p / s if s > 0 else p


def _normalize(p: np.ndarray) -> np.ndarray:
    """将已是概率分布（|amp|^2）的数组归一化，不再平方。

    run(shots=0)["probabilities"] 与 driver 的 probabilities 已经是 |amp|^2 分布，
    若误用 _probs 会再平方一次导致假阳性。
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    s = p.sum()
    return p / s if s > 0 else p


# ---------------------------------------------------------------------------
# 用例 1：分发完备性 —— 每个幺正门的 statevector.state() 必须等于独立参考
#         （y/z/t/tdg/cy 若被静默丢弃，这里立即显形）
# ---------------------------------------------------------------------------
_ONE_Q_CASES = [
    ("h", (0,)), ("x", (1,)), ("y", (2,)), ("z", (0,)),
    ("s", (1,)), ("sdg", (2,)), ("t", (0,)), ("tdg", (1,)),
    ("rx", (2,)), ("ry", (0,)), ("rz", (1,)),
]
_TWO_Q_CASES = [
    ("cx", (0, 1)), ("cy", (1, 2)), ("cz", (0, 2)),
    ("cry", (2, 0)), ("swap", (0, 1)), ("iswap", (1, 2)),
    ("rxx", (0, 2)), ("ryy", (1, 0)), ("rzz", (2, 1)),
]


def _build_gate_op(name, qubits):
    if name in _PARAM_1Q:
        return (name, qubits[0], 0.83)
    if name in _PARAM_2Q:
        return (name, qubits[0], qubits[1], 0.83)
    return (name, *qubits)


@pytest.mark.parametrize("name,qubits", _ONE_Q_CASES + _TWO_Q_CASES)
def test_statevector_state_matches_independent_reference(name, qubits):
    ops = _prep_ops(qubits) + [_build_gate_op(name, qubits)]
    ref = ref_statevector(ops, N)
    c = Circuit(N, ops=list(ops))
    got = np.asarray(StatevectorEngine().state(c)).reshape(-1)
    assert np.max(np.abs(got - ref)) < 1e-10, (
        f"gate '{name}' 末态偏离独立参考 —— 可能被静默丢弃或位序错误"
    )


# ---------------------------------------------------------------------------
# 用例 2：run()/state() parity —— 同一引擎两条路径必须一致（三引擎各一份）
# ---------------------------------------------------------------------------
_MIXED_OPS = [
    ("rx", 0, 0.9), ("h", 1), ("ry", 2, 1.3),
    ("cx", 0, 1), ("y", 2), ("cz", 1, 2),
    ("t", 0), ("cy", 1, 2), ("rz", 2, 0.4),
    ("rxx", 0, 2, 0.6), ("s", 1), ("iswap", 0, 1),
    ("tdg", 2), ("z", 0), ("swap", 1, 2),
]


def _engine_state_probs(eng, ops):
    """从各引擎 state() 的返回（态矢 / 密度矩阵 / MPS）统一提取概率分布。"""
    c = Circuit(N, ops=list(ops))
    st = eng.state(c)
    nm = getattr(eng, "name", "")
    if nm == "density_matrix":
        # 密度矩阵：对角即概率（已是分布，归一而不平方）
        return _normalize(np.real(np.diag(np.asarray(st))))
    if nm == "matrix_product_state":
        from tyxonq.libs.quantum_library.kernels.matrix_product_state import (
            to_statevector,
        )
        return _probs(np.asarray(to_statevector(st)).reshape(-1))
    # statevector：|amp|^2
    return _probs(np.asarray(st).reshape(-1))


def _engine_run_probs(eng, ops):
    c = Circuit(N, ops=list(ops) + [("measure_z", q) for q in range(N)])
    out = eng.run(c, shots=0)
    # 优先用 run() 直接给出的 probabilities（S5 单一源后）；否则回退到 state()
    prob = out.get("probabilities")
    if prob is None:
        exp = out.get("expectations", {})
        # 无法从 <Z_q> 唯一重建分布，跳过（该引擎尚未支持 shots=0 概率单一源）
        pytest.skip("run() 未在 shots=0 返回 probabilities（S5 前的已知状态）")
    return _normalize(prob)


@pytest.mark.parametrize(
    "eng_factory",
    [StatevectorEngine, DensityMatrixEngine, MatrixProductStateEngine],
    ids=["statevector", "density_matrix", "mps"],
)
def test_run_state_parity(eng_factory):
    eng = eng_factory()
    p_state = _engine_state_probs(eng, _MIXED_OPS)
    p_run = _engine_run_probs(eng, _MIXED_OPS)
    assert np.max(np.abs(p_state - p_run)) < 1e-9, (
        "run() 与 state() 分发不一致 —— 两套 op 循环已分叉"
    )


# ---------------------------------------------------------------------------
# 用例 3：跨引擎 parity —— 三引擎在公共门集上给出同一概率分布
# ---------------------------------------------------------------------------
def _mps_probs(eng, ops):
    c = Circuit(N, ops=list(ops))
    mps = eng.state(c)
    from tyxonq.libs.quantum_library.kernels.matrix_product_state import (
        to_statevector,
    )
    return _probs(np.asarray(to_statevector(mps)).reshape(-1))


def _dm_probs(eng, ops):
    c = Circuit(N, ops=list(ops))
    # density_matrix: 取 rho 对角（已是概率分布，归一而不平方）
    rho = eng.state(c) if hasattr(eng, "state") else None
    if rho is None:
        pytest.skip("density_matrix 尚无 state()")
    rho = np.asarray(rho)
    return _normalize(np.real(np.diag(rho)))


def test_cross_engine_parity():
    sv = _probs(np.asarray(StatevectorEngine().state(Circuit(N, ops=list(_MIXED_OPS)))).reshape(-1))
    dm = _dm_probs(DensityMatrixEngine(), _MIXED_OPS)
    mps = _mps_probs(MatrixProductStateEngine(), _MIXED_OPS)
    assert np.max(np.abs(sv - dm)) < 1e-9, "density_matrix 与 statevector 分发不一致"
    assert np.max(np.abs(sv - mps)) < 1e-9, "mps 与 statevector 分发不一致"


# ---------------------------------------------------------------------------
# 用例 4：未知 op 必须 loudly raise（三引擎），绝不静默跳过
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "eng_factory",
    [StatevectorEngine, DensityMatrixEngine, MatrixProductStateEngine],
    ids=["statevector", "density_matrix", "mps"],
)
def test_unknown_op_raises(eng_factory):
    eng = eng_factory()
    c = Circuit(N, ops=[("h", 0), ("__totally_unknown_gate__", 1)])
    with pytest.raises((ValueError, NotImplementedError)):
        eng.state(c)


# ---------------------------------------------------------------------------
# 用例 5：driver shots=0 单一源 —— device_base.run(shots=0) 的概率来自 run()，
#         且 driver 不再二次独立调用 eng.state()（S5 后生效）
# ---------------------------------------------------------------------------
def test_driver_shots0_single_source(monkeypatch):
    from tyxonq.devices.simulators import driver as sim_driver

    call_count = {"state": 0}
    orig_state = StatevectorEngine.state

    def _counting_state(self, circuit):
        call_count["state"] += 1
        return orig_state(self, circuit)

    monkeypatch.setattr(StatevectorEngine, "state", _counting_state)

    from tyxonq.devices import base as device_base

    c = Circuit(N, ops=list(_MIXED_OPS) + [("measure_z", q) for q in range(N)])
    tasks = device_base.run(provider="simulator", device="statevector",
                            circuit=[c], shots=0)
    res = tasks[0].get_result(wait=False)
    prob = res.get("probabilities")
    assert prob is not None, "driver shots=0 未返回 probabilities"
    ref = _probs(ref_statevector(_MIXED_OPS, N))
    assert np.max(np.abs(_normalize(prob) - ref)) < 1e-9
    assert call_count["state"] == 0, (
        "driver 仍在 shots=0 二次调用 eng.state() —— 未单一源化"
    )
