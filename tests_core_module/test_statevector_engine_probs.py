import numpy as np

from tyxonq.core.ir import Circuit
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine


def test_wavefunction_engine_state_prob_amp_sampling():
    eng = StatevectorEngine()
    # |+> on qubit 0, |0> on qubit 1
    c = Circuit(num_qubits=2, ops=[("h", 0)])

    s = eng.state(c)
    assert s.shape == (4,)
    p = eng.probability(c)
    # probabilities: 00 and 10 each 0.5 in big-endian indexing -> states 0 and 2
    assert np.isclose(np.sum(p), 1.0)
    assert np.isclose(p[0] + p[2], 1.0)

    a00 = eng.amplitude(c, "00")
    a10 = eng.amplitude(c, "10")
    a01 = eng.amplitude(c, "01")
    assert np.isclose(abs(a00) ** 2 + abs(a10) ** 2 + abs(a01) ** 2 + abs(eng.amplitude(c, "11")) ** 2, 1.0)
    assert np.isclose(abs(a01), 0.0)

    bits, prob = eng.perfect_sampling(c)
    assert bits in ("00", "10")
    assert 0.0 <= prob <= 1.0


def test_state_cry_not_silently_dropped():
    """回归：state() 分发曾缺 "cry" 分支，cry 被静默跳过（run() 正常），
    导致 UCC 门级单激发块在 shots=0 解析档退化为恒等算符。
    最小判据：|01> 上 cry(ctrl=1, tgt=0, pi/2) 应得 (|01>+|11>)/sqrt(2)。"""
    eng = StatevectorEngine()
    c = Circuit(num_qubits=2, ops=[("x", 1), ("cry", 1, 0, np.pi / 2)])
    s = np.asarray(eng.state(c)).reshape(-1)
    np.testing.assert_allclose(s, [0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)], atol=1e-12)


def test_state_dispatch_matches_dense_kron_simulation():
    """state() 分发循环 vs 独立的 numpy kron 稠密模拟逐门对照。

    背景：state() 与 run() 各自维护一套 op 分发，两者不同步时未知 op
    会被静默跳过（cry 曾因此丢失）。本测试用与引擎内部完全无关的
    kron 嵌入独立复算末态，拦截任何“分发缺分支”类回归。
    只用小端相邻比特对 (q, q+1)，kron 嵌入无需置换。
    """
    eng = StatevectorEngine()
    n = 3
    ops = [
        ("x", 2), ("h", 0),
        ("cry", 0, 1, 0.7), ("cx", 1, 2), ("ry", 2, 1.1),
        ("cz", 0, 1), ("cry", 1, 2, -0.9), ("rz", 0, 0.3), ("rx", 1, -0.4),
    ]

    # 独立矩阵定义（control 为高位，与库约定一致但实现无关）
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    def rot(pauli, theta):
        return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * pauli

    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    CZ = np.diag([1, 1, 1, -1]).astype(complex)

    def cry4(theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, -s], [0, 0, s, c]], dtype=complex)

    def embed(mat, q, k):
        """k 比特门嵌入 n 比特全空间（big-endian，wire0 为最高位）。"""
        left = np.eye(2 ** q, dtype=complex)
        right = np.eye(2 ** (n - q - k), dtype=complex)
        return np.kron(left, np.kron(mat, right))

    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    for op in ops:
        name = op[0]
        if name == "x":
            psi = embed(X, op[1], 1) @ psi
        elif name == "h":
            psi = embed(H, op[1], 1) @ psi
        elif name == "ry":
            psi = embed(rot(Y, op[2]), op[1], 1) @ psi
        elif name == "rz":
            psi = embed(rot(Z, op[2]), op[1], 1) @ psi
        elif name == "rx":
            psi = embed(rot(X, op[2]), op[1], 1) @ psi
        elif name == "cx":
            psi = embed(CX, op[1], 2) @ psi
        elif name == "cz":
            psi = embed(CZ, op[1], 2) @ psi
        elif name == "cry":
            psi = embed(cry4(op[3]), op[1], 2) @ psi
        else:
            raise AssertionError(f"test bug: unhandled op {name}")

    c = Circuit(num_qubits=n, ops=ops)
    s = np.asarray(eng.state(c)).reshape(-1)
    np.testing.assert_allclose(s, psi, atol=1e-12)


