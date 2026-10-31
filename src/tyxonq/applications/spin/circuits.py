"""构造自旋块、基态拟设和 Trotter 演化的 TyxonQ 门线路。"""

from dataclasses import dataclass

import numpy as np

from ...core.ir import Circuit
from .models import SpinChainModel


def cartan_ops(q0, q1, a, b, c):
    """三 CNOT 实现 exp[i(a XX+b YY+c ZZ)]，允许与目标相差全局相位。"""
    return [
        ("sdg", q1), ("cx", q1, q0),
        ("rz", q0, np.pi / 2 - 2 * c),
        ("ry", q1, 2 * a - np.pi / 2), ("cx", q0, q1),
        ("ry", q1, np.pi / 2 - 2 * b), ("cx", q1, q0), ("s", q0),
    ]


def initial_blocks(n, initial_state="singlet"):
    """singlet 对乘积和 Néel 态都从 |0...0> 通过基础门制备。"""
    blocks = []
    if initial_state == "singlet":
        for q in range(0, n, 2):
            blocks.extend([("x", q), ("x", q + 1), ("h", q + 1), ("cx", q + 1, q)])
    elif initial_state == "neel":
        blocks.extend(("x", q) for q in range(0, n, 2))
    else:
        raise ValueError("initial_state must be 'singlet' or 'neel'")
    return blocks


def hva_blocks(model, params, initial_state="singlet"):
    """每层两个共享参数；局域生成元为 (Jxy XX+Jxy YY+Jz ZZ)/2。

    HVA 角度是变分参数，与实时演化的物理时间独立。
    """
    params = np.asarray(params, dtype=float)
    if params.ndim != 1 or len(params) % 2 or not np.isfinite(params).all():
        raise ValueError("HVA parameters must be a finite even-length vector")
    blocks = initial_blocks(model.n, initial_state)
    for odd, even in params.reshape(-1, 2):
        for parity, theta in ((1, odd), (0, even)):
            for q in range(parity, model.n - 1, 2):
                blocks.append(("cartan", q, q + 1, -theta * model.jxy / 2,
                               -theta * model.jxy / 2, -theta * model.jz / 2))
    return blocks


def evolution_blocks(model, dt, steps, order=2):
    """保留块结构供 AQC 使用；零步返回空线路，整段二阶演化合并中间半步。"""
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if not np.isfinite(dt) or dt <= 0 or order not in (1, 2):
        raise ValueError("dt must be positive and order must be 1 or 2")
    if steps == 0:
        return []
    blocks = []

    def layer(parity, duration):
        for q in range(parity, model.n - 1, 2):
            blocks.append(("cartan", q, q + 1, -duration * model.jxy / 4,
                           -duration * model.jxy / 4, -duration * model.jz / 4))

    if order == 1:
        for _ in range(steps):
            layer(0, dt)
            layer(1, dt)
    else:
        layer(0, dt / 2)
        for k in range(steps):
            layer(1, dt)
            layer(0, dt / 2 if k == steps - 1 else dt)
    return blocks


def blocks_to_circuit(n, blocks):
    """只在绑定完成后生成原生数值 Circuit；所有相互作用展开成基础门。"""
    ops = []
    for block in blocks:
        if block[0] == "cartan":
            ops.extend(cartan_ops(*block[1:]))
        else:
            ops.append(tuple(block))
    return Circuit(n, ops=fold_local_rotations(n, ops))


def fold_local_rotations(n, ops):
    """合并同一线上连续同轴旋转；只允许跨过作用于其他位点的门。

    S/SDG 写成 RZ 时仅引入全局相位。该规则同时用于基准和 AQC 输出，
    不改动 CNOT，也不改变优化时的参数模板。
    """
    out = []
    # 每条线记录仍然有效的操作位置，抵消旋转后可以找到前一个操作。
    histories = [[] for _ in range(n)]
    for op in ops:
        if op[0] in ("s", "sdg"):
            op = ("rz", op[1], np.pi / 2 if op[0] == "s" else -np.pi / 2)
        name = op[0]
        if name in ("rx", "ry", "rz"):
            q, angle = op[1:]
            if abs(angle) < 1e-14:
                continue
            if histories[q] and out[histories[q][-1]][0] == name:
                index = histories[q][-1]
                angle += out[index][2]
                out[index] = (name, q, angle)
                if abs(angle) < 1e-14:
                    out[index] = None
                    histories[q].pop()
                continue
        qubits = op[1:3] if name == "cx" else op[1:2]
        for q in qubits:
            histories[q].append(len(out))
        out.append(tuple(op))
    return [op for op in out if op is not None]


def build_evolution(model: SpinChainModel, dt: float, steps: int, order: int = 2) -> Circuit:
    """返回不含测量的自旋链演化线路。"""
    return blocks_to_circuit(model.n, evolution_blocks(model, dt, steps, order))


def circuit_stats(circuit):
    """在共同的三 CNOT 相互作用分解下计算门数和深度，不声称硬件最优。"""
    levels = [0] * circuit.num_qubits
    levels_2q = [0] * circuit.num_qubits
    count = 0
    for op in circuit.ops:
        qs = op[1:3] if op[0] == "cx" else op[1:2]
        level = max(levels[q] for q in qs) + 1
        for q in qs:
            levels[q] = level
        if len(qs) == 2:
            count += 1
            level = max(levels_2q[q] for q in qs) + 1
            for q in qs:
                levels_2q[q] = level
    return {"num_2q_gates": count, "depth_2q": max(levels_2q),
            "depth": max(levels), "num_gates": len(circuit.ops)}


@dataclass(frozen=True)
class ParameterAngle:
    """内部线性参数表达式；不改变公共 Circuit 对角度的处理。"""

    index: int
    scale: float = 1.0
    offset: float = 0.0

    def __mul__(self, value):
        return ParameterAngle(self.index, self.scale * value, self.offset * value)

    __rmul__ = __mul__

    def __sub__(self, value):
        return ParameterAngle(self.index, self.scale, self.offset - value)

    def __rsub__(self, value):
        return ParameterAngle(self.index, -self.scale, value - self.offset)


@dataclass
class SpinAnsatz:
    """带参数的基础门模板；参数绑定前不创建数值 Circuit。"""

    n: int
    ops: list
    initial_parameters: np.ndarray

    def bind(self, parameters) -> Circuit:
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != self.initial_parameters.shape or not np.isfinite(parameters).all():
            raise ValueError("parameters must match the ansatz and be finite")
        ops = [tuple(v.scale * parameters[v.index] + v.offset if isinstance(v, ParameterAngle) else v
                     for v in op) for op in self.ops]
        return Circuit(self.n, ops=fold_local_rotations(self.n, ops))


def make_spin_ansatz(n, seed_blocks):
    """从已知自旋块解析生成 ansatz，初始局域 ZXZ 为单位操作。

    每个 Cartan 块独立优化三轴角度和后置局域旋转；HVA 部分同样参与拟合。
    固定 Clifford 初态前缀被保留，不进行任意两比特矩阵分解。
    """
    params, ops = [], []

    def parameter(value):
        params.append(float(value))
        return ParameterAngle(len(params) - 1)

    def zxz(q):
        for gate in ("rz", "rx", "rz"):
            ops.append((gate, q, parameter(0.0)))

    for q in range(n):
        zxz(q)
    for block in seed_blocks:
        name = block[0]
        if name == "cartan":
            q0, q1 = block[1:3]
            a, b, c = [parameter(v) for v in block[3:]]
            ops.extend(cartan_ops(q0, q1, a, b, c))
            zxz(q0)
            zxz(q1)
        elif name in ("rx", "ry", "rz"):
            ops.append((name, block[1], parameter(block[2])))
        else:
            ops.append(block)
    return SpinAnsatz(n, ops, np.array(params))
