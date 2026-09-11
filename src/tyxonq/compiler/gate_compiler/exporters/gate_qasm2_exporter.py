"""Gate-level OpenQASM 2.0 exporter for TyxonQ Circuit IR.

将门级电路（不含脉冲算子）直译为 OpenQASM 2.0（qelib1.inc）文本。当用户指定
``output="qasm2"``/``output="qasm"`` 时使用。

与 ``gate_qasm3_exporter.py`` 对称：纯 IR → 字符串、**零外部依赖**（不再绕道 qiskit，
避免「IR→qiskit→transpile→qasm2」这段多余翻译带来的准确度/性能损失）。

注意：qelib1.inc 未定义 ``iswap/rxx/ryy/rzz`` 等门，遇到时抛 ``ValueError``，要求先分解到
qelib1 兼容基（``compiler`` 的 qasm2 输出路径会默认降级到 ``["cx","h","rz","rx","cz"]``）。
"""

from typing import Any, Dict, FrozenSet, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from tyxonq.core.ir import Circuit


class GateQASM2Exporter:
    """Export gate-level TyxonQ Circuit to OpenQASM 2.0 (qelib1.inc) format."""

    # qelib1.inc 支持的门集合（按参数个数分类）。
    _SINGLE_NOPARAM: FrozenSet[str] = frozenset(
        {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "id"}
    )
    _SINGLE_ONEPARAM: FrozenSet[str] = frozenset({"rx", "ry", "rz", "p", "u1"})
    _SINGLE_TWOPARAM: FrozenSet[str] = frozenset({"u2"})
    _SINGLE_THREEPARAM: FrozenSet[str] = frozenset({"u3", "u"})
    _TWO_NOPARAM: FrozenSet[str] = frozenset({"cx", "cy", "cz", "ch", "swap"})
    _TWO_ONEPARAM: FrozenSet[str] = frozenset({"crx", "cry", "crz", "cu1", "cp"})
    _TWO_THREEPARAM: FrozenSet[str] = frozenset({"cu3"})
    _THREE_NOPARAM: FrozenSet[str] = frozenset({"ccx", "cswap"})
    _RESET: FrozenSet[str] = frozenset({"reset"})

    # 名称归一：IR 里的别名 → qelib1.inc 标准门名。
    _ALIASES: Dict[str, str] = {
        "cnot": "cx",
        "toffoli": "ccx",
        "fredkin": "cswap",
        "i": "id",
        "phase": "p",
        "cphase": "cp",
    }

    # 明确不支持（qelib1.inc 未定义），需先分解。
    _UNSUPPORTED: FrozenSet[str] = frozenset({"iswap", "rxx", "ryy", "rzz"})

    # 脉冲算子：门级电路不应出现，出现则跳过（与 qasm3 exporter 一致）。
    _PULSE_OPS: FrozenSet[str] = frozenset(
        {"pulse", "pulse_inline", "play", "set_phase", "shift_phase", "set_frequency"}
    )

    def __init__(self):
        """Initialize the exporter."""
        pass

    @staticmethod
    def _is_measure(op: Any) -> bool:
        return (
            isinstance(op, (list, tuple))
            and len(op) >= 2
            and str(op[0]).lower() in ("measure", "measure_z")
        )

    def export(self, circuit: "Circuit") -> str:
        """Export circuit to OpenQASM 2.0 format.

        Args:
            circuit: TyxonQ Circuit IR

        Returns:
            OpenQASM 2.0 code string

        Raises:
            ValueError: If circuit contains operations unsupported by qelib1.inc
        """
        ops: List[Tuple] = list(circuit.ops)
        n = int(circuit.num_qubits)

        lines: List[str] = []
        # 1. Header
        lines.append("OPENQASM 2.0;")
        lines.append('include "qelib1.inc";')
        # 2. Registers：经典寄存器大小 = measure 算子数，按出现顺序映射 c[0..M-1]
        #    （与旧 to_qiskit(add_measures=True) 的顺序约定保持一致，下游按此还原计数）。
        lines.append(f"qreg q[{n}];")
        n_measures = sum(1 for op in ops if self._is_measure(op))
        if n_measures > 0:
            lines.append(f"creg c[{n_measures}];")

        # 3. Gate operations
        measure_index = 0
        for op in ops:
            if self._is_measure(op):
                q = int(op[1])
                lines.append(f"measure q[{q}]->c[{measure_index}];")
                measure_index += 1
                continue
            gate_line = self._export_gate(op)
            if gate_line:
                lines.append(gate_line)

        # 4. Final newline
        lines.append("")
        return "\n".join(lines)

    def _export_gate(self, op: Tuple) -> str:
        """Export a single non-measure operation to a QASM2 line.

        Args:
            op: Operation tuple from Circuit.ops

        Returns:
            QASM2 code line, or empty string if skipped

        Raises:
            ValueError: If operation is unsupported by qelib1.inc
        """
        if not isinstance(op, (list, tuple)) or len(op) == 0:
            return ""

        name = str(op[0]).lower()

        # 跳过脉冲算子（门级电路不应出现）
        if name in self._PULSE_OPS:
            return ""

        # 名称归一
        name = self._ALIASES.get(name, name)

        # 明确的非 qelib1 门：要求先分解
        if name in self._UNSUPPORTED:
            raise ValueError(
                f"OpenQASM 2.0 (qelib1.inc) 不支持门 '{name}'；请先分解到 qelib1 兼容基"
                f"（例如 output='qasm2' 路径默认降级到 ['cx','h','rz','rx','cz']）。"
            )

        # 单比特无参
        if name in self._SINGLE_NOPARAM:
            if len(op) < 2:
                raise ValueError(f"Gate {name} requires at least one qubit")
            return f"{name} q[{int(op[1])}];"

        # 单比特一参
        if name in self._SINGLE_ONEPARAM:
            if len(op) < 3:
                raise ValueError(f"Gate {name} requires one qubit and one parameter")
            return f"{name}({float(op[2])}) q[{int(op[1])}];"

        # 单比特二参（u2(phi, lambda)）
        if name in self._SINGLE_TWOPARAM:
            if len(op) < 4:
                raise ValueError(f"Gate {name} requires one qubit and two parameters")
            return f"{name}({float(op[2])}, {float(op[3])}) q[{int(op[1])}];"

        # 单比特三参（u3/u(theta, phi, lambda)）
        if name in self._SINGLE_THREEPARAM:
            if len(op) < 5:
                raise ValueError(f"Gate {name} requires one qubit and three parameters")
            return f"{name}({float(op[2])}, {float(op[3])}, {float(op[4])}) q[{int(op[1])}];"

        # 双比特无参
        if name in self._TWO_NOPARAM:
            if len(op) < 3:
                raise ValueError(f"Gate {name} requires two qubits")
            return f"{name} q[{int(op[1])}],q[{int(op[2])}];"

        # 双比特一参（crx/cry/crz/cu1/cp）：op = (name, control, target, theta)
        if name in self._TWO_ONEPARAM:
            if len(op) < 4:
                raise ValueError(f"Gate {name} requires two qubits and one parameter")
            return f"{name}({float(op[3])}) q[{int(op[1])}],q[{int(op[2])}];"

        # 双比特三参（cu3）：op = (name, control, target, theta, phi, lambda)
        if name in self._TWO_THREEPARAM:
            if len(op) < 6:
                raise ValueError(f"Gate {name} requires two qubits and three parameters")
            return (
                f"{name}({float(op[3])}, {float(op[4])}, {float(op[5])}) "
                f"q[{int(op[1])}],q[{int(op[2])}];"
            )

        # 三比特无参（ccx/cswap）
        if name in self._THREE_NOPARAM:
            if len(op) < 4:
                raise ValueError(f"Gate {name} requires three qubits")
            return f"{name} q[{int(op[1])}],q[{int(op[2])}],q[{int(op[3])}];"

        # reset
        if name in self._RESET:
            if len(op) < 2:
                raise ValueError(f"Gate {name} requires one qubit")
            return f"reset q[{int(op[1])}];"

        # barrier
        if name == "barrier":
            qs = [int(q) for q in op[1:]] if len(op) > 1 else []
            if qs:
                return "barrier " + ",".join(f"q[{q}]" for q in qs) + ";"
            return "barrier;"

        # 其余未知算子
        raise ValueError(f"Unsupported operation for QASM2 export: {name}")
