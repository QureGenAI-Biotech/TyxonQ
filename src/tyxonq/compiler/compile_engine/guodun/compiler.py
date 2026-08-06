"""把 TyxonQ 门线路离线编译为国盾平台使用的 QCIS。"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Sequence, TYPE_CHECKING

from ..qiskit import QiskitCompiler

if TYPE_CHECKING:
    from ...api import CompileResult
    from ....core.ir import Circuit


_CQLIB_VERSION = "1.3.11"
_ALLOWED_QCIS_INSTRUCTIONS = {
    "X",
    "H",
    "RZ",
    "CZ",
    "M",
}


def _load_cqlib_circuit():
    """延迟加载可选依赖，避免未使用国盾时强制安装 cqlib。"""
    try:
        installed_version = version("cqlib")
        from cqlib import Circuit as CqlibCircuit
    except (ImportError, PackageNotFoundError) as exc:
        raise RuntimeError(
            "Guodun 编译需要可选依赖 cqlib==1.3.11；"
            "请执行 `pip install 'tyxonq[guodun]'`。"
        ) from exc

    if installed_version != _CQLIB_VERSION:
        raise RuntimeError(
            "Guodun 编译当前仅兼容 cqlib==1.3.11，"
            f"检测到版本 {installed_version!r}。"
        )
    return CqlibCircuit


def _validate_physical_qubits(
    physical_qubits: Any, num_qubits: int
) -> list[int]:
    """校验“逻辑比特 i -> 列表第 i 个物理比特”的显式映射。"""
    if physical_qubits is None:
        return list(range(num_qubits))
    if isinstance(physical_qubits, (str, bytes)) or not isinstance(
        physical_qubits, Sequence
    ):
        raise TypeError("physical_qubits 必须是整数列表")

    mapped = list(physical_qubits)
    if len(mapped) != num_qubits:
        raise ValueError(
            "physical_qubits 长度必须等于线路逻辑比特数："
            f"期望 {num_qubits}，实际 {len(mapped)}"
        )
    if any(isinstance(q, bool) or not isinstance(q, int) for q in mapped):
        raise TypeError("physical_qubits 中的每一项都必须是整数")
    if any(q < 0 for q in mapped):
        raise ValueError("physical_qubits 不能包含负数")
    if len(set(mapped)) != len(mapped):
        raise ValueError("physical_qubits 不能包含重复物理比特")
    return mapped


def _load_qasm2_circuit(qasm2: str):
    """重新解析 OpenQASM 2，确保 QCIS 确实来自约定的中间层。"""
    try:
        from qiskit.qasm2 import loads

        return loads(qasm2)
    except Exception as exc:  # pragma: no cover - 由 Qiskit 版本决定
        raise RuntimeError(f"无法解析 Qiskit 生成的 OpenQASM 2：{exc}") from exc


def _qiskit_to_gate_qcis(qiskit_circuit: Any, physical_qubits: Sequence[int]) -> str:
    """把已降门的 Qiskit 线路转成 gd_sim1 验证过的门级 QCIS。"""
    qcis_lines: list[str] = []
    for instruction in qiskit_circuit.data:
        operation = instruction.operation
        name = operation.name
        logical_qubits = [
            int(qiskit_circuit.find_bit(qubit).index)
            for qubit in instruction.qubits
        ]
        mapped_qubits = [physical_qubits[index] for index in logical_qubits]

        if name == "barrier":
            continue
        if name == "x":
            qcis_lines.append(f"X Q{mapped_qubits[0]}")
        elif name == "h":
            qcis_lines.append(f"H Q{mapped_qubits[0]}")
        elif name == "rz":
            angle = float(operation.params[0])
            qcis_lines.append(f"RZ Q{mapped_qubits[0]} {angle:.9g}")
        elif name == "cz":
            qcis_lines.append(
                f"CZ Q{mapped_qubits[0]} Q{mapped_qubits[1]}"
            )
        elif name == "measure":
            qcis_lines.append(f"M Q{mapped_qubits[0]}")
        else:
            raise ValueError(f"Qiskit 降门后仍包含未支持指令 {name!r}")
    return "\n".join(qcis_lines)


def _validate_qcis_instructions(qcis: str) -> Dict[str, int]:
    """确认 cqlib 输出只包含本期允许提交的门线路指令。"""
    gate_stats: Dict[str, int] = {}
    for line_number, line in enumerate(qcis.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        instruction = stripped.split()[0]
        if instruction not in _ALLOWED_QCIS_INSTRUCTIONS:
            raise ValueError(
                f"QCIS 第 {line_number} 行包含未允许指令 {instruction!r}"
            )
        gate_stats[instruction] = gate_stats.get(instruction, 0) + 1
    return gate_stats


class GuodunCompiler:
    """TyxonQ Circuit -> Qiskit 基础门 -> QASM2 -> 门级 QCIS。"""

    name = "guodun"

    def compile(
        self,
        circuit: "Circuit",
        options: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "CompileResult":
        opts = dict(options or {})
        output = str(opts.get("output", "qcis")).lower()
        if output != "qcis":
            raise ValueError("Guodun 编译器当前只支持 output='qcis'")

        # 国盾不会替用户猜测测量；线路必须显式 add_measure/measure_z。
        has_measure_op = any(op and op[0] == "measure_z" for op in circuit.ops)
        has_measure_instruction = any(
            instruction and instruction[0] == "measure"
            for instruction in circuit.instructions
        )
        if not (has_measure_op or has_measure_instruction):
            raise ValueError(
                "Guodun 线路必须显式调用 add_measure(...) 或 measure_z(...)"
            )

        # 兼容 Circuit.add_measure() 的 instruction 表示，统一交给 Qiskit。
        qiskit_circuit = circuit
        if has_measure_instruction:
            from copy import copy

            qiskit_circuit = copy(circuit)
            qiskit_circuit.ops = list(circuit.ops)
            measured_in_ops = {
                int(op[1]) for op in circuit.ops if op and op[0] == "measure_z"
            }
            for name, qubits in circuit.instructions:
                if name != "measure":
                    continue
                for qubit in qubits:
                    if int(qubit) not in measured_in_ops:
                        qiskit_circuit.ops.append(("measure_z", int(qubit)))
                        measured_in_ops.add(int(qubit))

        physical_qubits = _validate_physical_qubits(
            opts.pop("physical_qubits", None), circuit.num_qubits
        )
        CqlibCircuit = _load_cqlib_circuit()

        # gd_sim1 已验证的门集合是 X/H/RZ/CZ/M；不生成 X2P/Y2M。
        qiskit_options = dict(opts)
        qiskit_options.update(
            {
                "output": "qasm2",
                "transpile": True,
                # 已在上面确认存在显式测量，因此这里不会触发自动 measure_all。
                "add_measures": True,
                "basis_gates": ["x", "h", "rz", "cz"],
                "optimization_level": int(opts.get("optimization_level", 0)),
            }
        )
        qasm_result = QiskitCompiler().compile(qiskit_circuit, options=qiskit_options)
        qasm2 = qasm_result["compiled_source"]
        lowered_qiskit_circuit = _load_qasm2_circuit(qasm2)
        qcis = _qiskit_to_gate_qcis(lowered_qiskit_circuit, physical_qubits)
        gate_stats = _validate_qcis_instructions(qcis)

        # 用 cqlib 自己的解析器做一次离线语法验收。
        CqlibCircuit.load(qcis)
        mapping = {index: qubit for index, qubit in enumerate(physical_qubits)}
        metadata = {
            "output": "qcis",
            "basis_gates": ["x", "h", "rz", "cz"],
            "logical_physical_mapping": mapping,
            "physical_qubits": list(physical_qubits),
            "gate_stats": gate_stats,
            "conversion": "qasm2_to_gate_qcis",
            "qiskit": qasm_result.get("metadata", {}),
        }
        return {
            "circuit": circuit,
            "compiled_source": qcis,
            "metadata": metadata,
        }


__all__ = ["GuodunCompiler"]
