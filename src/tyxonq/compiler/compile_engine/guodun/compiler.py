"""把 TyxonQ 门线路离线编译为国盾平台使用的 QCIS。"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Sequence, TYPE_CHECKING

from .lowering import lower_circuit_to_gate_qcis

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
    """TyxonQ Circuit -> 门级 QCIS（IR 直译降门，不经 QASM 文本、不依赖 Qiskit）。"""

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

        physical_qubits = _validate_physical_qubits(
            opts.pop("physical_qubits", None), circuit.num_qubits
        )
        CqlibCircuit = _load_cqlib_circuit()

        # 直译：TyxonQ IR -> 门级 QCIS（不经 QASM 文本、不依赖 Qiskit/cqlib 分解）。
        # lowering 内部完成 measure 归一、显式测量校验、递归降门到 {X,H,RZ,CZ,M}；
        # 每条分解恒等式已由离线数值 unitary 等价校验（允许全局相位差）。
        qcis, gate_stats, _basis_ops = lower_circuit_to_gate_qcis(
            circuit, physical_qubits
        )

        # 双保险：确认只含允许提交的门级指令（gd_sim1 已验证 X/H/RZ/CZ/M）。
        _validate_qcis_instructions(qcis)
        # 用 cqlib 自己的解析器做一次离线语法验收。
        CqlibCircuit.load(qcis)

        mapping = {index: qubit for index, qubit in enumerate(physical_qubits)}
        metadata = {
            "output": "qcis",
            "basis_gates": ["x", "h", "rz", "cz"],
            "logical_physical_mapping": mapping,
            "physical_qubits": list(physical_qubits),
            "gate_stats": gate_stats,
            "conversion": "ir_to_gate_qcis_direct",
        }
        return {
            "circuit": circuit,
            "compiled_source": qcis,
            "metadata": metadata,
        }


__all__ = ["GuodunCompiler"]
