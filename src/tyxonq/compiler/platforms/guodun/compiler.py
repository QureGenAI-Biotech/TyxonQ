"""把 TyxonQ 门线路离线编译为国盾平台使用的 QCIS。"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Sequence, TYPE_CHECKING

from .lowering import lower_circuit_to_gate_qcis

if TYPE_CHECKING:
    from ...api import CompileResult
    from ....core.ir import Circuit
    from ....devices import DeviceRule
    from ...gate_compiler import DecomposeSpec, RoutingSpec


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
    """TyxonQ Circuit -> 门级 QCIS（先委托 gate_compiler 核心分解，再 IR 直译降门）。

    C 阶段起本编译器 **委托 gate_compiler 公用核心**（对齐 lqcloud）：先由核心完成
    设备无关的分解（matchgate + rotations）与可选路由，得到编译后的 IR，再 1:1 直译
    为门级 QCIS。如此 matchgate / 旋转分解只在核心实现一次（消除 lowering 与
    stages/decompose/rotations.py 的重复恒等式），lowering 退化为纯方言翻译。

    与 lqcloud 的本质差异：physical_qubits 在 guodun 是**编译期**参数，烘焙进 QCIS
    文本（Q60/Q55）；lqcloud 则是运行期 initial_layout，编译器不消费。
    """

    name = "guodun"

    def compile(
        self,
        circuit: "Circuit",
        *,
        decompose: "bool | DecomposeSpec" = True,
        route: "bool | RoutingSpec" = False,
        device_rule: "DeviceRule | None" = None,
        options: Dict[str, Any] | None = None,
    ) -> "CompileResult":
        opts = dict(options or {})
        output = str(opts.get("output", "qcis")).lower()
        if output != "qcis":
            raise ValueError("Guodun 编译器当前只支持 output='qcis'")

        # physical_qubits 是 guodun 的编译期参数：校验后从 options 移除，稍后烘焙进 QCIS
        # 文本（区别于 lqcloud 的运行期 initial_layout）。缺省用恒等映射 0..n-1。
        physical_qubits = _validate_physical_qubits(
            opts.pop("physical_qubits", None), circuit.num_qubits
        )
        CqlibCircuit = _load_cqlib_circuit()

        # 委托 gate_compiler 公用核心：先做设备无关分解（matchgate + rotations）+ 可选路由，
        # 得到编译后的 IR（output="ir" 时 compiled_source 就是 Circuit），再直译降门为 QCIS。
        # guodun 的 QCIS 原生基 {X,H,RZ,CZ,M} 无法表达 matchgate 与参数化旋转，且瘦身后
        # lowering 不再自带这些分解，故此处**强制** matchgate/rotations 均开（decompose 参数
        # 仅为与 lqcloud 的 API 对称而保留，其取值被 guodun 的固定目标基覆盖）。
        # basis_gates 含 "x" 不含 "rx"：否则 GatesTransform 会把 x 重写成 rx(pi)，
        # 破坏 QCIS 的 X 指令（如精确文本断言 "X Q60"）。
        from ...gate_compiler import DecomposeSpec, GateCompiler

        core_opts = dict(opts)
        core_opts["output"] = "ir"
        core_opts["basis_gates"] = ["x", "h", "rz", "cz", "cx"]
        core_result = GateCompiler().compile(
            circuit,
            decompose=DecomposeSpec(matchgate=True, rotations=True),
            route=route,
            device_rule=device_rule,
            options=core_opts,
        )
        compiled_ir = core_result["compiled_source"]

        # 直译：编译后的 IR -> 门级 QCIS（不经 QASM 文本、不依赖 Qiskit/cqlib 分解）。
        # lowering 只做纯方言翻译：1q fixed(y/z/s/t..)→rz/x、cx→H·CZ·H、swap/iswap/cy/cry，
        # 以及测量归一 + 物理比特烘焙；旋转与 matchgate 已由核心分解，不再在此重复。
        qcis, gate_stats, _basis_ops = lower_circuit_to_gate_qcis(
            compiled_ir, physical_qubits
        )

        # 双保险：确认只含允许提交的门级指令（gd_sim1 已验证 X/H/RZ/CZ/M）。
        _validate_qcis_instructions(qcis)
        # 用 cqlib 自己的解析器做一次离线语法验收。
        CqlibCircuit.load(qcis)

        core_meta = core_result.get("metadata", {}) or {}
        mapping = {index: qubit for index, qubit in enumerate(physical_qubits)}
        metadata = {
            "output": "qcis",
            "basis_gates": ["x", "h", "rz", "cz"],
            "logical_physical_mapping": mapping,
            "physical_qubits": list(physical_qubits),
            "gate_stats": gate_stats,
            "conversion": "core_compile_then_ir_to_gate_qcis",
            # 并入核心的编译痕迹（步骤序列 / 设备规则 / 分解规格），便于排查与委托测试断言。
            "core_steps": core_meta.get("steps"),
            "core": core_meta,
        }
        return {
            "circuit": circuit,
            "compiled_source": qcis,
            "metadata": metadata,
        }


__all__ = ["GuodunCompiler"]
