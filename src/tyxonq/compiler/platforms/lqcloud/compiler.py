"""把 TyxonQ 门线路离线编译为官方 LQCloud QuantumCircuit 对象。

与 guodun 的一等 compile_engine 对称：源方言是 TyxonQ IR，目标方言是 lqcloud SDK 的
QuantumCircuit **对象**（纯对象方言，无 qasm/文本中间层，故 compiled_source 是对象而非字符串）。

R1 起本编译器 **委托 gate_compiler 公用核心**：先由核心完成设备无关的分解
（matchgate/rotations）与路由，得到编译后的 IR，再 1:1 直译为 lqcloud SDK 对象。
如此 matchgate 分解只在核心实现一次（消除交接手册 §4.7“两处重复 matchgate”坑），
lowering 退化为纯翻译（不再自带 2-qubit unitary 分解）。

注意 physical_qubits 的语义差异：
  - guodun 把 physical_qubits 烘焙进 QCIS 文本（Q21/Q28），是编译期参数；
  - lqcloud 的 physical_qubits 是运行期 backend 参数（映射到 SDK 的 initial_layout），
    不烘焙进对象，因此本编译器**不消费** physical_qubits——它继续随 Circuit.run() 流向驱动。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, TYPE_CHECKING

from .lowering import to_lqcloud

if TYPE_CHECKING:
    from ...api import CompileResult
    from ....core.ir import Circuit
    from ....devices import DeviceRule
    from ...gate_compiler import DecomposeSpec, RoutingSpec


# compiled_source 是 SDK 对象；output 只用于语义标注，接受若干等价别名。
_SUPPORTED_OUTPUTS = ("lqcloud", "lqcloud_circuit", "object")


def _gate_stats(qc: Any) -> Dict[str, int]:
    """统计官方线路对象中的指令名。

    SDK 对象（以及测试替身）都暴露 ``.data``，每项形如 ``(instruction, qubits, clbits)``，
    ``instruction.name`` 即门名。任何异常都退化为空统计，绝不因统计失败阻塞编译。
    """
    data = getattr(qc, "data", None)
    if not data:
        return {}
    try:
        return dict(Counter(str(item[0].name) for item in data))
    except Exception:
        return {}


class LQCloudCompiler:
    """TyxonQ Circuit -> lqcloud.QuantumCircuit 对象（先委托 gate_compiler 核心，再 IR 直译）。"""

    name = "lqcloud"

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
        output = str(opts.get("output", "lqcloud")).lower()
        if output not in _SUPPORTED_OUTPUTS:
            raise ValueError(
                "LQCloud 编译器当前只支持 output='lqcloud'（产物为 SDK 对象）"
            )

        # physical_qubits 是运行期 initial_layout，不在编译期消费；误传入编译选项则忽略。
        opts.pop("physical_qubits", None)

        # 委托 gate_compiler 公用核心：先做设备无关的分解（matchgate/rotations）+ 路由，
        # 得到编译后的 IR（output="ir" 时 compiled_source 就是 Circuit），再直译为 SDK 对象。
        # 如此 matchgate 分解只在核心发生一次，lowering 不再重复实现（见模块 docstring）。
        # 注：decompose 先于 route——to_qiskit 不认 unitary op，必须先分解再路由。
        from ...gate_compiler import GateCompiler

        core_opts = dict(opts)
        core_opts["output"] = "ir"
        core_result = GateCompiler().compile(
            circuit,
            decompose=decompose,
            route=route,
            device_rule=device_rule,
            options=core_opts,
        )
        compiled_ir = core_result["compiled_source"]

        # 直译：编译后的 IR -> lqcloud.QuantumCircuit 对象（1:1 调用 SDK 门方法，无网络请求）。
        qc = to_lqcloud(compiled_ir)

        core_meta = core_result.get("metadata", {}) or {}
        metadata = {
            "output": "lqcloud",
            "target": "lqcloud.QuantumCircuit",
            "num_qubits": int(compiled_ir.num_qubits),
            "gate_stats": _gate_stats(qc),
            "conversion": "core_compile_then_ir_to_lqcloud_object",
            # 并入核心的编译痕迹（步骤序列 / 设备规则 / 路由信息），便于排查与委托测试断言。
            "core_steps": core_meta.get("steps"),
            "core": core_meta,
        }
        return {
            "circuit": circuit,
            "compiled_source": qc,
            "metadata": metadata,
        }


__all__ = ["LQCloudCompiler"]
