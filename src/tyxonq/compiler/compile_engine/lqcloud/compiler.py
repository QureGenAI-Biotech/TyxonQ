"""把 TyxonQ 门线路离线编译为官方 LQCloud QuantumCircuit 对象。

与 guodun 的一等 compile_engine 对称：源方言是 TyxonQ IR，目标方言是 lqcloud SDK 的
QuantumCircuit **对象**（纯对象方言，无 qasm/文本中间层，故 compiled_source 是对象而非字符串）。

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
    """TyxonQ Circuit -> lqcloud.QuantumCircuit 对象（IR 直译，无 qasm/文本中间层）。"""

    name = "lqcloud"

    def compile(
        self,
        circuit: "Circuit",
        options: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "CompileResult":
        opts = dict(options or {})
        output = str(opts.get("output", "lqcloud")).lower()
        if output not in _SUPPORTED_OUTPUTS:
            raise ValueError(
                "LQCloud 编译器当前只支持 output='lqcloud'（产物为 SDK 对象）"
            )

        # physical_qubits 是运行期 initial_layout，不在编译期消费；误传入编译选项则忽略。
        opts.pop("physical_qubits", None)

        # 直译：TyxonQ IR -> lqcloud.QuantumCircuit 对象（1:1 调用 SDK 门方法，无网络请求）。
        qc = to_lqcloud(circuit)

        metadata = {
            "output": "lqcloud",
            "target": "lqcloud.QuantumCircuit",
            "num_qubits": int(circuit.num_qubits),
            "gate_stats": _gate_stats(qc),
            "conversion": "ir_to_lqcloud_object_direct",
        }
        return {
            "circuit": circuit,
            "compiled_source": qc,
            "metadata": metadata,
        }


__all__ = ["LQCloudCompiler"]
