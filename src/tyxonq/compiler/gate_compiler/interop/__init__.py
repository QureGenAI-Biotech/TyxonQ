"""外部框架/格式互操作适配器（TyxonQ 原生表示 ↔ 第三方库对象/格式）。

本子包与 ``platforms/``（硬件方言直译通道）、``exporters/``（IR → 文本序列化）语义并列，
专门存放 TyxonQ 与**外部框架/格式**之间的转换适配器：

- ``qiskit_interop``：门级 Circuit IR ↔ qiskit ``QuantumCircuit`` 双向适配 + QASM2 解析；
- ``mpo_converters``：外部张量网络 MPO（tensornetwork/quimb 鸭子类型）→ 稠密矩阵，单向入站。
"""

from __future__ import annotations

from .mpo_converters import (
    contract_mpo_to_matrix,
    from_quimb_mpo,
    from_tensornetwork_mpo,
)
from .qiskit_interop import (
    DEFAULT_BASIS_GATES,
    DEFAULT_OPT_LEVEL,
    OP_MAPPING,
    comment_dict,
    comment_qasm,
    free_pi,
    from_qiskit,
    ir_to_qasm,
    ir_to_qiskit,
    normalize_transpile_options,
    qasm_to_ir,
    qiskit_to_ir,
    to_qiskit,
)

__all__ = [
    "DEFAULT_BASIS_GATES",
    "DEFAULT_OPT_LEVEL",
    "OP_MAPPING",
    "comment_dict",
    "comment_qasm",
    "contract_mpo_to_matrix",
    "free_pi",
    "from_qiskit",
    "from_quimb_mpo",
    "from_tensornetwork_mpo",
    "ir_to_qasm",
    "ir_to_qiskit",
    "normalize_transpile_options",
    "qasm_to_ir",
    "qiskit_to_ir",
    "to_qiskit",
]
