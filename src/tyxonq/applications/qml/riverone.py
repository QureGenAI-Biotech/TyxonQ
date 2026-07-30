"""RiverONE ``TQVQC`` checkpoint 到 OpenQASM 2 的适配器。

本模块导入训练后的变分门参数，并生成包含振幅编码、数据重新上传
语义和 X/Y/Z 测量的线路。经典 ``WeightEncoder``、``BlockHyper`` 和
完整 ``VQCWeightGenerator`` 不属于本适配范围。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


_ROTATION_NAMES = ("rx", "ry", "rz")
_MEASUREMENT_BASES = ("X", "Y", "Z")
_QASM_BASIS_GATES = ("cx", "h", "rz", "rx", "cz")
_REUPLOAD_EVERY = 2
_STATE_KEY_PATTERN = re.compile(
    r"^vqcs\.(?P<vqc>\d+)\.variational\.l(?P<layer>\d+)_w(?P<wire>\d+)\."
    r"(?P<gate>rx|ry|rz)\.params$"
)

AngleTriple = tuple[float, float, float]
AngleLayer = tuple[AngleTriple, ...]


@dataclass(frozen=True)
class RiverONEVQCSpec:
    """一条训练后的 RiverONE ``TQVQC``。"""

    n_qubits: int
    n_layers: int
    angles: Sequence[Sequence[Sequence[float]]]
    reupload_every: int = _REUPLOAD_EVERY

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValueError("RiverONE VQC 至少需要 2 个量子比特。")
        if self.n_layers < 1:
            raise ValueError("RiverONE VQC 至少需要 1 层。")
        if self.reupload_every < 1:
            raise ValueError("reupload_every 必须大于 0。")
        if len(self.angles) != self.n_layers:
            raise ValueError(
                f"angles 层数应为 {self.n_layers}，实际为 {len(self.angles)}。"
            )

        # 把外部列表规范化为不可变元组，同时逐项检查角度形状和数值。
        normalized_layers: list[AngleLayer] = []
        for layer_index, layer in enumerate(self.angles):
            if len(layer) != self.n_qubits:
                raise ValueError(
                    f"第 {layer_index} 层应包含 {self.n_qubits} 个量子比特，"
                    f"实际为 {len(layer)}。"
                )
            normalized_wires: list[AngleTriple] = []
            for wire_index, gate_angles in enumerate(layer):
                if len(gate_angles) != 3:
                    raise ValueError(
                        f"第 {layer_index} 层第 {wire_index} 个量子比特必须按"
                        " [RX, RY, RZ] 提供 3 个角度。"
                    )
                values = tuple(float(value) for value in gate_angles)
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(
                        f"第 {layer_index} 层第 {wire_index} 个量子比特包含非有限角度。"
                    )
                normalized_wires.append(values)  # type: ignore[arg-type]
            normalized_layers.append(tuple(normalized_wires))

        object.__setattr__(self, "angles", tuple(normalized_layers))


def _checkpoint_state(checkpoint_path: Path) -> Mapping[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"未找到 RiverONE checkpoint：{checkpoint_path}")

    import torch

    payload = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("RiverONE checkpoint 顶层必须是字典。")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("RiverONE checkpoint 必须包含字典字段 'state'。")
    return state


def _infer_shape(state: Mapping[str, Any], vqc_index: int) -> tuple[int, int]:
    available_indices: set[int] = set()
    layers: set[int] = set()
    wires: set[int] = set()

    for key in state:
        match = _STATE_KEY_PATTERN.match(str(key))
        if match is None:
            continue
        current_vqc = int(match.group("vqc"))
        available_indices.add(current_vqc)
        if current_vqc == vqc_index:
            layers.add(int(match.group("layer")))
            wires.add(int(match.group("wire")))

    if vqc_index not in available_indices:
        available = ", ".join(str(index) for index in sorted(available_indices)) or "无"
        raise IndexError(
            f"checkpoint 中不存在 vqc_index={vqc_index}；可用索引：{available}。"
        )

    n_layers = max(layers) + 1
    n_qubits = max(wires) + 1
    for layer in range(n_layers):
        for wire in range(n_qubits):
            for gate in _ROTATION_NAMES:
                key = (
                    f"vqcs.{vqc_index}.variational.l{layer}_w{wire}.{gate}.params"
                )
                if key not in state:
                    raise ValueError(
                        f"checkpoint 缺少 VQC {vqc_index} 的 l{layer}_w{wire}.{gate} 参数。"
                    )
    return n_qubits, n_layers


def _scalar_parameter(value: Any, *, location: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numel") and int(value.numel()) != 1:
        raise ValueError(f"{location} 必须恰好包含一个角度。")
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} 包含非有限角度。")
    return result


def load_riverone_vqc(
    checkpoint_path: str | Path,
    vqc_index: int = 0,
) -> RiverONEVQCSpec:
    """从 RiverONE checkpoint 加载一条训练后的 ``TQVQC``。"""

    if vqc_index < 0:
        raise ValueError("vqc_index 不能为负数。")

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    state = _checkpoint_state(checkpoint)
    n_qubits, n_layers = _infer_shape(state, vqc_index)

    # checkpoint 已保存完整门参数，直接读取即可，不需要导入 TorchQuantum。
    layers: list[list[AngleTriple]] = []
    for layer in range(n_layers):
        wire_angles: list[AngleTriple] = []
        for wire in range(n_qubits):
            values = tuple(
                _scalar_parameter(
                    state[
                        f"vqcs.{vqc_index}.variational."
                        f"l{layer}_w{wire}.{gate_name}.params"
                    ],
                    location=f"l{layer}_w{wire}.{gate_name}.params",
                )
                for gate_name in _ROTATION_NAMES
            )
            wire_angles.append(values)  # type: ignore[arg-type]
        layers.append(wire_angles)

    return RiverONEVQCSpec(
        n_qubits=n_qubits,
        n_layers=n_layers,
        angles=layers,
        reupload_every=_REUPLOAD_EVERY,
    )


def _normalize_amplitudes(
    amplitudes: Sequence[complex] | np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """复现 TorchQuantum ``AmplitudeEncoder`` 的补零和归一化。"""

    raw_values = np.asarray(amplitudes)
    dtype = np.complex128 if np.iscomplexobj(raw_values) else np.float64
    values = np.asarray(raw_values, dtype=dtype)
    if values.ndim != 1:
        raise ValueError("amplitudes 必须是一维数组。")
    if values.size == 0:
        raise ValueError("amplitudes 不能为空。")

    dimension = 1 << n_qubits
    if values.size > dimension:
        raise ValueError(
            f"amplitudes 最多包含 {dimension} 个元素，实际为 {values.size}。"
        )
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        raise ValueError("amplitudes 包含非有限值。")

    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("amplitudes 不能是全零向量。")

    state = np.zeros(dimension, dtype=np.complex128)
    state[: values.size] = values / norm
    return state


def _active_layers(spec: RiverONEVQCSpec) -> Sequence[AngleLayer]:
    # RiverONE 的 re-upload 会直接覆盖状态，之前的门不再影响最终输出。
    start = ((spec.n_layers - 1) // spec.reupload_every) * spec.reupload_every
    return spec.angles[start:]


def _append_variational_layers(
    circuit: Any,
    spec: RiverONEVQCSpec,
) -> None:
    for layer in _active_layers(spec):
        for wire, (theta_rx, theta_ry, theta_rz) in enumerate(layer):
            circuit.rx(theta_rx, wire)
            circuit.ry(theta_ry, wire)
            circuit.rz(theta_rz, wire)
        for wire in range(spec.n_qubits - 1):
            circuit.cx(wire, wire + 1)
        circuit.cx(spec.n_qubits - 1, 0)


def _qiskit_amplitudes(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """把 RiverONE 的 q0-first 状态轴转换为 Qiskit 的小端序。"""

    axes = tuple(reversed(range(n_qubits)))
    ordered = state.reshape((2,) * n_qubits).transpose(axes).reshape(-1)
    # RiverONE 的 WeightEncoder 产生实数。保留实数 dtype 可避免 Qiskit
    # 在大型纯实状态上误走数值更不稳定的复数 isometry 分解。
    if np.all(ordered.imag == 0.0):
        return ordered.real
    return ordered


def _build_qiskit_circuit(spec: RiverONEVQCSpec, state: np.ndarray, basis: str) -> Any:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    circuit = QuantumCircuit(spec.n_qubits, spec.n_qubits)
    circuit.append(
        StatePreparation(_qiskit_amplitudes(state, spec.n_qubits)),
        range(spec.n_qubits),
    )
    _append_variational_layers(circuit, spec)

    # 真机最终仍执行 Z 读出；X/Y 通过标准换基门获得对应期望值。
    if basis == "X":
        for wire in range(spec.n_qubits):
            circuit.h(wire)
    elif basis == "Y":
        for wire in range(spec.n_qubits):
            circuit.sdg(wire)
            circuit.h(wire)
    circuit.measure(range(spec.n_qubits), range(spec.n_qubits))
    return circuit


def riverone_to_qasm2(
    spec: RiverONEVQCSpec,
    amplitudes: Sequence[complex] | np.ndarray,
) -> dict[str, str]:
    """生成 RiverONE X/Y/Z 三种测量基对应的 OpenQASM 2。"""

    from qiskit import qasm2, transpile

    state = _normalize_amplitudes(amplitudes, spec.n_qubits)
    qasm_by_basis: dict[str, str] = {}
    allowed_operations = set(_QASM_BASIS_GATES) | {"measure", "barrier"}

    for basis in _MEASUREMENT_BASES:
        # Qiskit 2.4 对部分 bit-reversal 状态无法直接完成高层综合；先展开
        # StatePreparation 可稳定进入后续基础门分解。
        circuit = _build_qiskit_circuit(spec, state, basis).decompose()
        compiled = transpile(
            circuit,
            basis_gates=list(_QASM_BASIS_GATES),
            optimization_level=1,
        )
        unexpected = set(compiled.count_ops()) - allowed_operations
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise RuntimeError(f"QASM2 编译后仍包含服务器基础门之外的操作：{names}")

        source = qasm2.dumps(compiled)
        if not all(token in source for token in ("OPENQASM 2", "qreg", "creg", "measure")):
            raise RuntimeError(f"{basis} 基未生成完整的 OpenQASM 2。")
        qasm_by_basis[basis] = source

    return qasm_by_basis


__all__ = [
    "RiverONEVQCSpec",
    "load_riverone_vqc",
    "riverone_to_qasm2",
]
