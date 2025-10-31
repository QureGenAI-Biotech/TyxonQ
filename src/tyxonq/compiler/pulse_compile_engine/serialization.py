"""Pulse circuit serialization utilities.

This module provides utilities for serializing and deserializing pulse circuits,
enabling file I/O, cloud submission, and inter-process communication.

Supports两种序列化格式：
    1. JSON格式 - 人类可读，适合配置文件和调试
    2. Pickle格式 - Python原生，保留对象类型

使用场景：
    - 保存编译好的pulse circuit到文件
    - 通过网络传输pulse circuit（云端提交）
    - 跨进程共享pulse circuit（分布式计算）
"""

from __future__ import annotations

import json
import pickle
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit


def serialize_pulse_circuit_to_json(circuit: "Circuit") -> str:
    """将pulse circuit序列化为JSON字符串.
    
    This format is human-readable and suitable for:
        - Configuration files
        - Debugging and inspection
        - Version control (text-based)
        - Cross-language compatibility
    
    Args:
        circuit: Compiled pulse circuit (with pulse or pulse_inline operations)
    
    Returns:
        JSON string representation
    
    Example:
        >>> pulse_circuit = compiler.compile(c, output="pulse_ir", inline_pulses=True)
        >>> json_str = serialize_pulse_circuit_to_json(pulse_circuit)
        >>> with open("pulse_circuit.json", "w") as f:
        ...     f.write(json_str)
    
    Note:
        - Requires inline_pulses=True for full serialization
        - Waveforms are converted to {"type": ..., "args": [...]} format
        - Python objects (like waveform instances) are not preserved
    """
    # Extract circuit data
    data = {
        "num_qubits": circuit.num_qubits,
        "ops": list(circuit.ops),
        "metadata": _serialize_metadata(circuit.metadata),
        "instructions": list(circuit.instructions)
    }
    
    return json.dumps(data, indent=2)


def deserialize_pulse_circuit_from_json(json_str: str) -> "Circuit":
    """从JSON字符串反序列化pulse circuit.
    
    Args:
        json_str: JSON string from serialize_pulse_circuit_to_json()
    
    Returns:
        Reconstructed Circuit object
    
    Example:
        >>> with open("pulse_circuit.json", "r") as f:
        ...     json_str = f.read()
        >>> pulse_circuit = deserialize_pulse_circuit_from_json(json_str)
        >>> result = pulse_circuit.run()
    """
    from tyxonq import Circuit
    
    data = json.loads(json_str)
    
    return Circuit(
        num_qubits=data["num_qubits"],
        ops=data.get("ops", []),
        metadata=data.get("metadata", {}),
        instructions=data.get("instructions", [])
    )


def serialize_pulse_circuit_to_pickle(circuit: "Circuit") -> bytes:
    """将pulse circuit序列化为pickle字节流.
    
    This format preserves Python objects and is suitable for:
        - Fast serialization/deserialization
        - Preserving waveform objects (no inlining needed)
        - Python-to-Python communication
        - Distributed computing (same Python version)
    
    Args:
        circuit: Compiled pulse circuit (pulse or pulse_inline)
    
    Returns:
        Pickle bytes
    
    Example:
        >>> pulse_circuit = compiler.compile(c, output="pulse_ir")
        >>> with open("pulse_circuit.pkl", "wb") as f:
        ...     f.write(serialize_pulse_circuit_to_pickle(pulse_circuit))
    
    Note:
        - Preserves waveform objects (no need for inline_pulses=True)
        - Only works with same Python version
        - Binary format, not human-readable
    """
    return pickle.dumps(circuit)


def deserialize_pulse_circuit_from_pickle(data: bytes) -> "Circuit":
    """从pickle字节流反序列化pulse circuit.
    
    Args:
        data: Pickle bytes from serialize_pulse_circuit_to_pickle()
    
    Returns:
        Reconstructed Circuit object with preserved waveform objects
    
    Example:
        >>> with open("pulse_circuit.pkl", "rb") as f:
        ...     pulse_circuit = deserialize_pulse_circuit_from_pickle(f.read())
        >>> result = pulse_circuit.run()
    """
    return pickle.loads(data)


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """序列化metadata中的特殊对象.
    
    Handles:
        - pulse_library: Convert waveform objects to dicts
        - Other objects: Keep as-is (JSON-serializable items)
    
    Args:
        metadata: Circuit metadata
    
    Returns:
        JSON-serializable metadata dict
    """
    serialized = {}
    
    for key, value in metadata.items():
        if key == "pulse_library":
            # Serialize waveform objects
            serialized[key] = _serialize_pulse_library(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            # JSON-safe primitives
            serialized[key] = value
        elif isinstance(value, (list, tuple)):
            serialized[key] = list(value)
        elif isinstance(value, dict):
            serialized[key] = dict(value)
        else:
            # Skip non-serializable objects
            serialized[key] = str(value)
    
    return serialized


def _serialize_pulse_library(pulse_library: Dict[str, Any]) -> Dict[str, Dict]:
    """序列化pulse_library中的waveform对象.
    
    Args:
        pulse_library: Dict of {pulse_key: waveform_object}
    
    Returns:
        Dict of {pulse_key: waveform_dict}
    """
    serialized = {}
    
    for key, waveform in pulse_library.items():
        if hasattr(waveform, "qasm_name") and hasattr(waveform, "to_args"):
            # Standard waveform object
            serialized[key] = {
                "type": waveform.qasm_name(),
                "args": waveform.to_args(),
                "class": type(waveform).__name__
            }
        else:
            # Unknown object, use string representation
            serialized[key] = {"type": "unknown", "repr": str(waveform)}
    
    return serialized


# ========== 便利函数 ==========

def save_pulse_circuit(circuit: "Circuit", filepath: str, format: str = "json") -> None:
    """保存pulse circuit到文件.
    
    Args:
        circuit: Compiled pulse circuit
        filepath: Output file path
        format: "json" or "pickle" (default: "json")
    
    Example:
        >>> save_pulse_circuit(pulse_circuit, "my_pulse.json", format="json")
        >>> save_pulse_circuit(pulse_circuit, "my_pulse.pkl", format="pickle")
    """
    if format == "json":
        with open(filepath, "w") as f:
            f.write(serialize_pulse_circuit_to_json(circuit))
    elif format == "pickle":
        with open(filepath, "wb") as f:
            f.write(serialize_pulse_circuit_to_pickle(circuit))
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'.")


def load_pulse_circuit(filepath: str, format: str = "json") -> "Circuit":
    """从文件加载pulse circuit.
    
    Args:
        filepath: Input file path
        format: "json" or "pickle" (default: "json")
    
    Returns:
        Loaded Circuit object
    
    Example:
        >>> pulse_circuit = load_pulse_circuit("my_pulse.json", format="json")
        >>> result = pulse_circuit.run()
    """
    if format == "json":
        with open(filepath, "r") as f:
            return deserialize_pulse_circuit_from_json(f.read())
    elif format == "pickle":
        with open(filepath, "rb") as f:
            return deserialize_pulse_circuit_from_pickle(f.read())
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'.")
