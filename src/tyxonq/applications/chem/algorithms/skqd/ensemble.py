"""SKQD 线路集合与样本合同：先逐线路撤销映射，再合并计数。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from operator import index
from pathlib import Path

import numpy as np

from tyxonq.core.ir.circuit import Circuit


def validate_permutation(permutation, size):
    """验证 old mode -> new mode 的完整排列。"""
    values = tuple(index(i) for i in permutation)
    if sorted(values) != list(range(size)):
        raise ValueError("permutation must contain each mode exactly once")
    return values


@dataclass
class CircuitRecord:
    """经典字符串按 c[n-1]...c[0] 存储；映射描述每个 c 对应的线路模式。"""

    circuit_id: str
    k: int
    time: float
    randomization_id: int = 0
    seed: int | None = None
    sampled_group_ids: tuple[int, ...] = ()
    mode_permutation: tuple[int, ...] = ()
    measurement_map: tuple[int | None, ...] = ()
    basis_id: str = "input"
    diagnostics: dict = field(default_factory=dict)


@dataclass
class SKQDEnsemble:
    """尚未测量的原生线路；构造此对象不会运行任何设备。"""

    circuits: list[Circuit]
    records: list[CircuitRecord]
    norb: int
    nelec: tuple[int, int]
    method: str

    def __post_init__(self):
        if len(self.circuits) != len(self.records) or not self.records:
            raise ValueError("one non-empty record is required for each circuit")
        if len({r.circuit_id for r in self.records}) != len(self.records):
            raise ValueError("circuit_id must be unique")
        if any(c.num_qubits != 2 * self.norb for c in self.circuits):
            raise ValueError("logical circuits must have 2*norb qubits")

    def save(self, path):
        """保存原生门及 unitary 矩阵缓存，供另一进程独立采样；不使用 pickle。"""
        def encode(value):
            if isinstance(value, np.ndarray):
                return {"__skqd_array__": True, "real": value.real.tolist(),
                        "imag": value.imag.tolist() if np.iscomplexobj(value) else None}
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f"cannot serialize {type(value).__name__}")

        data = {"schema": "tyxonq.skqd.ensemble.v1", "norb": self.norb,
                "nelec": self.nelec, "method": self.method,
                "records": [asdict(r) for r in self.records],
                "circuits": [{"ir": c.to_json_obj(),
                              "unitary_cache": getattr(c, "_unitary_cache", {})} for c in self.circuits]}
        Path(path).write_text(json.dumps(data, default=encode, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        """恢复包含复数小矩阵的原生线路，不调用第三方线路库。"""
        def decode(value):
            if value.get("__skqd_array__") is True:
                real = np.asarray(value["real"])
                return real if value["imag"] is None else real + 1j * np.asarray(value["imag"])
            return value

        data = json.loads(Path(path).read_text(encoding="utf-8"), object_hook=decode)
        if data.get("schema") != "tyxonq.skqd.ensemble.v1":
            raise ValueError("unsupported SKQD ensemble schema")
        circuits = []
        for stored in data["circuits"]:
            circuit = Circuit.from_json_obj(stored["ir"])
            circuit._unitary_cache = stored["unitary_cache"]
            if any(op[0] == "unitary" and op[-1] not in circuit._unitary_cache for op in circuit.ops):
                raise ValueError("serialized circuit is missing a unitary matrix")
            circuits.append(circuit)
        records = []
        for record in data["records"]:
            for key in ("mode_permutation", "measurement_map", "sampled_group_ids"):
                record[key] = tuple(record[key])
            records.append(CircuitRecord(**record))
        return cls(circuits, records, data["norb"], tuple(data["nelec"]), data["method"])


def merge_skqd_samples(results: dict[str, dict[str, int]], ensemble: SKQDEnsemble):
    """返回 SQD 内部 [beta倒序|alpha倒序] counts，按实际 shots 加权。"""
    if set(results) != {r.circuit_id for r in ensemble.records}:
        raise ValueError("sample circuit IDs must match ensemble records exactly")
    if len({r.basis_id for r in ensemble.records}) != 1:
        raise ValueError("samples in different orbital bases cannot be merged")
    width = 2 * ensemble.norb
    merged = Counter()
    for record in ensemble.records:
        perm = validate_permutation(record.mode_permutation or range(width), width)
        mapping = record.measurement_map or tuple(range(width))
        mapped = [i for i in mapping if i is not None]
        if sorted(mapped) != list(range(width)):
            raise ValueError("measurement_map must measure each logical mode exactly once")
        inverse_measurement = {mode: c for c, mode in enumerate(mapping) if mode is not None}
        for text, count in results[record.circuit_id].items():
            # 多 classical register 的空格仅是显示分隔符，位映射仍须显式提供。
            bits = text.replace(" ", "")
            if len(bits) != len(mapping) or set(bits) - {"0", "1"}:
                raise ValueError("count keys must match the declared classical register width")
            count = index(count)
            if count < 0:
                raise ValueError("counts must be non-negative integers")
            logical = [bits[-1 - inverse_measurement[perm[p]]] for p in range(width)]
            merged["".join(reversed(logical))] += count
    if sum(merged.values()) == 0:
        raise ValueError("samples must contain at least one shot")
    return dict(merged)


def sample_statevector_ensemble(ensemble: SKQDEnsemble, *, shots: int = 500, seed=24,
                               max_qubits: int = 20):
    """显式调用本地 TyxonQ 模拟器；只用于可控小体系，输出统一经典位顺序。"""
    shots = index(shots)
    if shots < 1 or 2 * ensemble.norb > max_qubits:
        raise ValueError("shots must be positive and circuit must fit max_qubits")
    rng = np.random.default_rng(seed)
    output = {}
    for circuit, record in zip(ensemble.circuits, ensemble.records):
        if record.measurement_map != tuple(range(circuit.num_qubits)):
            raise ValueError("local statevector sampling requires the identity measurement_map")
        state = np.asarray(circuit.state(engine="statevector", backend="numpy", form="numpy"))
        probs = np.abs(state.reshape(-1)) ** 2
        probs /= probs.sum()
        draws = rng.choice(len(probs), size=shots, p=probs)
        # TyxonQ statevector 的最高有效位是 q0；转成 c[n-1]...c[0] 文本。
        counts = Counter(format(int(i), f"0{circuit.num_qubits}b")[::-1] for i in draws)
        output[record.circuit_id] = dict(counts)
    return output
