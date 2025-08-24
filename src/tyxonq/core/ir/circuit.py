from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Dict, Optional, Sequence, Tuple
import json
from ...compiler.api import compile as compile_api  # lazy import to avoid hard deps

@dataclass
class Circuit:
    """Minimal intermediate representation (IR) for a quantum circuit.

    Attributes:
        num_qubits: Number of qubits in the circuit.
        ops: A sequence of operation descriptors. The concrete type is left
            open for backends/compilers to interpret (e.g., gate tuples, IR
            node objects). Keeping this generic allows the IR to evolve while
            tests exercise the structural contract.
    """

    num_qubits: int
    ops: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    instructions: List[Tuple[str, Tuple[int, ...]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.num_qubits < 0:
            raise ValueError("num_qubits must be non-negative")
        # Lightweight structural validation: ints used as qubit indices are in range
        for op in self.ops:
            if not isinstance(op, tuple) and not isinstance(op, list):
                raise TypeError("op must be a tuple or list")
            if not op:
                raise ValueError("op cannot be empty")
            if not isinstance(op[0], str):
                raise TypeError("op name must be a string")
            # Validate any int-like argument as qubit index
            for arg in op[1:]:
                if isinstance(arg, int):
                    if arg < 0 or arg >= self.num_qubits:
                        raise ValueError("qubit index out of range in op")
        # Validate instructions
        for inst in self.instructions:
            if not isinstance(inst, tuple) or len(inst) != 2:
                raise TypeError("instruction must be (name, (indices,)) tuple")
            iname, idxs = inst
            if not isinstance(iname, str):
                raise TypeError("instruction name must be a string")
            if not isinstance(idxs, tuple):
                raise TypeError("instruction indices must be a tuple")
            for q in idxs:
                if not isinstance(q, int) or q < 0 or q >= self.num_qubits:
                    raise ValueError("instruction qubit index out of range")

    def with_metadata(self, **kwargs: Any) -> "Circuit":
        """Return a new Circuit with merged metadata (shallow merge)."""
        new_meta = dict(self.metadata)
        new_meta.update(kwargs)
        return replace(self, metadata=new_meta)

    # ---- Lightweight helpers ----
    def gate_count(self, gate_list: Optional[Sequence[str]] = None) -> int:
        """Count ops by name. If gate_list is provided, count only those (case-insensitive)."""
        if gate_list is None:
            return len(self.ops)
        names = {str(x).lower() for x in (gate_list if isinstance(gate_list, (list, tuple, set)) else [gate_list])}
        count = 0
        for op in self.ops:
            if str(op[0]).lower() in names:
                count += 1
        return count

    def gate_summary(self) -> Dict[str, int]:
        """Return a mapping of op name (lower-case) to frequency."""
        summary: Dict[str, int] = {}
        for op in self.ops:
            k = str(op[0]).lower()
            summary[k] = summary.get(k, 0) + 1
        return summary

    def extended(self, extra_ops: Sequence[Sequence[Any]]) -> "Circuit":
        """Return a new Circuit with ops extended by extra_ops (no mutation)."""
        new_ops = list(self.ops) + [tuple(op) for op in extra_ops]
        return Circuit(num_qubits=self.num_qubits, ops=new_ops, metadata=dict(self.metadata), instructions=list(self.instructions))

    def compose(self, other: "Circuit", indices: Optional[Sequence[int]] = None) -> "Circuit":
        """Append another Circuit's ops. If `indices` given, remap other's qubits by indices[i]."""
        if indices is None:
            if other.num_qubits != self.num_qubits:
                raise ValueError("compose requires equal num_qubits when indices is None")
            mapped_ops = list(other.ops)
        else:
            # indices maps other's logical i -> self physical indices[i]
            idx_list = list(indices)
            def _map_op(op: Sequence[Any]) -> tuple:
                mapped: List[Any] = [op[0]]
                for a in op[1:]:
                    if isinstance(a, int):
                        if a < 0 or a >= len(idx_list):
                            raise ValueError("compose indices out of range for other circuit")
                        mapped.append(int(idx_list[a]))
                    else:
                        mapped.append(a)
                return tuple(mapped)
            mapped_ops = [_map_op(op) for op in other.ops]
        return self.extended(mapped_ops)

    def remap_qubits(self, mapping: Dict[int, int], *, new_num_qubits: Optional[int] = None) -> "Circuit":
        """Return a new Circuit with qubit indices remapped according to `mapping`.

        All int arguments in ops are treated as qubit indices and must be present in mapping.
        """
        def _remap_op(op: Sequence[Any]) -> tuple:
            out: List[Any] = [op[0]]
            for a in op[1:]:
                if isinstance(a, int):
                    if a not in mapping:
                        raise KeyError(f"qubit {a} missing in mapping")
                    out.append(int(mapping[a]))
                else:
                    out.append(a)
            return tuple(out)
        nn = int(new_num_qubits) if new_num_qubits is not None else self.num_qubits
        return Circuit(num_qubits=nn, ops=[_remap_op(op) for op in self.ops], metadata=dict(self.metadata), instructions=list(self.instructions))

    def positional_logical_mapping(self) -> Dict[int, int]:
        """Return positional->logical mapping from explicit instructions or measure_z ops."""
        # Prefer explicit instructions if present
        measures = [idxs for (n, idxs) in self.instructions if str(n).lower() == "measure"]
        if measures:
            pos_to_logical: Dict[int, int] = {}
            for pos, idxs in enumerate(measures):
                if not idxs:
                    continue
                pos_to_logical[pos] = int(idxs[0])
            return pos_to_logical or {i: i for i in range(self.num_qubits)}
        # Fallback to scanning measure_z ops
        pos_to_logical: Dict[int, int] = {}
        pos = 0
        for op in self.ops:
            if op and str(op[0]).lower() == "measure_z":
                q = int(op[1])
                pos_to_logical[pos] = q
                pos += 1
        return pos_to_logical or {i: i for i in range(self.num_qubits)}

    def inverse(self, *, strict: bool = False) -> "Circuit":
        """Return a unitary inverse circuit for supported ops (h, cx, rz).

        Non-unitary ops like measure/reset/barrier are skipped unless strict=True (then error).
        Unknown ops raise if strict=True, else skipped.
        """
        inv_ops: List[tuple] = []
        for op in reversed(self.ops):
            name = str(op[0]).lower()
            if name == "h":
                inv_ops.append(("h", int(op[1])))
            elif name == "cx":
                inv_ops.append(("cx", int(op[1]), int(op[2])))
            elif name == "rz":
                inv_ops.append(("rz", int(op[1]), -float(op[2])))
            elif name in ("measure_z", "reset", "barrier"):
                if strict:
                    raise ValueError(f"non-unitary op not invertible: {name}")
                continue
            else:
                if strict:
                    raise NotImplementedError(f"inverse not implemented for op: {name}")
                continue
        return Circuit(num_qubits=self.num_qubits, ops=inv_ops, metadata=dict(self.metadata), instructions=list(self.instructions))

    # ---- JSON IO (provider-agnostic, minimal) ----
    def to_json_obj(self) -> Dict[str, Any]:
        return {
            "num_qubits": int(self.num_qubits),
            "ops": list(self.ops),
            "metadata": dict(self.metadata),
            "instructions": [(n, list(idxs)) for (n, idxs) in self.instructions],
        }

    def to_json_str(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_json_obj(), ensure_ascii=False, indent=indent)


    # ---- Provider adapters (thin convenience wrappers) ----
    def to_openqasm(self) -> str:
        """Serialize this IR circuit to OpenQASM 2 using the compiler facade.

        Delegates to compiler API (provider='qiskit', output='qasm2').
        """

        r = compile_api(self, provider="qiskit", output="qasm2")
        return r["circuit"]  # type: ignore[return-value]

    def compile(self, *, target: str = "ir", provider: str = "default", add_measures: bool = True, **options: Any) -> Any:
        """Compile/convert this circuit via the compiler facade.

        - target: 编译产物类型或执行目标：
          - "ir"/"native"/"tyxonq": 返回 IR（经 native pipeline 降低后）
          - "openqasm": OpenQASM 2 字符串（需 provider='qiskit'）
          - "qiskit": qiskit.QuantumCircuit（需 provider='qiskit'）
          - "json": 先经 native compiler 降低 IR，再返回 JSON 序列化
        """
        t = str(target).lower()
        prov = str(provider).lower()
        if prov in ("native", "tyxonq", "default"):
            prov = "default"

        if t in ("ir", "native", "tyxonq"):
            # Always lower via native pipeline first, returning IR
            r = compile_api(self, provider="default", output="ir")
            return r["circuit"]
        if t == "json":
            # Lower then serialize, so结构变化能反映到JSON
            r = compile_api(self, provider="default", output="ir")
            lowered = r["circuit"]
            try:
                return lowered.to_json_str()  # type: ignore[attr-defined]
            except Exception:
                return self.to_json_str()
        if t == "openqasm":
            # provider selection: default to qiskit for qasm emission
            out = compile_api(self, provider="qiskit", output="qasm2", options={"add_measures": add_measures})
            return out["circuit"]
        if t == "qiskit":
            r = compile_api(self, provider="qiskit", output="qiskit", options={"add_measures": add_measures})
            return r["circuit"]
        raise ValueError(f"Unsupported compile target: {target}")

    def run(
        self,
        *,
        provider: Optional[str] = None,
        device: Optional[str] = None,
        shots: int = 1024,
        compiler: str = "qiskit",
        auto_compile: bool = True,
        **opts: Any,
    ) -> Any:
        """Submit this circuit to a provider/device.

        - simulator/local: 直接以 IR 运行
        - 其它 provider:
          - auto_compile=True: compile→openqasm 后提交
          - auto_compile=False: 原样以 IR 提交（驱动需支持 IR）
        """
        # Resolve defaults
        from ...devices.hardware import config as hwcfg

        prov = provider or hwcfg.get_default_provider()
        dev = device or hwcfg.get_default_device()
        tok = hwcfg.get_token(provider=prov, device=dev)

        # Simulator: IR直跑
        if prov in ("simulator", "local"):
            from ...devices.simulators import driver as sdrv

            return sdrv.run(dev, tok, circuit=self, shots=shots, **opts)

        # Hardware providers
        def _hw_driver(p: str):
            if p == "tyxonq":
                from ...devices.hardware.tyxonq import driver as drv

                return drv
            if p == "ibm":
                from ...devices.hardware.ibm import driver as drv

                return drv
            raise ValueError(f"Unsupported provider: {p}")

        drv = _hw_driver(prov)
        if auto_compile:
            qasm = self.compile(target="openqasm", provider="qiskit")
            return drv.submit_task(dev, tok, source=qasm, shots=shots, **opts)
        # No compile: submit IR as-is (driver must support IR path)
        return drv.submit_task(dev, tok, circuit=self, shots=shots, **opts)

    # ---- Task helpers for cloud.api thin wrappers ----
    def get_task_details(self, task: Any, *, prettify: bool = False) -> Dict[str, Any]:
        # For simulator tasks, the object carries a .results() method
        if hasattr(task, "results"):
            try:
                return task.results()
            except Exception:
                pass
        # Hardware tasks: delegate to provider driver via device string
        dev = getattr(task, "device", None)
        if dev is None:
            raise ValueError("Task handle missing device information")
        dev_str = str(dev)
        prov = (dev_str.split("::", 1)[0]) if "::" in dev_str else "simulator"
        from ...devices.base import resolve_driver
        from ...devices.hardware import config as hwcfg

        tok = hwcfg.get_token(provider=prov, device=dev_str)
        drv = resolve_driver(prov, dev_str)
        return drv.get_task_details(task, tok)

    def cancel(self, task: Any) -> Any:
        dev = getattr(task, "device", None)
        if dev is None:
            raise ValueError("Task handle missing device information")
        dev_str = str(dev)
        prov = (dev_str.split("::", 1)[0]) if "::" in dev_str else "simulator"
        from ...devices.base import resolve_driver
        from ...devices.hardware import config as hwcfg

        tok = hwcfg.get_token(provider=prov, device=dev_str)
        drv = resolve_driver(prov, dev_str)
        if hasattr(drv, "remove_task"):
            return drv.remove_task(task, tok)
        raise NotImplementedError("cancel not supported for this provider/task type")

    def submit_task(
        self,
        *,
        provider: Optional[str] = None,
        device: Optional[str] = None,
        shots: int = 1024,
        compiler: str = "qiskit",
        auto_compile: bool = True,
        **opts: Any,
    ) -> Any:
        # Submit is an alias of run with identical semantics
        return self.run(provider=provider, device=device, shots=shots, compiler=compiler, auto_compile=auto_compile, **opts)

    # Note: builder-style gate helpers have been moved to `CircuitBuilder`.

    # Instruction helpers
    def add_measure(self, *qubits: int) -> "Circuit":
        new_inst = list(self.instructions)
        for q in qubits:
            new_inst.append(("measure", (int(q),)))
        return replace(self, instructions=new_inst)

    def add_reset(self, *qubits: int) -> "Circuit":
        new_inst = list(self.instructions)
        for q in qubits:
            new_inst.append(("reset", (int(q),)))
        return replace(self, instructions=new_inst)

    def add_barrier(self, *qubits: int) -> "Circuit":
        new_inst = list(self.instructions)
        if qubits:
            new_inst.append(("barrier", tuple(int(q) for q in qubits)))
        else:
            new_inst.append(("barrier", tuple(range(self.num_qubits))))
        return replace(self, instructions=new_inst)

    # ---- Builder-style ergonomic gate helpers (in-place; return self) ----
    def h(self, q: int):
        self.ops.append(("h", int(q)))
        return self

    def H(self, q: int):
        return self.h(q)

    def rz(self, q: int, theta: Any):
        self.ops.append(("rz", int(q), theta))
        return self

    def RZ(self, q: int, theta: Any):
        return self.rz(q, theta)

    def rx(self, q: int, theta: Any):
        self.ops.append(("rx", int(q), theta))
        return self

    def RX(self, q: int, theta: Any):
        return self.rx(q, theta)

    def cx(self, c: int, t: int):
        self.ops.append(("cx", int(c), int(t)))
        return self

    def CX(self, c: int, t: int):
        return self.cx(c, t)

    def cnot(self, c: int, t: int):
        return self.cx(c, t)

    def CNOT(self, c: int, t: int):
        return self.cx(c, t)

    def measure_z(self, q: int):
        self.ops.append(("measure_z", int(q)))
        return self

    def MEASURE_Z(self, q: int):
        return self.measure_z(q)
    
    @classmethod
    def from_json_obj(cls, obj: Dict[str, Any]) -> "Circuit":
        inst_raw = obj.get("instructions", [])
        inst: List[Tuple[str, Tuple[int, ...]]] = []
        for n, idxs in inst_raw:
            inst.append((str(n), tuple(int(x) for x in idxs)))
        return cls(
            num_qubits=int(obj.get("num_qubits", 0)),
            ops=list(obj.get("ops", [])),
            metadata=dict(obj.get("metadata", {})),
            instructions=inst,
        )

    @classmethod
    def from_json_str(cls, s: str) -> "Circuit":
        return cls.from_json_obj(json.loads(s))


@dataclass
class Hamiltonian:
    """IR for a Hamiltonian.

    The `terms` field may contain a backend-specific structure, such as a
    Pauli-sum, sparse representation, or dense matrix. The type is intentionally
    loose at this stage and will be specialized by compiler stages or devices.
    """

    terms: Any


# ---- Module-level task helpers (for cloud.api thin delegation) ----
def get_task_details(task: Any, *, prettify: bool = False) -> Dict[str, Any]:
    dev = getattr(task, "device", None)
    if dev is None:
        # simulator inline task may still provide results()
        if hasattr(task, "results"):
            try:
                return task.results()
            except Exception:
                pass
        raise ValueError("Task handle missing device information")
    dev_str = str(dev)
    prov = (dev_str.split("::", 1)[0]) if "::" in dev_str else "simulator"
    from ...devices.base import resolve_driver
    from ...devices.hardware import config as hwcfg

    tok = hwcfg.get_token(provider=prov, device=dev_str)
    drv = resolve_driver(prov, dev_str)
    return drv.get_task_details(task, tok)


def cancel_task(task: Any) -> Any:
    dev = getattr(task, "device", None)
    if dev is None:
        raise ValueError("Task handle missing device information")
    dev_str = str(dev)
    prov = (dev_str.split("::", 1)[0]) if "::" in dev_str else "simulator"
    from ...devices.base import resolve_driver
    from ...devices.hardware import config as hwcfg

    tok = hwcfg.get_token(provider=prov, device=dev_str)
    drv = resolve_driver(prov, dev_str)
    if hasattr(drv, "remove_task"):
        return drv.remove_task(task, tok)
    raise NotImplementedError("cancel not supported for this provider/task type")

