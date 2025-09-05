from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Dict, Optional, Sequence, Tuple, overload, Literal
import time
import warnings
import json
from ...compiler.api import compile as compile_api  # lazy import to avoid hard deps
import warnings

# ---- Global defaults for chainable stages (configurable via top-level API) ----
_GLOBAL_COMPILE_DEFAULTS: Dict[str, Any] = {}
_GLOBAL_DEVICE_DEFAULTS: Dict[str, Any] = {}
_GLOBAL_POSTPROC_DEFAULTS: Dict[str, Any] = {"method": None}


def set_global_compile_defaults(options: Dict[str, Any]) -> Dict[str, Any]:
    _GLOBAL_COMPILE_DEFAULTS.update(dict(options))
    return dict(_GLOBAL_COMPILE_DEFAULTS)


def get_global_compile_defaults() -> Dict[str, Any]:
    return dict(_GLOBAL_COMPILE_DEFAULTS)


def set_global_device_defaults(options: Dict[str, Any]) -> Dict[str, Any]:
    _GLOBAL_DEVICE_DEFAULTS.update(dict(options))
    return dict(_GLOBAL_DEVICE_DEFAULTS)


def get_global_device_defaults() -> Dict[str, Any]:
    return dict(_GLOBAL_DEVICE_DEFAULTS)


def set_global_postprocessing_defaults(options: Dict[str, Any]) -> Dict[str, Any]:
    _GLOBAL_POSTPROC_DEFAULTS.update(dict(options))
    if "method" not in _GLOBAL_POSTPROC_DEFAULTS:
        _GLOBAL_POSTPROC_DEFAULTS["method"] = None
    return dict(_GLOBAL_POSTPROC_DEFAULTS)


def get_global_postprocessing_defaults() -> Dict[str, Any]:
    base = dict(_GLOBAL_POSTPROC_DEFAULTS)
    if "method" not in base:
        base["method"] = None
    return base

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
    ops: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    instructions: List[Tuple[str, Tuple[int, ...]]] = field(default_factory=list)

    def __init__(self, num_qubits: int, ops: Optional[List[Any]] = None, 
                 metadata: Optional[Dict[str, Any]] = None,
                 instructions: Optional[List[Tuple[str, Tuple[int, ...]]]] = None,
                 # Compile-stage defaults (visible, with defaults)
                 compile_engine: str = "default",
                 compile_output: str = "ir",
                 compile_target: str = "simulator::statevector",
                 compile_options: Optional[Dict[str, Any]] = None,
                 # Device-stage defaults
                 device_provider: str = "simulator",
                 device_device: str = "statevector",
                 device_shots: int = 1024,
                 device_options: Optional[Dict[str, Any]] = None,
                 # Result handling: whether to fetch final result (None=auto)
                 wait_async_result: Optional[bool] = False,
                 # Postprocessing defaults
                 postprocessing_method: Optional[str] = None,
                 postprocessing_options: Optional[Dict[str, Any]] = None,
                 # Draw defaults
                 draw_output: Optional[str] = None,
                 # Optional pre-compiled or provider-native source
                 source: Optional[Any] = None):
        """Initialize a Circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit.
            ops: List of operations. Defaults to empty list if not provided.
            metadata: Circuit metadata. Defaults to empty dict if not provided.
            instructions: List of instructions. Defaults to empty list if not provided.
        """
        self.num_qubits = num_qubits
        self.ops = ops if ops is not None else []
        self.metadata = metadata if metadata is not None else {}
        self.instructions = instructions if instructions is not None else []
        # Chainable stage options seeded from global defaults
        self._compile_opts: Dict[str, Any] = get_global_compile_defaults()
        self._device_opts: Dict[str, Any] = get_global_device_defaults()
        self._post_opts: Dict[str, Any] = get_global_postprocessing_defaults()
        # Visible defaults applied (constructor-specified overrides)
        self._compile_engine: str = str(compile_engine)
        self._compile_output: str = str(compile_output)
        self._compile_target: str = str(compile_target)
        if compile_options:
            self._compile_opts.update(dict(compile_options))
        # Device defaults
        self._device_opts.setdefault("provider", str(device_provider))
        self._device_opts.setdefault("device", str(device_device))
        self._device_opts.setdefault("shots", int(device_shots))
        if device_options:
            self._device_opts.update(dict(device_options))
        # Result handling
        self._wait_async_result: Optional[bool] = wait_async_result
        # Postprocessing defaults
        if postprocessing_options:
            self._post_opts.update(dict(postprocessing_options))
        if "method" not in self._post_opts:
            self._post_opts["method"] = postprocessing_method

        # Optional direct-execution source (e.g., QASM string or provider object)
        self._source = source
        # Draw defaults (e.g., "text", "mpl", "latex")
        self._draw_output: Optional[str] = str(draw_output) if draw_output is not None else None

        # Ensure structural validation runs even with custom __init__
        self.__post_init__()

    # Context manager support for simple builder-style usage in tests
    def __enter__(self) -> "Circuit":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Do not suppress exceptions
        return False

    # Builder compatibility: expose .circuit() to return self
    def circuit(self) -> "Circuit":
        return self

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

    # ---- Chainable configuration stages ----
    def device(self, **options: Any) -> "Circuit":
        """Set device options for chainable configuration."""
        self._device_opts.update(dict(options))
        return self

    def postprocessing(self, **options: Any) -> "Circuit":
        """Set postprocessing options for chainable configuration."""
        self._post_opts.update(dict(options))
        if "method" not in self._post_opts:
            self._post_opts["method"] = None
        return self

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

    # ---- Analysis helpers (lightweight, backend-agnostic) ----
    def count_flop(self) -> Optional[int]:
        """Return a heuristic FLOP estimate for statevector simulation.

        This avoids tensor network dependencies. The estimate is coarse:
        - 1q gate ~ O(2^n)
        - 2q gate ~ O(2^(n+1))
        - other gates ignored
        Returns None if n is not available.
        """
        try:
            n = int(self.num_qubits)
        except Exception:
            return None
        flop: int = 0
        for op in self.ops:
            name = str(op[0]).lower()
            if name in ("h", "rx", "ry", "rz", "x", "y", "z", "s", "sdg"):
                flop += 1 << n
            elif name in ("cx", "cz", "cy", "cnot", "rxx", "rzz"):
                flop += 1 << (n + 1)
        return flop

    def get_circuit_summary(self):  # pragma: no cover - optional pandas
        """Return a dict summarizing the circuit if pandas is available.

        Columns: #qubits, #gates, #CNOT, #multicontrol, depth (if qiskit available), #FLOP (heuristic).
        """

        n_qubits = self.num_qubits
        n_gates = len(self.ops)
        n_cnot = self.gate_count(["cnot", "cx"])
        n_mc = sum(1 for op in self.ops if "multicontrol" in str(op[0]).lower())
        depth = None
        try:
            from ...compiler.compile_engine.qiskit.dialect import to_qiskit  # type: ignore
            qc = to_qiskit(self, add_measures=False)
            depth = qc.depth()
        except Exception:
            depth = None
        flop = self.count_flop()
        return {"#qubits": n_qubits, "#gates": [n_gates], "#CNOT": [n_cnot], "#multicontrol": [n_mc], "depth": [depth], "#FLOP": [flop]}

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

    @overload
    def compile(self, *, provider: None = ..., output: None = ..., target: Any | None = ..., options: Dict[str, Any] | None = ...) -> "Circuit": ...

    @overload
    def compile(self, *, provider: str = ..., output: str = ..., target: Any | None = ..., options: Dict[str, Any] | None = ...) -> Any: ...

    def compile(
        self,
        *,
        compile_engine: Optional[str] = None,
        output: Optional[str] = None,
        target: Any | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Any:
        """Delegate to compiler.api.compile or act as chainable setter when no args.

        - Chainable模式：若 provider/output/target/options 全为 None，则返回 self（不触发编译）。
        - 编译模式：转发到 compiler.api.compile，options 与已记录的 _compile_opts 合并。
        """
        # Chainable: no explicit args means only marking intent
        if compile_engine is None and output is None and target is None and options is None:
            return self

        # If a direct source is present, skip compilation entirely
        if self._source is not None:
            return self

        # Delegate to compiler facade exactly following its contract
        prov = compile_engine or "default"
        out = output or "ir"
        merged_opts = dict(self._compile_opts)
        if options:
            merged_opts.update(options)
        # compiler/api.compile 现在只接受 (circuit, compile_engine, output, options)
        res = compile_api(self, compile_engine=prov, output=out, options=merged_opts)
        return res["circuit"]

    def run(
        self,
        *,
        provider: Optional[str] = None,
        device: Optional[str] = None,
        shots: int = 1024,
        wait_async_result: Optional[bool] = False,
        **opts: Any,
    ) -> Any:
        """执行电路：
        - 若构造时提供了 source，则直接按 source 提交给设备层；
        - 否则，先按 compile_engine/output/target 进行编译，再根据产物类型提交。
        注意：不在此处补测量或发出告警；该逻辑归属编译阶段。
        """
        from ...devices import base as device_base
        from ...devices.hardware import config as hwcfg

        # Merge device options with call-time overrides and extract reserved keys
        dev_opts = {**self._device_opts, **opts}
        dev_provider = dev_opts.pop("provider", provider)
        dev_device = dev_opts.pop("device", device)
        dev_shots = int(dev_opts.pop("shots", shots))

        # If pre-compiled/native source exists, submit directly
        if self._source is not None:
            tasks = device_base.run(
                provider=dev_provider,
                device=dev_device,
                source=self._source,
                shots=dev_shots,
                **dev_opts,
            )
        else:
            # Compile first using current defaults
            compiled = self.compile(
                compile_engine=self._compile_engine,
                output=self._compile_output,
                target=self._compile_target,
                options=self._compile_opts,
            )
            # For hardware providers, ensure we submit provider-native source (e.g., qasm2)
            prov_norm = (dev_provider or "").lower() if isinstance(dev_provider, str) else str(dev_provider).lower()
            # is_hw = prov_norm not in ("simulator", "local", "")
            is_tyxonq_hw = prov_norm in ("tyxonq")
            if isinstance(compiled, str):
                source_to_submit = compiled
            elif is_tyxonq_hw:
                # Auto-compile to qasm2 for hardware submission via qiskit provider
                qiskit_opts = dict(self._compile_opts)
                if not qiskit_opts.get("basis_gates"):
                    qiskit_opts["basis_gates"] = ["cx", "h", "rz", "rx", "cz"]
                qasm_res = compile_api(self, compile_engine="qiskit", output="qasm2", options=qiskit_opts)
                source_to_submit = qasm_res["circuit"]
            else:
                source_to_submit = None

            if source_to_submit is not None:
                tasks = device_base.run(
                    provider=dev_provider,
                    device=dev_device,
                    source=source_to_submit,
                    shots=dev_shots,
                    **dev_opts,
                )
            else:
                # Submit IR directly (simulator/local)
                tasks = device_base.run(
                    provider=dev_provider,
                    device=dev_device,
                    circuit=compiled,
                    shots=dev_shots,
                    **dev_opts,
                )

        # unified_list = tasks if isinstance(tasks, list) else [tasks]
        unified_result_list=[]
        # Normalize to list of unified payloads
        if wait_async_result is False:
            for t in tasks:
                task_result = t.get_result(wait=False)
                unified_result_list.append(task_result)

        else:
            for t in tasks:
                task_result = t.get_result(wait=True)
                unified_result_list.append(task_result)

    
        # Fetch final results where needed and attach postprocessing
        from ...postprocessing import apply_postprocessing  # 延迟导入，保持解耦
        results: List[Dict[str, Any]] = []
        for rr in unified_result_list:
            if isinstance(rr, dict):
                has_payload = (rr.get('result') is not None) or (rr.get('counts') is not None) or (rr.get('expectations') is not None)
                if has_payload:
                    post = apply_postprocessing(rr, self._post_opts if isinstance(self._post_opts, dict) else {})
                    rr["postprocessing"] = post
                    results.append(rr)
                else:
                    error_result = {
                        'result':{},
                        'result_meta': rr.get('result_meta', {}),
                        'postprocessing': {
                            'method': None,
                            'result': None
                        }
                    }
                    results.append(error_result)
            else:
                raise TypeError('result is not a dict',rr)
             
        return results if isinstance(tasks, list) else results[0]

    # ---- Task helpers for cloud.api thin wrappers ----
    def get_task_details(self,task: Any, *, wait: bool = False, poll_interval: float = 2.0, timeout: float = 15.0) -> Dict[str, Any]:
        return task.get_result(task=task, wait=wait, poll_interval=poll_interval, timeout=timeout)
    
    def get_result(self, task: Any, *, wait: bool = False, poll_interval: float = 2.0, timeout: float = 15.0)-> Dict[str, Any]:
        return task.get_result(task=task, wait=wait, poll_interval=poll_interval, timeout=timeout)

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

    def reset(self, q: int):
        """Warning: reset operation is typically not supported by hardware in logical circuits.
        This is a simulation-only operation that projects qubit to |0⟩ state."""
        warnings.warn("reset operation is typically not supported by hardware in logical circuits. "
                    "This is a simulation-only operation that projects qubit to |0⟩ state.", 
                    UserWarning, stacklevel=2)
        self.ops.append(("reset", int(q)))
        return self

    def RESET(self, q: int):
        return self.reset(q)
    
    # --- Additional common gates to preserve legacy examples ---
    def x(self, q: int):
        self.ops.append(("x", int(q)))
        return self

    def X(self, q: int):
        return self.x(q)

    def y(self, q: int):
        self.ops.append(("y", int(q)))
        return self

    def Y(self, q: int):
        return self.y(q)

    def ry(self, q: int, theta: Any):
        self.ops.append(("ry", int(q), theta))
        return self

    def RY(self, q: int, theta: Any):
        return self.ry(q, theta)

    def cz(self, c: int, t: int):
        self.ops.append(("cz", int(c), int(t)))
        return self

    def CZ(self, c: int, t: int):
        return self.cz(c, t)

    def cy(self, c: int, t: int):
        self.ops.append(("cy", int(c), int(t)))
        return self

    def CY(self, c: int, t: int):
        return self.cy(c, t)

    def rxx(self, c: int, t: int, theta: Any):
        self.ops.append(("rxx", int(c), int(t), theta))
        return self

    def RXX(self, c: int, t: int, theta: Any):
        return self.rxx(c, t, theta)

    def rzz(self, c: int, t: int, theta: Any):
        self.ops.append(("rzz", int(c), int(t), theta))
        return self

    def RZZ(self, c: int, t: int, theta: Any):
        return self.rzz(c, t, theta)
    
    # --- draw() typing overloads to improve IDE/linter navigation ---
    @overload
    def draw(self, output: Literal["text"], *args: Any, **kwargs: Any) -> str: ...

    @overload
    def draw(self, output: Literal["mpl"], *args: Any, **kwargs: Any) -> Any: ...

    @overload
    def draw(self, output: Literal["latex"], *args: Any, **kwargs: Any) -> str: ...

    @overload
    def draw(self, *args: Any, **kwargs: Any) -> Any: ...

    # --- Draw via Qiskit provider: compile IR→QuantumCircuit and delegate draw ---
    def draw(self, *args: Any, **kwargs: Any) -> Any:
        """Render the circuit using Qiskit if available.

        Behavior:
        - Convert IR → Qiskit QuantumCircuit directly (no intermediate qasm2 dump),
          auto-adding measurements if none present.
        - Delegate all args/kwargs to `QuantumCircuit.draw`.
        - If Qiskit is not installed, return a minimal textual `gate_summary()` string.
        """
        try:
            from ...compiler.compile_engine.qiskit.dialect import to_qiskit  # type: ignore

            qc = to_qiskit(self, add_measures=True)
            # Resolve default output: prefer per-circuit _draw_output, else 'text'
            if "output" not in kwargs and (len(args) == 0):
                kwargs["output"] = self._draw_output or "text"
            return qc.draw(*args, **kwargs)
        except Exception:
            return str(self.gate_summary())
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

