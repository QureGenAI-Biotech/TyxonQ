from __future__ import annotations

import re
from typing import Any, Dict, List

try:
    # Optional import: only required when user chooses qiskit provider
    from qiskit import QuantumCircuit, ClassicalRegister  # type: ignore
except Exception:  # pragma: no cover - keep import optional
    QuantumCircuit = None  # type: ignore
    ClassicalRegister = None  # type: ignore

from ....core.ir import Circuit

OP_MAPPING: Dict[str, str] = {
    "h": "h",
    "rx": "rx",
    "rz": "rz",
    "cx": "cx",
}


DEFAULT_BASIS_GATES: List[str] = ["cx", "h", "rz", "rx", "cz"]
DEFAULT_OPT_LEVEL: int = 2


def normalize_transpile_options(options: Dict[str, Any] | None) -> Dict[str, Any]:
    norm: Dict[str, Any] = {}
    options = options or {}
    norm.update(options)
    if "opt_level" in norm and "optimization_level" not in norm:
        try:
            norm["optimization_level"] = int(norm.pop("opt_level"))
        except Exception:
            norm.pop("opt_level", None)
    # Fill default basis_gates when missing or empty
    if not norm.get("basis_gates"):
        norm["basis_gates"] = list(DEFAULT_BASIS_GATES)
    norm.setdefault("optimization_level", DEFAULT_OPT_LEVEL)
    return norm


def free_pi(s: str) -> str:
    rs: List[str] = []
    pistr = "3.141592653589793"
    s = s.replace("pi", pistr)
    for r in s.split("\n"):
        inc = re.search(r"\(.*\)", r)
        if inc is None:
            rs.append(r)
        else:
            v = r[inc.start() : inc.end()]
            v = eval(v)  # nosec
            if not isinstance(v, tuple):
                r = r[: inc.start()] + "(" + str(v) + ")" + r[inc.end() :]
            else:
                r = r[: inc.start()] + str(v) + r[inc.end() :]
            rs.append(r)
    return "\n".join(rs)


def comment_qasm(s: str) -> str:
    nslist: List[str] = []
    nslist.append("//circuit begins")
    for line in s.split("\n"):
        nslist.append("//" + line)
    nslist.append("//circuit ends")
    return "\n".join(nslist)


def comment_dict(d: Dict[int, int], name: str = "logical_physical_mapping") -> str:
    nslist: List[str] = []
    nslist.append(f"//{name} begins")
    for k, v in d.items():
        nslist.append("// " + str(k) + " : " + str(v))
    nslist.append(f"//{name} ends")
    return "\n".join(nslist)


def _get_positional_logical_mapping_from_qiskit(qc: Any) -> Dict[int, int]:
    i = 0
    positional_logical_mapping: Dict[int, int] = {}
    for inst in qc.data:
        # Use modern attributes if available to avoid deprecation warnings
        op = getattr(inst, "operation", None)
        qubits = getattr(inst, "qubits", None)
        if op is not None and getattr(op, "name", "") == "measure" and qubits:
            positional_logical_mapping[i] = qc.find_bit(qubits[0]).index
            i += 1
        elif isinstance(inst, (list, tuple)) and inst and getattr(inst[0], "name", "") == "measure":  # fallback
            positional_logical_mapping[i] = qc.find_bit(inst[1][0]).index
            i += 1
    return positional_logical_mapping


def _get_logical_physical_mapping_from_qiskit(qc_after: Any, qc_before: Any | None = None) -> Dict[int, int]:
    logical_physical_mapping: Dict[int, int] = {}
    for inst in qc_after.data:
        op_after = getattr(inst, "operation", None)
        qubits_after = getattr(inst, "qubits", None)
        clbits_after = getattr(inst, "clbits", None)
        is_measure_after = (getattr(op_after, "name", "") == "measure") if op_after is not None else False
        if is_measure_after or (isinstance(inst, (list, tuple)) and getattr(inst[0], "name", "") == "measure"):
            if qc_before is None:
                cbit = clbits_after[0] if clbits_after else inst[2][0]
                logical_q = qc_after.find_bit(cbit).index
            else:
                for instb in qc_before.data:
                    op_before = getattr(instb, "operation", None)
                    qubits_before = getattr(instb, "qubits", None)
                    clbits_before = getattr(instb, "clbits", None)
                    is_measure_before = (getattr(op_before, "name", "") == "measure") if op_before is not None else False
                    if is_measure_before or (isinstance(instb, (list, tuple)) and getattr(instb[0], "name", "") == "measure"):
                        c_before = clbits_before[0] if clbits_before else instb[2][0]
                        c_after = clbits_after[0] if clbits_after else inst[2][0]
                        if qc_before.find_bit(c_before).index == qc_after.find_bit(c_after).index:
                            q_before = qubits_before[0] if qubits_before else instb[1][0]
                            logical_q = qc_before.find_bit(q_before).index
                            break
            q_after = qubits_after[0] if qubits_after else inst[1][0]
            logical_physical_mapping[logical_q] = qc_after.find_bit(q_after).index
    return logical_physical_mapping


def _add_measure_all_if_none(qc: Any) -> Any:
    for inst in qc.data:
        if inst[0].name == "measure":
            break
    else:
        qc.measure_all()
    return qc


def qasm2_dumps_compat(qc: Any) -> str:
    try:
        from qiskit.qasm2 import dumps  # type: ignore

        return dumps(qc)
    except Exception:
        return qc.qasm()  # type: ignore[attr-defined]


# -------- IR <-> Qiskit adapters (provider-scoped) --------

def to_qiskit(circuit: Circuit, *, add_measures: bool = True) -> Any:
    """Convert IR `Circuit` to a Qiskit `QuantumCircuit`.

    Supports ops: h, rx(theta), rz(theta), cx, measure_z.
    """
    if QuantumCircuit is None:
        raise RuntimeError("qiskit is not available; please install qiskit to use this provider")
    qc = QuantumCircuit(circuit.num_qubits)

    measure_indices: List[int] = []
    for op in circuit.ops:
        name = op[0]
        if name == "h":
            qc.h(int(op[1]))
        elif name == "rx":
            qc.rx(float(op[2]), int(op[1]))
        elif name == "rz":
            qc.rz(float(op[2]), int(op[1]))
        elif name == "cx":
            qc.cx(int(op[1]), int(op[2]))
        elif name == "measure_z":
            measure_indices.append(int(op[1]))
        else:
            raise NotImplementedError(f"Unsupported op for qiskit adapter: {name}")

    if add_measures:
        if measure_indices:
            if ClassicalRegister is None:
                raise RuntimeError("qiskit classical register not available")
            creg = ClassicalRegister(len(measure_indices))
            qc.add_register(creg)
            for i, q in enumerate(measure_indices):
                qc.measure(q, creg[i])
        else:
            # No explicit measure ops in IR: measure all for execution outputs
            qc.measure_all()
    return qc


def from_qiskit(qc: Any) -> Circuit:
    """Convert a Qiskit `QuantumCircuit` to IR `Circuit`.

    Recognizes: h, rx(theta), rz(theta), cx, measure.
    """
    ops: List[Any] = []
    for inst in getattr(qc, "data", []):
        op = getattr(inst, "operation", None)
        qubits = getattr(inst, "qubits", None)
        params = getattr(op, "params", []) if op is not None else []
        name = getattr(op, "name", None) if op is not None else None

        # fallback older tuple format
        if op is None and isinstance(inst, (list, tuple)) and inst:
            op = inst[0]
            name = getattr(op, "name", None)
            qubits = inst[1]
            params = getattr(op, "params", [])

        if name == "h":
            ops.append(("h", int(qc.find_bit(qubits[0]).index)))
        elif name == "rx":
            theta = float(params[0]) if params else 0.0
            ops.append(("rx", int(qc.find_bit(qubits[0]).index), theta))
        elif name == "rz":
            theta = float(params[0]) if params else 0.0
            ops.append(("rz", int(qc.find_bit(qubits[0]).index), theta))
        elif name == "cx":
            c = int(qc.find_bit(qubits[0]).index)
            t = int(qc.find_bit(qubits[1]).index)
            ops.append(("cx", c, t))
        elif name == "measure":
            q = int(qc.find_bit(qubits[0]).index)
            ops.append(("measure_z", q))
        elif name == "barrier":
            # Non-unitary barrier: ignore for IR ops; layout/scheduling info not preserved here
            continue
        else:
            raise NotImplementedError(f"Unsupported qiskit op in adapter: {name}")

    return Circuit(num_qubits=qc.num_qubits, ops=ops)


# -------- OpenQASM adapters via Qiskit provider --------

def qasm_to_ir(qasm_str: str) -> Circuit:
    """Parse OpenQASM 2 string to IR `Circuit` using Qiskit if available."""
    if QuantumCircuit is None:
        raise RuntimeError("qiskit is not available; please install qiskit to use this provider")
    try:
        from qiskit.qasm2 import loads  # type: ignore

        qc = loads(qasm_str)
    except Exception:
        qc = QuantumCircuit.from_qasm_str(qasm_str)  # type: ignore[attr-defined]
    return from_qiskit(qc)


def ir_to_qasm(circuit: Circuit) -> str:
    """Serialize IR `Circuit` to OpenQASM 2 string via Qiskit."""
    qc = to_qiskit(circuit, add_measures=True)
    return qasm2_dumps_compat(qc)

