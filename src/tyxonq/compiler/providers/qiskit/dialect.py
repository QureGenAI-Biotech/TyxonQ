from __future__ import annotations

import re
from typing import Any, Dict, List


OP_MAPPING: Dict[str, str] = {
    "h": "h",
    "rz": "rz",
    "cx": "cx",
}


DEFAULT_BASIS_GATES: List[str] = ["h", "rz", "cx"]
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
    norm.setdefault("basis_gates", list(DEFAULT_BASIS_GATES))
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


