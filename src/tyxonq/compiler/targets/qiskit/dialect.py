from __future__ import annotations

import re
from typing import Any, Dict, List


# Minimal gate/op mapping aligned with current basis gates.
OP_MAPPING: Dict[str, str] = {
    "h": "h",
    "rz": "rz",
    "cx": "cx",
}


DEFAULT_BASIS_GATES: List[str] = ["h", "rz", "cx"]
DEFAULT_OPT_LEVEL: int = 2  # level 3 can induce issues depending on versions


def normalize_transpile_options(options: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return normalized qiskit.transpile options with TyxonQ defaults.

    - basis_gates defaults to ["h", "rz", "cx"] unless provided
    - optimization_level defaults to 2 unless provided
    Other keys from `options` are preserved.
    """

    norm: Dict[str, Any] = {}
    options = options or {}
    norm.update(options)
    norm.setdefault("basis_gates", list(DEFAULT_BASIS_GATES))
    norm.setdefault("optimization_level", DEFAULT_OPT_LEVEL)
    return norm


def free_pi(s: str) -> str:
    """Replace symbolic `pi` in OpenQASM-like argument lists with numbers.

    This mirrors historical behavior to make QASM downstream consumers robust
    against symbolic constants. Only parenthesized argument tuples are parsed
    and evaluated.
    """

    rs: List[str] = []
    pistr = "3.141592653589793"
    s = s.replace("pi", pistr)
    for r in s.split("\n"):
        inc = re.search(r"\(.*\)", r)
        if inc is None:
            rs.append(r)
        else:
            v = r[inc.start() : inc.end()]
            v = eval(v)  # nosec - maintained for compatibility with legacy behavior
            if not isinstance(v, tuple):
                r = r[: inc.start()] + "(" + str(v) + ")" + r[inc.end() :]
            else:  # u gate case
                r = r[: inc.start()] + str(v) + r[inc.end() :]
            rs.append(r)
    return "\n".join(rs)


def comment_qasm(s: str) -> str:
    """Return QASM string wrapped as line comments for embedding in logs/files."""

    nslist: List[str] = []
    nslist.append("//circuit begins")
    for line in s.split("\n"):
        nslist.append("//" + line)
    nslist.append("//circuit ends")
    return "\n".join(nslist)


def comment_dict(d: Dict[int, int], name: str = "logical_physical_mapping") -> str:
    """Serialize a mapping dictionary into a commented QASM-like block."""

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
        if inst[0].name == "measure":
            positional_logical_mapping[i] = qc.find_bit(inst[1][0]).index
            i += 1
    return positional_logical_mapping


def _get_logical_physical_mapping_from_qiskit(qc_after: Any, qc_before: Any | None = None) -> Dict[int, int]:
    logical_physical_mapping: Dict[int, int] = {}
    for inst in qc_after.data:
        if inst[0].name == "measure":
            if qc_before is None:
                logical_q = qc_after.find_bit(inst[2][0]).index
            else:
                for instb in qc_before.data:
                    if (
                        instb[0].name == "measure"
                        and qc_before.find_bit(instb[2][0]).index
                        == qc_after.find_bit(inst[2][0]).index
                    ):
                        logical_q = qc_before.find_bit(instb[1][0]).index
                        break
            logical_physical_mapping[logical_q] = qc_after.find_bit(inst[1][0]).index
    return logical_physical_mapping


def _add_measure_all_if_none(qc: Any) -> Any:
    for inst in qc.data:
        if inst[0].name == "measure":
            break
    else:
        qc.measure_all()
    return qc


def qasm2_dumps_compat(qc: Any) -> str:
    """Dump OpenQASM2 from a qiskit QuantumCircuit across versions."""

    try:
        from qiskit.qasm2 import dumps  # type: ignore

        return dumps(qc)
    except Exception:
        return qc.qasm()  # type: ignore[attr-defined]


