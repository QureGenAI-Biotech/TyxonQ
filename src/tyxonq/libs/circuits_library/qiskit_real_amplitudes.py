from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


def real_amplitudes_circuit_template_converter(qc: Any) -> List[Tuple]:
    """
    Convert a Qiskit RealAmplitudes (or compatible parameterized QuantumCircuit)
    into a parameterized template usable by HEA.get_circuit.

    The returned template is a list of IR-like ops where rotation angles are placeholders:
      - ("ry"|"rz"|"rx", q, ("p", param_index)) for parameterized rotations
      - ("cx"|"cz", c, t) for two-qubit entangling gates
      - ("h", q) for H gate

    Parameters
    ----------
    qc: Qiskit QuantumCircuit

    Returns
    -------
    template: List[tuple]
    """
    try:
        from qiskit import QuantumCircuit  # type: ignore
    except Exception as _:
        raise ImportError("qiskit is required to convert RealAmplitudes circuit")

    if not isinstance(qc, QuantumCircuit):
        raise TypeError("qc must be a Qiskit QuantumCircuit")

    # establish parameter index mapping (use creation/order from qc.parameters)
    param_list = list(getattr(qc, "parameters", []))
    p2idx = {p: i for i, p in enumerate(param_list)}

    template: List[Tuple] = []
    # iterate modern Qiskit instruction API
    for inst in getattr(qc, "data", []):
        op = getattr(inst, "operation", None)
        qubits = getattr(inst, "qubits", None)
        name = getattr(op, "name", None) if op is not None else None
        params_q = list(getattr(op, "params", [])) if op is not None else []

        if name in ("ry", "rz", "rx"):
            angle = params_q[0] if params_q else 0.0
            q = int(qc.find_bit(qubits[0]).index)
            if angle in p2idx:
                template.append((name, q, ("p", int(p2idx[angle]))))
            else:
                # try resolve numeric; fallback to first parameter in expression, else 0.0
                try:
                    val = float(angle)
                except Exception:
                    pars = list(getattr(angle, "parameters", []))
                    if pars and pars[0] in p2idx:
                        template.append((name, q, ("p", int(p2idx[pars[0]]))))
                        continue
                    val = 0.0
                template.append((name, q, float(val)))
        elif name in ("cx", "cz"):
            c = int(qc.find_bit(qubits[0]).index)
            t = int(qc.find_bit(qubits[1]).index)
            template.append(("cx" if name == "cx" else "cz", c, t))
        elif name in ("h",):
            q = int(qc.find_bit(qubits[0]).index)
            template.append(("h", q))
        elif name in ("barrier", "measure"):
            continue
        else:
            raise NotImplementedError(f"Unsupported qiskit op in converter: {name}")

    return template


def build_circuit_from_template(template: List[Tuple], params: np.ndarray, *, n_qubits: int) -> Any:
    """
    Instantiate our IR Circuit from a parameterized template and parameter vector.
    """
    from tyxonq.core.ir.circuit import Circuit  # lazy import

    params = np.asarray(params, dtype=np.float64)
    ops: List[Tuple] = []
    for op in template:
        name = op[0]
        if len(op) >= 3 and isinstance(op[2], tuple) and len(op[2]) == 2 and op[2][0] == "p":
            idx = int(op[2][1])
            ops.append((name, int(op[1]), float(params[idx])))
        else:
            ops.append(tuple(op))
    return Circuit(n_qubits, ops=ops)
