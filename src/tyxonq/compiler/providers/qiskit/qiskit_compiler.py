from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...api import CompileRequest, CompileResult
    from ...api import Compiler  # Protocol
    from ....core.ir import Circuit
    from ....devices import DeviceCapabilities

from .dialect import (
    normalize_transpile_options,
    qasm2_dumps_compat,
    _get_logical_physical_mapping_from_qiskit,
    _get_positional_logical_mapping_from_qiskit,
)


class QiskitCompiler:
    name = "qiskit"

    def compile(self, request: "CompileRequest") -> "CompileResult":  # type: ignore[override]
        circuit: "Circuit" = request["circuit"]
        target: "DeviceCapabilities" = request.get("target", {})  # type: ignore[assignment]
        options: Dict[str, Any] = request.get("options", {})
        output = str(options.get("output", "qiskit")).lower()
        do_transpile = bool(options.get("transpile", True))
        norm_opts = normalize_transpile_options(options)

        if output in ("qiskit", "qasm", "qasm2") or do_transpile:
            try:
                from qiskit import QuantumCircuit, ClassicalRegister
                from qiskit.compiler import transpile as qk_transpile
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"qiskit not available: {exc}")
        else:
            QuantumCircuit = None  # type: ignore
            ClassicalRegister = None  # type: ignore

        if QuantumCircuit is not None:
            qc = QuantumCircuit(circuit.num_qubits)
        else:
            qc = None

        measure_indices = []
        for op in circuit.ops:
            name = op[0]
            if name == "h" and qc is not None:
                qc.h(int(op[1]))
            elif name == "rz" and qc is not None:
                qc.rz(float(op[2]), int(op[1]))
            elif name == "cx" and qc is not None:
                qc.cx(int(op[1]), int(op[2]))
            elif name == "measure_z":
                measure_indices.append(int(op[1]))
            else:
                raise NotImplementedError(f"Unsupported op for qiskit target: {name}")

        if measure_indices and qc is not None:
            creg = ClassicalRegister(len(measure_indices))
            qc.add_register(creg)
            for i, q in enumerate(measure_indices):
                qc.measure(q, creg[i])

        compiled_qc = qc
        if do_transpile and qc is not None:
            tp_opts = {k: v for k, v in norm_opts.items() if k not in ("output", "transpile")}
            compiled_qc = qk_transpile(qc, **tp_opts)

        try:
            if qc is not None and compiled_qc is not None:
                lpm = _get_logical_physical_mapping_from_qiskit(compiled_qc, qc)
                plm = _get_positional_logical_mapping_from_qiskit(qc)
            else:
                lpm = {}
                plm = {}
        except Exception:
            lpm = {}
            plm = {}

        metadata: Dict[str, Any] = {
            "target": "qiskit",
            "options": dict(norm_opts),
            "caps": dict(target),
            "logical_physical_mapping": lpm,
            "positional_logical_mapping": plm,
        }

        if output == "qiskit":
            return {"circuit": compiled_qc, "metadata": metadata}
        if output in ("qasm", "qasm2"):
            return {"circuit": qasm2_dumps_compat(compiled_qc), "metadata": metadata}
        if output == "ir":
            return {"circuit": circuit, "metadata": metadata}
        return {"circuit": compiled_qc, "metadata": metadata}


