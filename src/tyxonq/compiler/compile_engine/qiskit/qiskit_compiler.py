from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...api import CompileResult
    from ....core.ir import Circuit
    from ....devices import DeviceRule

from .dialect import (
    normalize_transpile_options,
    qasm2_dumps_compat,
    to_qiskit as ir_to_qiskit,
    from_qiskit as qiskit_to_ir,
    _get_logical_physical_mapping_from_qiskit,
    _get_positional_logical_mapping_from_qiskit,
)


class QiskitCompiler:
    name = "qiskit"

    def compile(self, circuit: "Circuit", options:Dict[str, Any] = {},**kwargs) -> "CompileResult":  # type: ignore[override]
        output = str(options.get("output", "qiskit")).lower()
        add_measures = bool(options.get("add_measures", True))
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
            qc = ir_to_qiskit(circuit, add_measures=add_measures)
        else:
            qc = None

        # circuit already encoded into qc by adapter

        compiled_qc = qc
        if do_transpile and qc is not None:
            # Strip non-transpile options before passing to qiskit.transpile
            tp_opts = {k: v for k, v in norm_opts.items() if k not in ("output", "transpile", "add_measures", "compile_engine")}
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
            "output": "output",
            "options": dict(norm_opts),
            "device_rule": {},
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


# Convenience exports
__all__ = ["QiskitCompiler", "ir_to_qiskit", "qiskit_to_ir"]


