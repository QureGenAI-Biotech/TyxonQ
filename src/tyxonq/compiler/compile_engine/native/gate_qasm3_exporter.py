"""Gate-level QASM3 exporter for TyxonQ Circuit IR.

This module exports gate-level circuits (without pulse operations) to 
OpenQASM 3.0 format. This is used for gate-level compilation output
when user specifies output="qasm3" or output="openqasm3".

Unlike tqasm_exporter.py (which handles pulse circuits), this exporter
focuses solely on gate-level translation, producing clean and simple
OpenQASM 3.0 code.
"""

from typing import Any, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from tyxonq.core.ir import Circuit


class GateQASM3Exporter:
    """Export gate-level TyxonQ Circuit to OpenQASM 3.0 format."""
    
    def __init__(self):
        """Initialize the exporter."""
        pass
    
    def export(self, circuit: "Circuit") -> str:
        """Export circuit to OpenQASM 3.0 format.
        
        Args:
            circuit: TyxonQ Circuit IR
        
        Returns:
            OpenQASM 3.0 code string
        
        Raises:
            ValueError: If circuit contains unsupported operations
        """
        lines = []
        
        # 1. Header
        lines.append("OPENQASM 3.0;")
        lines.append("")
        
        # 2. Qubit declaration (standard OpenQASM 3.0 style)
        lines.append(f"qubit[{circuit.num_qubits}] q;")
        lines.append("")
        
        # 3. Gate operations
        for op in circuit.ops:
            gate_line = self._export_operation(op)
            if gate_line:
                lines.append(gate_line)
        
        # 4. Final newline
        lines.append("")
        
        return "\n".join(lines)
    
    def _export_operation(self, op: Tuple) -> str:
        """Export a single operation to QASM3 format.
        
        Args:
            op: Operation tuple from Circuit.ops
        
        Returns:
            QASM3 code line for this operation, or empty string if skipped
        
        Raises:
            ValueError: If operation is unsupported
        """
        if not isinstance(op, (list, tuple)) or len(op) == 0:
            return ""
        
        op_name = str(op[0]).lower()
        
        # Skip pulse operations (shouldn't appear in gate-level circuit)
        if op_name in ("pulse", "pulse_inline", "play", "set_phase", "shift_phase", "set_frequency"):
            return ""
        
        # Single-qubit gates without parameters
        if op_name in ("h", "x", "y", "z", "s", "t", "sdg", "tdg"):
            if len(op) < 2:
                raise ValueError(f"Gate {op_name} requires at least one qubit")
            qubit = int(op[1])
            return f"{op_name} q[{qubit}];"
        
        # Single-qubit gates with one parameter
        elif op_name in ("rx", "ry", "rz", "p", "u1"):
            if len(op) < 3:
                raise ValueError(f"Gate {op_name} requires one qubit and one parameter")
            qubit = int(op[1])
            angle = float(op[2])
            return f"{op_name}({angle}) q[{qubit}];"
        
        # Single-qubit gate with three parameters
        elif op_name in ("u3", "u"):
            if len(op) < 5:
                raise ValueError(f"Gate {op_name} requires one qubit and three parameters")
            qubit = int(op[1])
            theta = float(op[2])
            phi = float(op[3])
            lam = float(op[4])
            return f"u({theta}, {phi}, {lam}) q[{qubit}];"
        
        # Two-qubit gates
        elif op_name in ("cx", "cnot", "cz", "cy", "ch", "swap", "iswap"):
            if len(op) < 3:
                raise ValueError(f"Gate {op_name} requires two qubits")
            q0 = int(op[1])
            q1 = int(op[2])
            return f"{op_name} q[{q0}], q[{q1}];"
        
        # Three-qubit gates
        elif op_name in ("ccx", "toffoli", "cswap", "fredkin"):
            if len(op) < 4:
                raise ValueError(f"Gate {op_name} requires three qubits")
            q0 = int(op[1])
            q1 = int(op[2])
            q2 = int(op[3])
            return f"{op_name} q[{q0}], q[{q1}], q[{q2}];"
        
        # Measurement
        elif op_name in ("measure", "measure_z"):
            if len(op) < 2:
                raise ValueError(f"Gate {op_name} requires one qubit")
            qubit = int(op[1])
            return f"measure q[{qubit}];"
        
        # Barrier
        elif op_name == "barrier":
            return "barrier;"
        
        # Unsupported
        else:
            raise ValueError(f"Unsupported operation: {op_name}")
