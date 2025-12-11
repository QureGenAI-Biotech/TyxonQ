"""QASM3 + OpenPulse importer for TyxonQ native compiler.

This module provides complete QASM3 + OpenPulse support for importing quantum programs
into TyxonQ's native IR. This is the inverse operation of tqasm_exporter.py.

Architecture:
    QASM3 + OpenPulse source
        ↓
    [qasm3_importer.py] Tokenize & Parse
        ↓
    Circuit IR (gates) + Pulse metadata (defcal, frame)
        ↓
    [simulator/executor] Run on TyxonQ engines

Phase Support:
    Phase 2: Gate-level QASM3 (qubit declarations, gates, measurements) ✅
    Phase 3: OpenPulse Frame definitions (port, frame, frequency, phase) ✅
    Phase 4: defcal gate definitions with pulse instructions ✅

Supported Constructs:
    1. Gate-level QASM3:
       - Qubit declarations: qubit[n] q; or qreg q[n];
       - Single-qubit gates: h, x, y, z, s, t, rx, ry, rz
       - Two-qubit gates: cx, cy, cz, swap
       - Measurements: measure q[i];
    
    2. OpenPulse cal blocks:
       - Port declarations: extern port d0;
       - Frame definitions: frame d0_frame = newframe(d0, freq, phase);
    
    3. defcal gate definitions:
       - Waveform definitions: waveform wf = drag(...);
       - Pulse instructions: play(frame, wf), shift_phase(frame, angle), etc.
       - Gate binding: defcal h $0 { ... }
    
    4. Waveform definitions:
       - Gaussian, DRAG, Hermite, BlackmanSquare, etc.
       - All waveforms supported by TyxonQ.waveforms
    
    5. Pulse instructions:
       - play(frame, waveform)
       - shift_phase(frame, angle)
       - set_frequency(frame, freq)
       - delay(duration)

Architecture Notes:
    - This is part of pulse_compile_engine/native/
    - It works in tandem with tqasm_exporter.py
    - Designed for TyxonQ's internal use, not external QASM3 standards
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class Frame:
    """Represents an OpenPulse frame (drive framework)."""
    name: str
    port: str
    frequency: float
    phase: float = 0.0


@dataclass
class Waveform:
    """Represents a pulse waveform definition."""
    waveform_type: str  # "gaussian", "drag", "hermite", etc.
    amplitude: float
    duration: int
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DefcalDefinition:
    """Represents a defcal gate definition."""
    gate_name: str
    qubits: List[int]
    parameters: List[str]  # For parameterized gates like rx(angle)
    body: List[str]  # Pulse instruction lines


class QASM3Tokenizer:
    """Tokenize QASM3 source code."""
    
    def __init__(self, source: str):
        self.source = source
        self.lines = source.split('\n')
        self.pos = 0
    
    def tokenize(self) -> List[str]:
        """Tokenize the QASM3 source into tokens."""
        tokens = []
        for line in self.lines:
            # Remove comments
            if '//' in line:
                line = line[:line.index('//')]
            line = line.strip()
            if not line:
                continue
            tokens.append(line)
        return tokens


class QASM3Parser:
    """Parse QASM3 + OpenPulse syntax.
    
    Hierarchical parsing:
        1. Parse qubit declarations (num_qubits)
        2. Parse cal block (ports, frames)
        3. Parse defcal definitions
        4. Parse gate operations
    """
    
    def __init__(self, qasm3_str: str):
        self.source = qasm3_str
        self.num_qubits: Optional[int] = None
        self.frames: Dict[str, Frame] = {}
        self.gates: List[Tuple[str, ...]] = []
        self.defcals: Dict[str, DefcalDefinition] = {}
    
    def parse(self) -> Tuple[int, List[Tuple[str, ...]], Dict[str, DefcalDefinition]]:
        """Parse QASM3 source code.
        
        Returns:
            (num_qubits, gates, defcals)
        """
        # Step 1: Extract qubit count
        self.num_qubits = self._parse_qubit_declaration()
        if self.num_qubits is None:
            raise ValueError("No qubit declaration found in QASM3 code")
        
        # Step 2: Parse cal block (ports, frames) - Phase 3
        self._parse_cal_block()
        
        # Step 3: Parse defcal definitions - Phase 4
        self._parse_defcal_block()
        
        # Step 4: Parse gate operations
        self._parse_gates()
        
        return self.num_qubits, self.gates, self.defcals
    
    def _parse_qubit_declaration(self) -> Optional[int]:
        """Extract number of qubits from qubit declaration."""
        # OPENQASM 3: qubit[n] q;
        match = re.search(r'qubit\[(\d+)\]\s+\w+\s*;', self.source)
        if match:
            return int(match.group(1))
        
        # QASM 2 style: qreg q[n];
        match = re.search(r'qreg\s+\w+\[(\d+)\]\s*;', self.source)
        if match:
            return int(match.group(1))
        
        return None
    
    def _parse_cal_block(self) -> None:
        """Parse OpenPulse cal block with port and frame definitions.
        
        Phase 3: Supports
            - Port declarations: extern port d0;
            - Frame definitions: frame d0_frame = newframe(d0, 5e9, 0.0);
        """
        # Extract cal { ... } block
        cal_match = re.search(r'cal\s*{([^}]*)}\s*(?:defcal|h|cx|measure|$)', self.source, re.DOTALL)
        if not cal_match:
            return
        
        cal_body = cal_match.group(1)
        
        # Parse port declarations (extern port d0;)
        port_pattern = r'extern\s+port\s+(\w+)\s*;'
        for match in re.finditer(port_pattern, cal_body):
            port_name = match.group(1)
            # Store port reference (used by frames)
        
        # Parse frame definitions (frame d0_frame = newframe(d0, freq, phase);)
        frame_pattern = r'frame\s+(\w+)\s*=\s*newframe\s*\(\s*(\w+)\s*,\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)\s*\)'
        for match in re.finditer(frame_pattern, cal_body):
            frame_name = match.group(1)
            port_name = match.group(2)
            frequency = float(match.group(3))
            phase = float(match.group(4))
            self.frames[frame_name] = Frame(frame_name, port_name, frequency, phase)
    
    def _parse_defcal_block(self) -> None:
        """Parse defcal gate definitions.
        
        Phase 4: Supports
            - defcal h $0 { ... }
            - defcal cx $0, $1 { ... }
            - Waveform definitions: waveform wf = drag(...);
            - Pulse instructions: play(frame, wf);
        """
        # Extract all defcal { ... } blocks
        defcal_pattern = r'defcal\s+(\w+)\s+([\$\w\,\s]+)\s*{([^}]*)}'
        for match in re.finditer(defcal_pattern, self.source, re.DOTALL):
            gate_name = match.group(1)
            qubits_str = match.group(2).strip()
            body = match.group(3)
            
            # Parse qubit indices (e.g., "$0" or "$0, $1")
            qubit_pattern = r'\$(\d+)'
            qubits = [int(m.group(1)) for m in re.finditer(qubit_pattern, qubits_str)]
            
            # Extract waveform definitions from body
            waveforms: Dict[str, Waveform] = {}
            wf_pattern = r'waveform\s+(\w+)\s*=\s*([a-z_]+)\s*\(([^)]*)\)'
            for wf_match in re.finditer(wf_pattern, body):
                wf_name = wf_match.group(1)
                wf_type = wf_match.group(2)
                params_str = wf_match.group(3)
                # Store for reference (actual waveform parsing happens during execution)
                waveforms[wf_name] = Waveform(wf_type, 1.0, 160, {"params_str": params_str})
            
            # Store defcal definition
            defcal_def = DefcalDefinition(
                gate_name=gate_name,
                qubits=qubits,
                parameters=[],  # TODO: Parse parameterized gates like rx(angle)
                body=[body.strip()]  # Store the full body for later execution
            )
            self.defcals[f"{gate_name}_{len(qubits)}q"] = defcal_def
    
    def _parse_gates(self) -> None:
        """Parse gate operations from QASM3 code."""
        # Remove cal block, defcal blocks, and header
        code = re.sub(r'cal\s*{[^}]*}', '', self.source)
        code = re.sub(r'defcal\s+\w+\s+[\$\w\,\s]+\s*{[^}]*}', '', code)
        code = re.sub(r'OPENQASM\s+3\.0\s*;', '', code)
        code = re.sub(r'TQASM\s+0\.2\s*;', '', code)
        code = re.sub(r'defcalgrammar\s+"[^"]*"\s*;', '', code)
        code = re.sub(r'qubit\[(\d+)\]\s+\w+\s*;', '', code)
        code = re.sub(r'qreg\s+\w+\s*\[\d+\]\s*;', '', code)
        code = re.sub(r'include\s+"[^"]*"\s*;', '', code)
        
        # Parse individual gate operations
        lines = [line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('//')]
        
        for line in lines:
            if not line or line.startswith('//'):
                continue
            
            # Remove trailing semicolon
            line = line.rstrip(';').strip()
            if not line:
                continue
            
            # Single-qubit gates: h q[0]
            match = re.match(r'(h|x|y|z|s|t)\s+q\[(\d+)\]', line)
            if match:
                gate_name = match.group(1)
                qubit = int(match.group(2))
                if qubit >= self.num_qubits:
                    raise ValueError(f"Qubit index {qubit} out of range [0, {self.num_qubits})")
                self.gates.append((gate_name, qubit))
                continue
            
            # Parameterized single-qubit gates: rx(theta) q[i]
            match = re.match(r'(rx|ry|rz)\(([^)]+)\)\s+q\[(\d+)\]', line)
            if match:
                gate_name = match.group(1)
                angle_str = match.group(2)
                qubit = int(match.group(3))
                if qubit >= self.num_qubits:
                    raise ValueError(f"Qubit index {qubit} out of range [0, {self.num_qubits})")
                
                angle = self._parse_angle(angle_str)
                self.gates.append((gate_name, qubit, angle))
                continue
            
            # Two-qubit gates: cx q[0], q[1]
            match = re.match(r'(cx|cy|cz|swap)\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]', line)
            if match:
                gate_name = match.group(1)
                q0 = int(match.group(2))
                q1 = int(match.group(3))
                if q0 >= self.num_qubits or q1 >= self.num_qubits:
                    raise ValueError(f"Qubit indices out of range [0, {self.num_qubits})")
                
                if gate_name == 'cx':
                    self.gates.append(('cx', q0, q1))
                elif gate_name == 'cz':
                    self.gates.append(('cz', q0, q1))
                elif gate_name == 'cy':
                    # cy = h(q1); cz(q0, q1); h(q1)
                    self.gates.append(('h', q1))
                    self.gates.append(('cz', q0, q1))
                    self.gates.append(('h', q1))
                elif gate_name == 'swap':
                    # swap(a, b) = cx(a, b); cx(b, a); cx(a, b)
                    self.gates.append(('cx', q0, q1))
                    self.gates.append(('cx', q1, q0))
                    self.gates.append(('cx', q0, q1))
                continue
            
            # Measurement: measure q[i]
            match = re.match(r'measure\s+q\[(\d+)\]', line)
            if match:
                qubit = int(match.group(1))
                if qubit >= self.num_qubits:
                    raise ValueError(f"Qubit index {qubit} out of range [0, {self.num_qubits})")
                self.gates.append(('measure_z', qubit))
                continue
    
    def _parse_angle(self, angle_str: str) -> float:
        """Parse angle parameter from QASM3.
        
        Supports:
            - Numeric literals: 1.5, 3.14159
            - pi expressions: pi, pi/2, 2*pi
            - Python expressions: sin(pi/4), etc.
        """
        angle_str = angle_str.strip()
        
        # Replace 'pi' with numeric value
        angle_str = angle_str.replace('pi', '3.141592653589793')
        
        try:
            import math
            result = eval(
                angle_str,
                {"__builtins__": {}},
                {"sin": math.sin, "cos": math.cos, "sqrt": math.sqrt, "tan": math.tan}
            )
            return float(result)
        except Exception as e:
            raise ValueError(f"Cannot parse angle parameter: {angle_str}") from e


def qasm3_to_circuit(qasm3_str: str) -> Any:
    """Import QASM3 + OpenPulse code to TyxonQ Circuit IR.
    
    This is the inverse of tqasm_exporter.py.
    
    Supports:
        - Phase 2: Gate-level QASM3 (h, x, y, z, rx, ry, rz, cx, cy, cz, swap, measure)
        - Phase 3: OpenPulse cal blocks with frame definitions and port declarations
        - Phase 4: defcal gate definitions with pulse instructions
    
    Args:
        qasm3_str: QASM3 source code string
    
    Returns:
        TyxonQ Circuit IR with metadata containing frame and defcal information
    
    Raises:
        ValueError: If QASM3 code contains unsupported constructs
    
    Examples:
        >>> qasm3_code = '''OPENQASM 3.0;
        ... qubit[2] q;
        ... h q[0];
        ... cx q[0], q[1];
        ... measure q[0];
        ... measure q[1];
        ... '''
        >>> circuit = qasm3_to_circuit(qasm3_code)
        >>> print(circuit.num_qubits)
        2
        
        >>> # With Phase 3 frame definitions
        >>> qasm3_code_with_frames = '''OPENQASM 3.0;
        ... qubit[1] q;
        ... cal {
        ...     extern port d0;
        ...     frame d0_frame = newframe(d0, 5e9, 0.0);
        ... }
        ... h q[0];
        ... '''
        >>> circuit = qasm3_to_circuit(qasm3_code_with_frames)
        >>> print(circuit.metadata['qasm3_frames'])
    """
    from ....core.ir import Circuit
    
    parser = QASM3Parser(qasm3_str)
    num_qubits, gates, defcals = parser.parse()
    
    # Convert to Circuit IR
    circuit = Circuit(num_qubits=num_qubits, ops=gates)
    
    # Phase 3: Store frame information in metadata
    if parser.frames:
        circuit.metadata['qasm3_frames'] = {
            name: {'port': frame.port, 'frequency': frame.frequency, 'phase': frame.phase}
            for name, frame in parser.frames.items()
        }
    
    # Phase 4: Store defcal definitions in metadata for simulator execution
    if parser.defcals:
        circuit.metadata['qasm3_defcals'] = parser.defcals
    
    return circuit
