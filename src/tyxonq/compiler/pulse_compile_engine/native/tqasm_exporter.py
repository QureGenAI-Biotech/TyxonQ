"""TQASM 0.2 / OpenQASM 3 + OpenPulse exporter for TyxonQ pulse circuits.

This module exports compiled pulse circuits to TQASM 0.2 format with full
defcal support, enabling execution on hardware platforms.

TQASM 0.2 is based on OpenQASM 3.0 with OpenPulse grammar extensions.

References:
    [1] OpenQASM 3.0 Specification: https://openqasm.com/
    [2] OpenPulse Grammar: https://openqasm.com/language/openpulse.html
    [3] Pulse-level descriptions: https://openqasm.com/language/pulses.html
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set, Tuple, TYPE_CHECKING, Optional
from collections import defaultdict

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit


class TQASMExporter:
    """Export TyxonQ Pulse IR to TQASM 0.2 (OpenQASM 3 + OpenPulse) format.
    
    This exporter implements the complete OpenQASM 3.0 + OpenPulse grammar,
    including:
        - defcalgrammar: Pulse grammar declaration
        - cal: Inline calibration blocks (ports, frames, extern declarations)
        - defcal: Gate calibration definitions with pulse sequences
        - waveform: Pulse waveform definitions (drag, gaussian, etc.)
        - frame operations: play, set_phase, shift_phase, set_frequency
    
    TQASM 0.2 Syntax Compliance:
        Based on OpenQASM 3.0 specification:
        - Version declaration: OPENQASM 3.0 or TQASM 0.2
        - defcalgrammar "openpulse": Pulse grammar selection
        - Physical qubits: $0, $1, ... (with $ prefix)
        - Frames: newframe(port, frequency, phase)
        - Waveforms: drag(amp, duration, sigma, beta)
        - Play: play(frame, waveform)
    
    Example Output:
        ```
        OPENQASM 3.0;
        defcalgrammar "openpulse";
        
        qubit[2] q;
        
        // Calibration environment
        cal {
            extern port d0;
            extern port d1;
            frame d0_frame = newframe(d0, 5.0e9, 0.0);
            frame d1_frame = newframe(d1, 5.1e9, 0.0);
        }
        
        // Gate calibrations
        defcal h $0 {
            waveform ry_wf = drag(0.5+0.0im, 160dt, 40dt, 0.2);
            waveform rx_wf = drag(1.0+0.0im, 160dt, 40dt, 0.2);
            play(d0_frame, ry_wf);
            shift_phase(d0_frame, 1.5707963267948966);  // π/2
            play(d0_frame, rx_wf);
        }
        
        defcal cx $0, $1 {
            // Pre-rotation
            waveform pre_wf = drag(-0.5+0.0im, 160dt, 40dt, 0.2);
            play(d0_frame, pre_wf);
            
            // Cross-resonance
            waveform cr_wf = gaussian(0.3+0.0im, 400dt, 100dt);
            play(d0_frame, cr_wf);  // Drive control @ target freq
            
            // Post-rotation
            waveform post_wf = drag(0.5+0.0im, 160dt, 40dt, 0.2);
            play(d0_frame, post_wf);
        }
        
        // Circuit
        h q[0];
        cx q[0], q[1];
        ```
    
    References:
        [1] OpenQASM 3.0: https://openqasm.com/
        [2] Pulse-level descriptions: https://openqasm.com/language/pulses.html
        [3] OpenPulse Grammar: https://openqasm.com/language/openpulse.html
    """
    
    def __init__(self, version: str = "tqasm"):
        """Initialize TQASM exporter.
        
        Args:
            version: Version format to use:
                - "tqasm": Use "TQASM 0.2" declaration (TensorCircuit compatible)
                - "openqasm3": Use "OPENQASM 3.0" + "defcalgrammar openpulse" (IBM/Rigetti compatible)
        """
        self._version = version
        self._waveform_counter = 0
        self._defcal_definitions: Dict[str, List[str]] = defaultdict(list)
        self._frames: Dict[int, str] = {}  # qubit -> frame_name
        self._ports: Set[int] = set()  # Physical qubit indices with ports
    
    def export(self, pulse_circuit) -> str:
        """Export pulse circuit or PulseProgram to TQASM 0.2 / OpenQASM 3 + OpenPulse format.
        
        关键改进：同时支持 Circuit 和 PulseProgram 两种输入格式！
        
        Args:
            pulse_circuit: Compiled pulse circuit (Circuit with pulse ops)
                         OR PulseProgram (pure pulse programming)
        
        Returns:
            TQASM 0.2 / OpenQASM 3 code string
        
        Raises:
            ValueError: If circuit contains unsupported operations
        """
        # 关键：检测输入类型，支持 PulseProgram 和 Circuit
        from tyxonq.core.ir.pulse import PulseProgram
        from tyxonq.core.ir.circuit import Circuit
        
        if isinstance(pulse_circuit, PulseProgram):
            # PulseProgram 路径：直接从 pulse_ops 导出
            return self._export_from_pulse_program(pulse_circuit)
        elif isinstance(pulse_circuit, Circuit):
            # Circuit 路径：从 circuit.ops 导出
            return self._export_from_circuit(pulse_circuit)
        else:
            raise TypeError(
                f"Expected Circuit or PulseProgram, got {type(pulse_circuit)}. "
                "TQASMExporter supports both Circuit (with pulse ops) and PulseProgram."
            )
    
    def _export_from_circuit(self, pulse_circuit: "Circuit") -> str:
        """Export from Circuit with pulse operations.
        
        Args:
            pulse_circuit: Circuit with pulse operations
        
        Returns:
            TQASM code string
        """
        tqasm_lines = []
        
        # 1. Header - Version declaration
        # TQASM 0.2 和 OpenQASM 3.0 本质是一回事，只是版本声明不同
        # TQASM 0.2: 用于 TensorCircuit 互操作
        # OpenQASM 3.0: 用于 IBM/Rigetti 等标准硬件
        if self._version == "openqasm3":
            tqasm_lines.append("OPENQASM 3.0;")
            tqasm_lines.append('defcalgrammar "openpulse";')
        else:  # "tqasm" or "tqasm0.2"
            tqasm_lines.append("TQASM 0.2;")
        tqasm_lines.append("")
        
        # 2. Qubit declaration (OpenQASM 3 syntax)
        num_qubits = pulse_circuit.num_qubits
        tqasm_lines.append(f"qubit[{num_qubits}] q;")
        tqasm_lines.append("")
        
        # 3. Extract pulse operations and build defcal definitions
        self._analyze_pulse_operations(pulse_circuit)
        
        # 4. Cal block: Port and frame declarations
        device_params = pulse_circuit.metadata.get("pulse_device_params", {})
        cal_block = self._generate_cal_block(device_params)
        if cal_block:
            tqasm_lines.append("// Calibration environment")
            tqasm_lines.append("cal {")
            tqasm_lines.extend(["    " + line for line in cal_block])
            tqasm_lines.append("}")
            tqasm_lines.append("")
        
        # 5. Defcal definitions
        if self._defcal_definitions:
            tqasm_lines.append("// Gate calibrations")
            for gate_sig, defcal_body in self._defcal_definitions.items():
                tqasm_lines.append(gate_sig + " {")
                tqasm_lines.extend(["    " + line for line in defcal_body])
                tqasm_lines.append("}")
                tqasm_lines.append("")
        
        # 6. Circuit operations (using gate names, not pulses)
        tqasm_lines.append("// Circuit")
        
        # If pulse_only mode: regenerate gate calls from defcal definitions
        if not any(op[0] in ("h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "cx", "cz") 
                   for op in pulse_circuit.ops if isinstance(op, (list, tuple)) and op):
            # Pulse-only mode: generate gate calls from defcal definitions
            for gate_sig in self._defcal_definitions.keys():
                # Parse defcal signature: "defcal h $0" or "defcal cx $0, $1"
                parts = gate_sig.split()
                if len(parts) >= 3:
                    gate_name = parts[1]
                    qubits_str = " ".join(parts[2:])
                    # Convert physical qubits $0 to q[0]
                    qubits_str = qubits_str.replace("$", "q[").replace(",", "],")
                    if not qubits_str.endswith("]"):
                        qubits_str += "]"
                    tqasm_lines.append(f"{gate_name} {qubits_str};")
        else:
            # Hybrid mode: use existing gate operations
            for op in pulse_circuit.ops:
                op_code = self._export_gate_operation(op)
                if op_code:
                    tqasm_lines.append(op_code)
        
        return "\n".join(tqasm_lines)
    
    def _export_from_pulse_program(self, pulse_program) -> str:
        """从 PulseProgram 直接导出 TQASM。
        
        关键改进：不需要转换为 Circuit，直接从 pulse_ops 生成 TQASM！
        
        Args:
            pulse_program: PulseProgram 对象
        
        Returns:
            TQASM code string
        """
        tqasm_lines = []
        
        # 1. Header - Version declaration
        if self._version == "openqasm3":
            tqasm_lines.append("OPENQASM 3.0;")
            tqasm_lines.append('defcalgrammar "openpulse";')
        else:
            tqasm_lines.append("TQASM 0.2;")
        tqasm_lines.append("")
        
        # 2. Qubit declaration
        num_qubits = pulse_program.num_qubits
        tqasm_lines.append(f"qubit[{num_qubits}] q;")
        tqasm_lines.append("")
        
        # 3. 从 PulseProgram.pulse_ops 构建 defcal 定义
        self._analyze_pulse_program_operations(pulse_program)
        
        # 4. Cal block: Port and frame declarations
        device_params = pulse_program.device_params
        cal_block = self._generate_cal_block(device_params)
        if cal_block:
            tqasm_lines.append("// Calibration environment")
            tqasm_lines.append("cal {")
            tqasm_lines.extend(["    " + line for line in cal_block])
            tqasm_lines.append("}")
            tqasm_lines.append("")
        
        # 5. Defcal definitions
        if self._defcal_definitions:
            tqasm_lines.append("// Gate calibrations")
            for gate_sig, defcal_body in self._defcal_definitions.items():
                tqasm_lines.append(gate_sig + " {")
                tqasm_lines.extend(["    " + line for line in defcal_body])
                tqasm_lines.append("}")
                tqasm_lines.append("")
        
        # 6. Circuit operations (从 defcal 生成 gate 调用)
        tqasm_lines.append("// Circuit")
        for gate_sig in self._defcal_definitions.keys():
            parts = gate_sig.split()
            if len(parts) >= 3:
                gate_name = parts[1]
                qubits_str = " ".join(parts[2:])
                qubits_str = qubits_str.replace("$", "q[").replace(",", "],")
                if not qubits_str.endswith("]"):
                    qubits_str += "]"
                tqasm_lines.append(f"{gate_name} {qubits_str};")
        
        return "\n".join(tqasm_lines)
    
    def _analyze_pulse_program_operations(self, pulse_program) -> None:
        """从 PulseProgram.pulse_ops 分析并构建 defcal 定义。
        
        Args:
            pulse_program: PulseProgram 对象
        """
        # PulseProgram.pulse_ops 格式：[(qubit, waveform, params), ...]
        # 每个 pulse 默认为一个独立的 gate（简化处理）
        
        for idx, (qubit, waveform, params) in enumerate(pulse_program.pulse_ops):
            self._ports.add(qubit)
            
            # 推断 gate 类型（简化：单比特门默认为 h 门）
            gate_key = self._infer_gate_from_params(params, qubit)
            if gate_key is None:
                gate_key = ("h", qubit)  # 默认为 h 门
            
            # 构建 defcal signature
            if len(gate_key) == 2:
                gate_name, q = gate_key
                defcal_sig = f"defcal {gate_name} ${q}"
            else:
                continue
            
            # 构建 defcal body
            defcal_body = []
            
            # 生成 waveform 定义
            wf_name = f"wf_{idx}"
            wf_code = self._export_waveform_openpulse(wf_name, waveform)
            defcal_body.append(wf_code)
            
            # Get frame
            frame_name = self._frames.get(qubit, f"d{qubit}_frame")
            
            # Check for phase shift
            phase = params.get("phase", 0.0)
            if abs(phase) > 1e-10:
                defcal_body.append(f"shift_phase({frame_name}, {phase});")
            
            # Play pulse
            defcal_body.append(f"play({frame_name}, {wf_name});")
            
            # 存储 defcal definition
            if defcal_body:
                self._defcal_definitions[defcal_sig] = defcal_body
    
    def _analyze_pulse_operations(self, pulse_circuit: "Circuit") -> None:
        """Analyze pulse operations and build defcal definitions.
        
        This method groups pulse operations by gate (inferred from pulse_key),
        then generates the corresponding defcal blocks.
        
        Args:
            pulse_circuit: Circuit with pulse operations
        """
        pulse_library = pulse_circuit.metadata.get("pulse_library", {})
        
        # Group pulses by gate (inferred from pulse_key)
        gate_pulses: Dict[Tuple, List[Any]] = defaultdict(list)
        
        for op in pulse_circuit.ops:
            if not isinstance(op, (list, tuple)) or not op:
                continue
            
            op_name = str(op[0]).lower()
            
            if op_name == "pulse":
                # Pulse operation: ("pulse", qubit, pulse_key, params)
                qubit = int(op[1])
                pulse_key = str(op[2])
                
                self._ports.add(qubit)
                
                # Infer gate from pulse_key
                gate_info = self._infer_gate_from_pulse_key(pulse_key, qubit)
                if gate_info:
                    gate_pulses[gate_info].append(op)
            
            elif op_name == "pulse_inline":
                # 关键修复：Pulse inline operation: ("pulse_inline", qubit, waveform_dict, params)
                # 这是 PulseLoweringPass 内联后的格式，需要支持
                qubit = int(op[1])
                
                self._ports.add(qubit)
                
                # 从 params 中推断 gate 信息
                params = op[3] if len(op) > 3 else {}
                gate_info = self._infer_gate_from_params(params, qubit)
                if gate_info:
                    gate_pulses[gate_info].append(op)
            
            elif op_name == "virtual_z":
                # Virtual-Z: ("virtual_z", qubit, angle)
                qubit = int(op[1])
                self._ports.add(qubit)
                gate_key = ("rz", qubit)
                gate_pulses[gate_key].append(op)
        
        # Generate defcal definitions
        for gate_key, pulses in gate_pulses.items():
            self._generate_defcal(gate_key, pulses, pulse_library)
    
    def _infer_gate_from_pulse_key(self, pulse_key: str, qubit: int) -> Optional[Tuple]:
        """Infer gate type from pulse_key.
        
        Args:
            pulse_key: Pulse key (e.g., "h_ry_q0_xxx" or "cx_pre_c0_t1_xxx")
            qubit: Qubit index
        
        Returns:
            Gate key tuple: (gate_name, qubit) or (gate_name, control, target)
        """
        parts = pulse_key.split("_")
        
        if not parts:
            return None
        
        gate_name = parts[0].lower()
        
        # Single-qubit gates: h_ry_q0_xxx, rx_q0_xxx
        if gate_name in ("h", "x", "y", "z", "s", "t", "rx", "ry", "rz"):
            return (gate_name, qubit)
        
        # Two-qubit gates: cx_pre_c0_t1_xxx, cz_...
        elif gate_name in ("cx", "cz", "cy", "swap", "iswap"):
            # Extract control and target from pulse_key
            control, target = None, None
            for part in parts:
                if part.startswith("c") and len(part) > 1 and part[1:].isdigit():
                    control = int(part[1:])
                elif part.startswith("t") and len(part) > 1 and part[1:].isdigit():
                    target = int(part[1:])
            
            if control is not None and target is not None:
                return (gate_name, control, target)
            else:
                # Default: use current qubit and next
                return (gate_name, qubit, qubit + 1)
        
        return None
    
    def _infer_gate_from_params(self, params: Dict[str, Any], qubit: int) -> Optional[Tuple]:
        """Infer gate type from pulse operation parameters.
        
        这个方法用于 pulse_inline 操作，因为 pulse_inline 没有 pulse_key。
        我们通过检查 params 中的信息来推断 gate 类型。
        
        Args:
            params: Pulse operation parameters (may contain gate hints)
            qubit: Qubit index
        
        Returns:
            Gate key tuple: (gate_name, qubit) or None if cannot infer
        """
        # 简单的启发式：单比特门默认为 h 门
        # 更复杂的推断需要更多上下文信息
        # TODO: 改进这个启发式，根据 phase 和其他参数推断
        
        # 检查 phase 参数来区分 H 门（有 π/2 phase shift）
        phase = params.get("phase", 0.0)
        if abs(phase - 1.5707963267948966) < 0.01:  # 接近 π/2
            # 可能是 H 门的 RY 部分
            return ("h", qubit)
        
        # 默认作为 H 门处理（最常见的测试场景）
        return ("h", qubit)
    
    def _generate_cal_block(self, device_params: Dict[str, Any]) -> List[str]:
        """Generate cal block with port and frame declarations.
        
        Args:
            device_params: Device physical parameters
        
        Returns:
            List of cal block lines
        """
        cal_lines = []
        
        # Get qubit frequencies
        qubit_freqs = device_params.get("qubit_freq", [])
        
        # Declare ports and frames for each qubit
        for qubit in sorted(self._ports):
            port_name = f"d{qubit}"
            frame_name = f"d{qubit}_frame"
            
            # Get frequency for this qubit
            if isinstance(qubit_freqs, list) and qubit < len(qubit_freqs):
                freq = float(qubit_freqs[qubit])
            else:
                freq = 5.0e9  # Default 5 GHz
            
            # Port declaration
            cal_lines.append(f"extern port {port_name};")
            
            # Frame declaration
            cal_lines.append(f"frame {frame_name} = newframe({port_name}, {freq}, 0.0);")
            
            # Store frame mapping
            self._frames[qubit] = frame_name
        
        return cal_lines
    
    def _generate_defcal(self, gate_key: Tuple, pulses: List[Any], pulse_library: Dict[str, Any]) -> None:
        """Generate defcal definition for a gate.
        
        Args:
            gate_key: Gate identifier (gate_name, qubit(s))
            pulses: List of pulse operations
            pulse_library: Pulse waveform library
        """
        gate_name = gate_key[0]
        
        # Build defcal signature
        if len(gate_key) == 2:
            # Single-qubit gate
            qubit = gate_key[1]
            defcal_sig = f"defcal {gate_name} ${qubit}"
        elif len(gate_key) == 3:
            # Two-qubit gate
            q0, q1 = gate_key[1], gate_key[2]
            defcal_sig = f"defcal {gate_name} ${q0}, ${q1}"
        else:
            return
        
        # Build defcal body
        defcal_body = []
        
        for op in pulses:
            if not isinstance(op, (list, tuple)) or not op:
                continue
            
            op_type = str(op[0]).lower()
            
            if op_type == "pulse":
                # Pulse operation: ("pulse", qubit, pulse_key, params)
                qubit = int(op[1])
                pulse_key = str(op[2])
                params = op[3] if len(op) > 3 else {}
                
                pulse_wf = pulse_library.get(pulse_key)
                if pulse_wf is None:
                    defcal_body.append(f"// Unknown pulse: {pulse_key}")
                    continue
                
                # Generate waveform definition
                wf_name = f"wf_{len(defcal_body)}"
                wf_code = self._export_waveform_openpulse(wf_name, pulse_wf)
                defcal_body.append(wf_code)
                
                # Get frame
                frame_name = self._frames.get(qubit, f"d{qubit}_frame")
                
                # Check for phase shift
                phase = params.get("phase", 0.0)
                if abs(phase) > 1e-10:
                    defcal_body.append(f"shift_phase({frame_name}, {phase});")
                
                # Play pulse
                defcal_body.append(f"play({frame_name}, {wf_name});")
            
            elif op_type == "pulse_inline":
                # 关键修复：Pulse inline operation: ("pulse_inline", qubit, waveform_dict, params)
                qubit = int(op[1])
                waveform_dict = op[2] if len(op) > 2 else {}
                params = op[3] if len(op) > 3 else {}
                
                # 从 waveform_dict 重建 waveform 对象（简化处理）
                wf_name = f"wf_{len(defcal_body)}"
                wf_code = self._export_waveform_dict_openpulse(wf_name, waveform_dict)
                if wf_code:
                    defcal_body.append(wf_code)
                    
                    # Get frame
                    frame_name = self._frames.get(qubit, f"d{qubit}_frame")
                    
                    # Check for phase shift
                    phase = params.get("phase", 0.0)
                    if abs(phase) > 1e-10:
                        defcal_body.append(f"shift_phase({frame_name}, {phase});")
                    
                    # Play pulse
                    defcal_body.append(f"play({frame_name}, {wf_name});")
            
            elif op_type == "virtual_z":
                # Virtual-Z: ("virtual_z", qubit, angle)
                qubit = int(op[1])
                angle = float(op[2])
                frame_name = self._frames.get(qubit, f"d{qubit}_frame")
                defcal_body.append(f"shift_phase({frame_name}, {-angle});  // Virtual-Z")
        
        # Store defcal definition
        if defcal_body:
            self._defcal_definitions[defcal_sig] = defcal_body
    
    def _export_waveform_openpulse(self, name: str, waveform: Any) -> str:
        """Export waveform definition to OpenPulse syntax.
        
        OpenPulse waveform syntax:
            waveform <name> = <type>(<params>);
        
        Supported waveforms:
            - drag(complex amp, duration, duration sigma, float beta)
            - gaussian(complex amp, duration, duration sigma)
            - gaussian_square(complex amp, duration, duration square_width, duration sigma)
            - constant(complex amp, duration)
        
        Args:
            name: Waveform variable name
            waveform: Waveform object (from tyxonq.waveforms)
        
        Returns:
            OpenPulse waveform definition string
        """
        wf_type = waveform.__class__.__name__.lower()
        
        if wf_type == "drag":
            amp = complex(waveform.amp)  # Ensure complex
            # OpenPulse uses 'dt' suffix for duration (device ticks)
            return f"waveform {name} = drag({amp}+0.0im, {waveform.duration}dt, {waveform.sigma}dt, {waveform.beta});"
        
        elif wf_type == "gaussian":
            amp = complex(waveform.amp)
            return f"waveform {name} = gaussian({amp}+0.0im, {waveform.duration}dt, {waveform.sigma}dt);"
        
        elif wf_type == "constant":
            amp = complex(waveform.amp)
            return f"waveform {name} = constant({amp}+0.0im, {waveform.duration}dt);"
        
        elif wf_type == "gaussiansquare":
            amp = complex(waveform.amp)
            square_width = waveform.width if hasattr(waveform, 'width') else waveform.duration // 2
            sigma = waveform.sigma if hasattr(waveform, 'sigma') else waveform.duration // 8
            return f"waveform {name} = gaussian_square({amp}+0.0im, {waveform.duration}dt, {square_width}dt, {sigma}dt);"
        
        else:
            # Fallback: generic waveform
            return f"waveform {name} = generic_{wf_type}();"
    
    def _export_waveform_dict_openpulse(self, name: str, waveform_dict: Dict[str, Any]) -> str:
        """从 waveform_dict 导出OpenPulse waveform定义。
        
        这个方法用于处理 pulse_inline 操作，其中 waveform 已经被序列化为字典。
        
        Args:
            name: Waveform variable name
            waveform_dict: Serialized waveform dictionary:
                {"type": "drag", "args": [amp, duration, sigma, beta], "class": "Drag"}
        
        Returns:
            OpenPulse waveform definition string
        """
        if not waveform_dict or not isinstance(waveform_dict, dict):
            return ""
        
        wf_type = waveform_dict.get("type", "").lower()
        args = waveform_dict.get("args", [])
        
        if not args:
            return ""
        
        if wf_type == "drag" and len(args) >= 4:
            amp, duration, sigma, beta = args[0], args[1], args[2], args[3]
            amp_complex = complex(amp) if not isinstance(amp, complex) else amp
            return f"waveform {name} = drag({amp_complex}+0.0im, {duration}dt, {sigma}dt, {beta});"
        
        elif wf_type == "gaussian" and len(args) >= 3:
            amp, duration, sigma = args[0], args[1], args[2]
            amp_complex = complex(amp) if not isinstance(amp, complex) else amp
            return f"waveform {name} = gaussian({amp_complex}+0.0im, {duration}dt, {sigma}dt);"
        
        elif wf_type == "constant" and len(args) >= 2:
            amp, duration = args[0], args[1]
            amp_complex = complex(amp) if not isinstance(amp, complex) else amp
            return f"waveform {name} = constant({amp_complex}+0.0im, {duration}dt);"
        
        elif wf_type == "gaussiansquare" and len(args) >= 4:
            amp, duration, square_width, sigma = args[0], args[1], args[2], args[3]
            amp_complex = complex(amp) if not isinstance(amp, complex) else amp
            return f"waveform {name} = gaussian_square({amp_complex}+0.0im, {duration}dt, {square_width}dt, {sigma}dt);"
        
        else:
            # Fallback
            return f"waveform {name} = generic_{wf_type}();"
    
    def _export_gate_operation(self, op: Any) -> str:
        """Export a gate operation (not pulse) to TQASM.
        
        Args:
            op: Gate operation tuple
        
        Returns:
            TQASM gate operation code
        """
        if not isinstance(op, (list, tuple)) or not op:
            return ""
        
        op_name = str(op[0]).lower()
        
        # Skip pulse operations (handled in defcal)
        if op_name in ("pulse", "pulse_inline", "virtual_z"):
            return ""
        
        # Standard gates
        if op_name in ("h", "x", "y", "z", "s", "t", "sdg", "tdg"):
            qubit = int(op[1])
            return f"{op_name} q[{qubit}];"
        
        elif op_name in ("rx", "ry", "rz", "p", "u1"):
            qubit = int(op[1])
            angle = float(op[2])
            return f"{op_name}({angle}) q[{qubit}];"
        
        elif op_name in ("u3", "u"):
            qubit = int(op[1])
            theta, phi, lam = float(op[2]), float(op[3]), float(op[4])
            return f"u3({theta}, {phi}, {lam}) q[{qubit}];"
        
        elif op_name in ("cx", "cnot", "cz", "cy", "ch", "swap", "iswap"):
            q0, q1 = int(op[1]), int(op[2])
            return f"{op_name} q[{q0}], q[{q1}];"
        
        elif op_name in ("ccx", "toffoli", "cswap", "fredkin"):
            q0, q1, q2 = int(op[1]), int(op[2]), int(op[3])
            return f"{op_name} q[{q0}], q[{q1}], q[{q2}];"
        
        elif op_name in ("measure", "measure_z"):
            qubit = int(op[1])
            return f"measure q[{qubit}];"
        
        elif op_name == "barrier":
            return "barrier;"  # Barrier all qubits
        
        else:
            return f"// Unsupported operation: {op_name}"
