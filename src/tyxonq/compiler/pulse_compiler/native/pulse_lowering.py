"""Pulse lowering pass - Inline defcal definitions.

This pass expands pulse calibration (defcal) definitions into inline pulse
operations, removing the abstraction layer for direct hardware execution.

Functionality:
    - Inline user-defined pulse calibrations
    - Expand parametric pulse definitions
    - Resolve pulse scheduling dependencies

This is analogous to macro expansion in traditional compilers.
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit


class PulseLoweringPass:
    """Inline pulse calibration definitions.
    
    This pass resolves all symbolic pulse references to concrete pulse
    waveforms, preparing the circuit for direct execution.
    
    Example:
        >>> pass_instance = PulseLoweringPass()
        >>> lowered_circuit = pass_instance.execute_plan(pulse_circuit)
    """
    
    def __init__(self):
        """Initialize the pulse lowering pass."""
        pass
    
    def execute_plan(
        self,
        circuit: "Circuit",
        **options: Any
    ) -> "Circuit":
        """Execute pulse lowering (defcal inlining).
        
        Args:
            circuit: Input circuit with pulse operations
            **options: Compilation options
        
        Returns:
            Circuit with inlined pulse definitions
        """
        # Get pulse library from circuit metadata
        pulse_library = circuit.metadata.get("pulse_library", {})
        
        if not pulse_library:
            # No pulses to lower
            return circuit
        
        new_ops: List[Any] = []
        
        for op in circuit.ops:
            if not isinstance(op, (list, tuple)) or not op:
                new_ops.append(op)
                continue
            
            op_name = str(op[0]).lower()
            
            if op_name == "pulse":
                # Expand pulse operation
                expanded_ops = self._expand_pulse_op(op, pulse_library, circuit)
                new_ops.extend(expanded_ops)
            else:
                new_ops.append(op)
        
        from dataclasses import replace
        return replace(circuit, ops=new_ops)
    
    def _expand_pulse_op(
        self,
        op: Any,
        pulse_library: Dict[str, Any],
        circuit: "Circuit"
    ) -> List[Any]:
        """Expand a pulse operation into inline form.
        
        Args:
            op: Pulse operation tuple ("pulse", qubit, pulse_key, params)
            pulse_library: Circuit's pulse library (from metadata)
            circuit: Circuit object
        
        Returns:
            List of expanded pulse operations
        """
        if len(op) < 3:
            # Invalid pulse operation
            return [op]
        
        qubit = int(op[1])
        pulse_key = str(op[2])
        params = op[3] if len(op) > 3 else {}
        
        # Retrieve pulse waveform from library
        pulse_waveform = pulse_library.get(pulse_key)
        
        if pulse_waveform is None:
            # Pulse not found in cache, keep original
            return [op]
        
        # Create inline pulse operation with expanded parameters
        # Format: ("pulse_inline", qubit, waveform_dict, params)
        inline_op = (
            "pulse_inline",
            qubit,
            self._serialize_waveform(pulse_waveform),
            params
        )
        
        return [inline_op]
    
    def _serialize_waveform(self, waveform: Any) -> Dict[str, Any]:
        """Serialize a waveform object to dictionary.
        
        Args:
            waveform: Waveform object (from tyxonq.waveforms)
        
        Returns:
            Dictionary representation of waveform
        """
        if hasattr(waveform, "qasm_name") and hasattr(waveform, "to_args"):
            # Standard waveform object
            return {
                "type": waveform.qasm_name(),
                "args": waveform.to_args(),
                "class": type(waveform).__name__
            }
        else:
            # Generic object, try dataclass conversion
            if hasattr(waveform, "__dataclass_fields__"):
                from dataclasses import asdict
                return {
                    "type": "custom",
                    "data": asdict(waveform),
                    "class": type(waveform).__name__
                }
            else:
                # Fallback: return string representation
                return {
                    "type": "unknown",
                    "repr": str(waveform),
                    "class": type(waveform).__name__
                }
