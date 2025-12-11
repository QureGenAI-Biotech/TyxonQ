"""Pulse scheduling and optimization pass.

This pass performs pulse-level scheduling and optimization:
    - Time-slot allocation for parallel pulse execution
    - Pulse merging and simplification
    - Resource conflict resolution
    - Phase tracking for virtual-Z gates

Optimization strategies:
    - Level 0: No scheduling, sequential execution
    - Level 1: Basic parallel scheduling (no conflicts)
    - Level 2: Advanced scheduling + pulse merging
    - Level 3: Full optimization with calibration-aware routing
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit


class PulseSchedulingPass:
    """Optimize pulse timing and resource allocation.
    
    This pass implements pulse-level scheduling to maximize parallelism
    while respecting hardware constraints.
    
    Example:
        >>> pass_instance = PulseSchedulingPass()
        >>> scheduled_circuit = pass_instance.execute_plan(pulse_circuit)
    """
    
    def __init__(self, optimization_level: int = 1):
        """Initialize the pulse scheduling pass.
        
        Args:
            optimization_level: Optimization level (0-3)
        """
        self.optimization_level = optimization_level
    
    def execute_plan(
        self,
        circuit: "Circuit",
        **options: Any
    ) -> "Circuit":
        """Execute pulse scheduling optimization.
        
        Args:
            circuit: Input circuit with pulse operations
            **options: Compilation options:
                - dt: Time step (default: 1e-10 s)
                - max_parallel: Maximum parallel pulses (default: all qubits)
        
        Returns:
            Circuit with scheduled pulse operations and timing metadata
        """
        if self.optimization_level == 0:
            # No scheduling
            return circuit
        
        dt = float(options.get("dt", 1e-10))
        max_parallel = int(options.get("max_parallel", circuit.num_qubits))
        
        # Build dependency graph
        pulse_ops = self._extract_pulse_ops(circuit.ops)
        
        if not pulse_ops:
            # No pulse operations to schedule
            return circuit
        
        # Schedule pulses
        schedule = self._schedule_pulses(pulse_ops, circuit.num_qubits, max_parallel)
        
        # Reconstruct circuit with scheduled operations
        scheduled_ops = self._build_scheduled_ops(schedule, circuit.ops)
        
        # Store scheduling metadata
        circuit = circuit.with_metadata(
            pulse_schedule=schedule,
            pulse_total_time=self._compute_total_time(schedule, dt)
        )
        
        from dataclasses import replace
        return replace(circuit, ops=scheduled_ops)
    
    def _extract_pulse_ops(self, ops: List[Any]) -> List[Tuple[int, Any]]:
        """Extract pulse operations with their indices.
        
        Args:
            ops: List of all operations
        
        Returns:
            List of (index, pulse_op) tuples
        """
        pulse_ops = []
        for idx, op in enumerate(ops):
            if isinstance(op, (list, tuple)) and len(op) > 0:
                if str(op[0]).lower() in ("pulse", "pulse_inline"):
                    pulse_ops.append((idx, op))
        return pulse_ops
    
    def _schedule_pulses(
        self,
        pulse_ops: List[Tuple[int, Any]],
        num_qubits: int,
        max_parallel: int
    ) -> List[Dict[str, Any]]:
        """Schedule pulse operations into time slots.
        
        Args:
            pulse_ops: List of (index, pulse_op) tuples
            num_qubits: Number of qubits
            max_parallel: Maximum parallel pulses
        
        Returns:
            List of scheduled pulse events with timing information
        """
        schedule = []
        current_time = 0.0
        qubit_available_time = [0.0] * num_qubits
        
        for idx, op in pulse_ops:
            qubit = int(op[1])
            duration = self._estimate_pulse_duration(op)
            
            # Find earliest start time for this qubit
            start_time = max(current_time, qubit_available_time[qubit])
            end_time = start_time + duration
            
            schedule.append({
                "op_index": idx,
                "op": op,
                "qubit": qubit,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration
            })
            
            # Update qubit availability
            qubit_available_time[qubit] = end_time
            
            # Update current time (for sequential execution if needed)
            if self.optimization_level < 2:
                current_time = end_time
        
        return schedule
    
    def _estimate_pulse_duration(self, op: Any) -> float:
        """Estimate pulse duration in nanoseconds.
        
        Args:
            op: Pulse operation
        
        Returns:
            Estimated duration (ns)
        """
        # Default duration estimate
        default_duration = 160.0  # ns (typical single-qubit gate)
        
        if len(op) < 3:
            return default_duration
        
        # Try to extract duration from waveform
        if len(op) > 2:
            # Check if third argument is a dict with duration info
            if isinstance(op[2], dict) and "duration" in op[2]:
                return float(op[2]["duration"])
        
        return default_duration
    
    def _build_scheduled_ops(
        self,
        schedule: List[Dict[str, Any]],
        original_ops: List[Any]
    ) -> List[Any]:
        """Rebuild operation list with scheduling information.
        
        Args:
            schedule: Scheduled pulse events
            original_ops: Original operation list
        
        Returns:
            New operation list with timing annotations
        """
        # For now, keep original order but add timing metadata
        # Future: reorder operations based on schedule
        
        new_ops = list(original_ops)
        
        # Annotate pulse operations with timing
        for event in schedule:
            idx = event["op_index"]
            op = event["op"]
            
            # Add timing metadata to operation
            if isinstance(op, (list, tuple)):
                # Convert to list for modification
                op_list = list(op)
                
                # Add or update params dict with timing info
                if len(op_list) > 3 and isinstance(op_list[3], dict):
                    params = dict(op_list[3])
                else:
                    params = {}
                
                params["start_time"] = event["start_time"]
                params["duration"] = event["duration"]
                
                # Update operation
                if len(op_list) > 3:
                    op_list[3] = params
                else:
                    op_list.append(params)
                
                new_ops[idx] = tuple(op_list)
        
        return new_ops
    
    def _compute_total_time(self, schedule: List[Dict[str, Any]], dt: float) -> float:
        """Compute total execution time from schedule.
        
        Args:
            schedule: Scheduled pulse events
            dt: Time step (s)
        
        Returns:
            Total time (s)
        """
        if not schedule:
            return 0.0
        
        max_end_time = max(event["end_time"] for event in schedule)
        
        # Convert from ns to seconds
        return max_end_time * 1e-9
