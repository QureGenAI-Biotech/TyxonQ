"""
DefcalLibrary: User-defined Quantum Gate Calibration Management

This module provides a comprehensive calibration library system for managing
hardware-specific quantum gate pulse parameters. It enables users to:

1. Store calibrations for single-qubit and two-qubit gates
2. Retrieve calibrations efficiently with flexible query interfaces
3. Persist calibrations to JSON for cross-session reuse
4. Validate calibration data integrity

Physical Context:
    In real quantum hardware, the optimal pulse parameters for the same gate
    operation can differ across qubits. For example, X gate parameters on
    different qubits might be:
        q0: DRAG(amp=0.8, duration=40ns, sigma=10, beta=0.2)
        q1: DRAG(amp=0.85, duration=42ns, sigma=11, beta=0.18)
        q2: DRAG(amp=0.75, duration=39ns, sigma=9, beta=0.22)

    DefcalLibrary stores and manages these hardware-specific calibrations,
    enabling the pulse compiler to apply optimized gate implementations.

Author: TyxonQ Development Team
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """
    Data class representing a single quantum gate calibration.
    
    Attributes:
        gate: Gate name (e.g., 'x', 'y', 'z', 'cx', 'cz')
        qubits: Target qubits as tuple (e.g., (0,) for single-qubit, (0,1) for two-qubit)
        pulse: Pulse waveform object (e.g., DRAG, Gaussian)
        params: Additional calibration parameters (duration, amplitude, phase, etc.)
        timestamp: Creation/modification timestamp for version tracking
        description: User-provided description of the calibration
        hardware: Hardware identifier (e.g., 'Homebrew_S2')
    """
    
    gate: str
    qubits: Tuple[int, ...]
    pulse: Any  # Waveform object (typing left flexible for framework compatibility)
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    description: str = ""
    hardware: str = "Homebrew_S2"
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # Ensure qubits is a tuple
        if isinstance(self.qubits, int):
            self.qubits = (self.qubits,)
        elif isinstance(self.qubits, list):
            self.qubits = tuple(self.qubits)
        
        # Ensure params is a dict
        if self.params is None:
            self.params = {}
    
    def get_qubit_count(self) -> int:
        """Return the number of qubits this calibration targets."""
        return len(self.qubits)
    
    def is_single_qubit(self) -> bool:
        """Check if this is a single-qubit calibration."""
        return len(self.qubits) == 1
    
    def is_two_qubit(self) -> bool:
        """Check if this is a two-qubit calibration."""
        return len(self.qubits) == 2
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert calibration to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the calibration
        """
        return {
            'gate': self.gate,
            'qubits': list(self.qubits),
            'pulse_type': self.pulse.__class__.__name__ if hasattr(self.pulse, '__class__') else str(type(self.pulse)),
            'params': self.params.copy() if self.params else {},
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'description': self.description,
            'hardware': self.hardware,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], pulse_factory: callable = None) -> CalibrationData:
        """
        Create CalibrationData from dictionary.
        
        Args:
            data: Dictionary representation
            pulse_factory: Optional callable to reconstruct pulse object from data
        
        Returns:
            CalibrationData instance
        """
        timestamp = None
        if data.get('timestamp'):
            try:
                timestamp = datetime.fromisoformat(data['timestamp'])
            except (ValueError, TypeError):
                timestamp = None
        
        # Reconstruct pulse object if factory provided, otherwise use dict representation
        pulse = None
        if pulse_factory:
            pulse = pulse_factory(data.get('pulse_type'), data.get('params', {}))
        else:
            pulse = data.get('pulse_type', 'Unknown')  # Fallback for load without factory
        
        return cls(
            gate=data.get('gate', ''),
            qubits=tuple(data.get('qubits', [])),
            pulse=pulse,
            params=data.get('params', {}),
            timestamp=timestamp,
            description=data.get('description', ''),
            hardware=data.get('hardware', 'Homebrew_S2'),
        )
    
    def __hash__(self):
        """Make calibration hashable by gate and qubits."""
        return hash((self.gate, self.qubits))
    
    def __eq__(self, other):
        """Check equality based on gate and qubits."""
        if not isinstance(other, CalibrationData):
            return False
        return self.gate == other.gate and self.qubits == other.qubits


class DefcalLibrary:
    """
    Manages quantum gate calibration library with storage, retrieval, and validation.
    
    This class provides a centralized repository for hardware-specific gate
    calibrations, enabling the pulse compiler to apply optimized parameters
    during compilation.
    
    Example:
        >>> lib = DefcalLibrary(hardware="Homebrew_S2")
        >>> lib.add_calibration("x", 0, pulse_obj, {"duration": 40, "amp": 0.8})
        >>> calib = lib.get_calibration("x", (0,))
        >>> lib.export_to_json("calibrations.json")
    """
    
    def __init__(self, hardware: str = "Homebrew_S2"):
        """
        Initialize the calibration library.
        
        Args:
            hardware: Hardware identifier (default: "Homebrew_S2")
        """
        self.hardware = hardware
        self._calibrations: Dict[Tuple[str, Tuple[int, ...]], CalibrationData] = {}
        self._metadata = {
            'version': '1.0',
            'hardware': hardware,
            'created': datetime.now().isoformat(),
        }
    
    def _make_key(self, gate: str, qubits: Union[int, Tuple[int, ...], None]) -> Tuple[str, Union[Tuple[int, ...], None]]:
        """
        Create a normalized key for the calibration dictionary.
        
        Args:
            gate: Gate name
            qubits: Target qubit(s) or None for wildcard
        
        Returns:
            Tuple (gate, qubits_tuple) for dictionary lookup
        """
        gate = gate.lower().strip()
        
        if qubits is None:
            return (gate, None)
        
        if isinstance(qubits, int):
            qubits = (qubits,)
        elif isinstance(qubits, list):
            qubits = tuple(qubits)
        
        return (gate, qubits)
    
    def add_calibration(
        self,
        gate: str,
        qubits: Union[int, Tuple[int, ...], List[int]],
        pulse: Any,
        params: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> None:
        """
        Add or update a gate calibration.
        
        Args:
            gate: Gate name (e.g., 'x', 'cx')
            qubits: Target qubit(s) - single int, tuple, or list
            pulse: Pulse waveform object
            params: Optional calibration parameters (duration, amplitude, etc.)
            description: Optional description of the calibration
        
        Raises:
            ValueError: If gate or qubits are invalid
        
        Example:
            >>> lib.add_calibration("x", 0, drag_pulse, {"duration": 40, "amp": 0.8})
            >>> lib.add_calibration("cx", (0, 1), cx_pulse, {"duration": 200})
        """
        if not gate or not isinstance(gate, str):
            raise ValueError(f"Invalid gate name: {gate}")
        
        # Normalize qubits
        if isinstance(qubits, int):
            qubits = (qubits,)
        elif isinstance(qubits, list):
            qubits = tuple(qubits)
        
        if not qubits or not all(isinstance(q, int) and q >= 0 for q in qubits):
            raise ValueError(f"Invalid qubits: {qubits}")
        
        if pulse is None:
            raise ValueError("Pulse object cannot be None")
        
        # Create calibration data
        calib_data = CalibrationData(
            gate=gate.lower(),
            qubits=qubits,
            pulse=pulse,
            params=params or {},
            description=description,
            hardware=self.hardware,
        )
        
        # Store in dictionary
        key = (calib_data.gate, calib_data.qubits)
        self._calibrations[key] = calib_data
        
        logger.debug(f"Added calibration: {gate} on qubits {qubits}")
    
    def get_calibration(
        self,
        gate: str,
        qubits: Union[int, Tuple[int, ...], List[int], None] = None,
    ) -> Union[CalibrationData, List[CalibrationData], None]:
        """
        Retrieve calibration(s) by gate and qubits.
        
        If qubits is specified, returns exact match or None.
        If qubits is None, returns list of all calibrations for the gate.
        
        Args:
            gate: Gate name
            qubits: Target qubit(s), or None for all matching gate
        
        Returns:
            - Single CalibrationData if exact match found
            - List of CalibrationData if qubits=None
            - None if no match found
        
        Example:
            >>> calib = lib.get_calibration("x", (0,))  # Exact match
            >>> all_x_calibs = lib.get_calibration("x", None)  # All X gates
        """
        gate = gate.lower().strip()
        
        if qubits is None:
            # Return all calibrations for this gate
            results = [
                calib for calib in self._calibrations.values()
                if calib.gate == gate
            ]
            return results if results else None
        
        # Normalize qubits for exact lookup
        if isinstance(qubits, int):
            qubits = (qubits,)
        elif isinstance(qubits, list):
            qubits = tuple(qubits)
        
        key = (gate, qubits)
        return self._calibrations.get(key, None)
    
    def remove_calibration(
        self,
        gate: str,
        qubits: Union[int, Tuple[int, ...], List[int]],
    ) -> bool:
        """
        Remove a calibration from the library.
        
        Args:
            gate: Gate name
            qubits: Target qubit(s)
        
        Returns:
            True if calibration was removed, False if it didn't exist
        
        Example:
            >>> lib.remove_calibration("x", (0,))
        """
        gate = gate.lower().strip()
        
        if isinstance(qubits, int):
            qubits = (qubits,)
        elif isinstance(qubits, list):
            qubits = tuple(qubits)
        
        key = (gate, qubits)
        if key in self._calibrations:
            del self._calibrations[key]
            logger.debug(f"Removed calibration: {gate} on qubits {qubits}")
            return True
        
        return False
    
    def list_calibrations(
        self,
        gate: Optional[str] = None,
        qubit: Optional[int] = None,
    ) -> List[CalibrationData]:
        """
        List calibrations with optional filtering.
        
        Args:
            gate: Optional gate name to filter by
            qubit: Optional qubit index to filter by
        
        Returns:
            List of matching CalibrationData objects
        
        Example:
            >>> lib.list_calibrations(gate="x")  # All X gates
            >>> lib.list_calibrations(qubit=0)   # All gates on q0
            >>> lib.list_calibrations()           # All calibrations
        """
        results = list(self._calibrations.values())
        
        if gate:
            gate = gate.lower().strip()
            results = [c for c in results if c.gate == gate]
        
        if qubit is not None:
            results = [c for c in results if qubit in c.qubits]
        
        return results
    
    def has_calibration(self, gate: str, qubits: Union[int, Tuple[int, ...], List[int]]) -> bool:
        """
        Check if a calibration exists.
        
        Args:
            gate: Gate name
            qubits: Target qubit(s)
        
        Returns:
            True if calibration exists
        """
        return self.get_calibration(gate, qubits) is not None
    
    def validate(self) -> bool:
        """
        Validate all calibrations in the library.
        
        Checks:
        - All calibrations have non-None pulse objects
        - Gate names are valid strings
        - Qubit indices are non-negative integers
        - Parameters are properly structured
        
        Returns:
            True if all calibrations are valid
        
        Raises:
            ValueError: If any calibration is invalid (optionally)
        """
        for key, calib in self._calibrations.items():
            # Check gate name
            if not calib.gate or not isinstance(calib.gate, str):
                logger.error(f"Invalid gate in calibration: {key}")
                return False
            
            # Check qubits
            if not calib.qubits or not all(isinstance(q, int) and q >= 0 for q in calib.qubits):
                logger.error(f"Invalid qubits in calibration: {key}")
                return False
            
            # Check pulse object
            if calib.pulse is None:
                logger.error(f"Missing pulse in calibration: {key}")
                return False
            
            # Check params is dict
            if not isinstance(calib.params, dict):
                logger.error(f"Invalid params in calibration: {key}")
                return False
        
        logger.info(f"Validation passed: {len(self._calibrations)} calibrations OK")
        return True
    
    def export_to_json(self, filepath: Union[str, Path]) -> None:
        """
        Export calibration library to JSON file.
        
        Args:
            filepath: Path to JSON file
        
        Example:
            >>> lib.export_to_json("calibrations.json")
        """
        filepath = Path(filepath)
        
        # Prepare export data
        calibrations_data = {}
        for (gate, qubits), calib in self._calibrations.items():
            key = f"{gate}|{','.join(map(str, qubits))}"
            calibrations_data[key] = calib.to_dict()
        
        export_dict = {
            'version': self._metadata['version'],
            'hardware': self.hardware,
            'created': self._metadata['created'],
            'last_updated': datetime.now().isoformat(),
            'calibration_count': len(self._calibrations),
            'calibrations': calibrations_data,
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(export_dict, f, indent=2, default=str)
        
        logger.info(f"Exported {len(self._calibrations)} calibrations to {filepath}")
    
    def import_from_json(
        self,
        filepath: Union[str, Path],
        pulse_factory: Optional[callable] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Import calibrations from JSON file.
        
        Args:
            filepath: Path to JSON file
            pulse_factory: Optional callable to reconstruct pulse objects
            overwrite: If True, clear existing calibrations before import
        
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        
        Example:
            >>> lib.import_from_json("calibrations.json")
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Calibration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Validate version compatibility
        version = data.get('version', '1.0')
        if version != '1.0':
            logger.warning(f"Calibration file version {version} may not be fully compatible")
        
        # Check hardware compatibility
        import_hardware = data.get('hardware', 'Unknown')
        if import_hardware != self.hardware:
            logger.warning(
                f"Imported calibrations from {import_hardware}, "
                f"but library configured for {self.hardware}"
            )
        
        # Clear existing if requested
        if overwrite:
            self._calibrations.clear()
        
        # Import calibrations
        calibrations_data = data.get('calibrations', {})
        for key_str, calib_dict in calibrations_data.items():
            try:
                calib = CalibrationData.from_dict(calib_dict, pulse_factory)
                dict_key = (calib.gate, calib.qubits)
                self._calibrations[dict_key] = calib
            except Exception as e:
                logger.error(f"Error importing calibration {key_str}: {e}")
                continue
        
        logger.info(f"Imported {len(self._calibrations)} calibrations from {filepath}")
    
    def clear(self) -> None:
        """Clear all calibrations from the library."""
        self._calibrations.clear()
        logger.info("Cleared all calibrations from library")
    
    def __len__(self) -> int:
        """Return the number of calibrations in the library."""
        return len(self._calibrations)
    
    def __repr__(self) -> str:
        """String representation of the library."""
        return (
            f"DefcalLibrary(hardware={self.hardware!r}, "
            f"calibrations={len(self._calibrations)})"
        )
    
    def summary(self) -> str:
        """
        Return a human-readable summary of the library contents.
        
        Returns:
            Formatted string with calibration summary
        """
        lines = [
            f"DefcalLibrary Summary",
            f"=" * 50,
            f"Hardware: {self.hardware}",
            f"Total Calibrations: {len(self._calibrations)}",
            f"Created: {self._metadata.get('created', 'Unknown')}",
            f"",
            f"Calibrations by Gate:",
        ]
        
        # Group by gate
        gates = {}
        for calib in self._calibrations.values():
            if calib.gate not in gates:
                gates[calib.gate] = []
            gates[calib.gate].append(calib)
        
        for gate in sorted(gates.keys()):
            calibs = gates[gate]
            lines.append(f"  {gate.upper()}:")
            for calib in sorted(calibs, key=lambda c: c.qubits):
                qubits_str = ",".join(map(str, calib.qubits))
                pulse_type = calib.pulse.__class__.__name__ if hasattr(calib.pulse, '__class__') else 'Unknown'
                lines.append(f"    q[{qubits_str}]: {pulse_type}")
                if calib.description:
                    lines.append(f"      ({calib.description})")
        
        return "\n".join(lines)
