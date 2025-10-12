Custom Devices
==============

This guide provides detailed instructions for creating custom quantum devices in TyxonQ. Custom devices allow you to integrate new simulators, hardware backends, or specialized quantum computing platforms.

.. contents:: Table of Contents
   :local:
   :depth: 3

Device Architecture Overview
===========================

TyxonQ's device architecture is built on a hierarchical interface system:

.. code-block:: text

    QuantumDevice (Abstract Base)
    │
    ├── SimulatorDevice
    │   ├── StatevectorSimulator
    │   ├── DensityMatrixSimulator
    │   └── NoiseSimulator
    │
    ├── HardwareDevice
    │   ├── IBMQuantumDevice
    │   ├── IonQDevice
    │   └── RigettiDevice
    │
    └── CloudDevice
        ├── AWSBraketDevice
        ├── AzureQuantumDevice
        └── GoogleQuantumAIDevice

Base Device Interface
====================

All quantum devices must implement the ``QuantumDevice`` abstract base class:

.. code-block:: python

    from abc import ABC, abstractmethod
    from typing import Dict, Any, List, Optional
    from tyxonq.core import Circuit, Result
    
    class QuantumDevice(ABC):
        """Abstract base class for quantum devices"""
        
        def __init__(self):
            self._name = None
            self._num_qubits = None
            self._capabilities = {}
            self._status = "offline"
        
        @property
        @abstractmethod
        def name(self) -> str:
            """Device name identifier"""
            pass
        
        @property
        @abstractmethod
        def num_qubits(self) -> int:
            """Number of available qubits"""
            pass
        
        @abstractmethod
        def get_capabilities(self) -> Dict[str, Any]:
            """Return device capabilities and specifications"""
            pass
        
        @abstractmethod
        def execute(self, circuit: Circuit, shots: int = 1024, **kwargs) -> Result:
            """Execute quantum circuit on device"""
            pass
        
        def batch_execute(self, circuits: List[Circuit], shots: int = 1024) -> List[Result]:
            """Execute multiple circuits (default implementation)"""
            return [self.execute(circuit, shots) for circuit in circuits]
        
        def is_available(self) -> bool:
            """Check if device is available for execution"""
            return self._status == "online"
        
        def get_status(self) -> Dict[str, Any]:
            """Get device status information"""
            return {
                'status': self._status,
                'num_qubits': self.num_qubits,
                'capabilities': self.get_capabilities()
            }

Custom Simulator Example
=======================

.. code-block:: python

    import numpy as np
    from typing import Dict, Any
    from tyxonq.core import Circuit, Result
    
    class CustomStatevectorSimulator(QuantumDevice):
        """Custom statevector simulator implementation"""
        
        def __init__(self, num_qubits: int = 20, precision: str = 'double'):
            super().__init__()
            self._name = "custom_statevector_simulator"
            self._num_qubits = num_qubits
            self._precision = precision
            self._dtype = np.complex128 if precision == 'double' else np.complex64
            self._status = "online"
        
        @property
        def name(self) -> str:
            return self._name
        
        @property
        def num_qubits(self) -> int:
            return self._num_qubits
        
        def get_capabilities(self) -> Dict[str, Any]:
            return {
                'max_qubits': self._num_qubits,
                'simulation_method': 'statevector',
                'precision': self._precision,
                'supports_noise': False,
                'supports_midcircuit_measurement': True,
                'native_gates': ['x', 'y', 'z', 'h', 's', 't', 'rx', 'ry', 'rz', 'cnot', 'cz'],
                'max_shots': 10**6
            }
        
        def execute(self, circuit: Circuit, shots: int = 1024, **kwargs) -> Result:
            """Execute circuit using statevector simulation"""
            # Validate circuit
            self._validate_circuit(circuit)
            
            # Initialize quantum state
            state = self._initialize_state(circuit.num_qubits)
            
            # Apply gates
            state = self._apply_circuit(circuit, state)
            
            # Handle measurements
            if circuit.has_measurements():
                counts = self._sample_measurements(circuit, state, shots)
            else:
                # Return statevector if no measurements
                counts = {'statevector': state}
            
            return Result(
                counts=counts,
                shots=shots,
                circuit=circuit,
                device=self.name,
                execution_time=kwargs.get('execution_time', 0.0)
            )
        
        def _initialize_state(self, num_qubits: int) -> np.ndarray:
            """Initialize |0...0> state"""
            state = np.zeros(2**num_qubits, dtype=self._dtype)
            state[0] = 1.0  # |00...0>
            return state
        
        def _apply_circuit(self, circuit: Circuit, state: np.ndarray) -> np.ndarray:
            """Apply circuit gates to quantum state"""
            current_state = state.copy()
            
            for gate in circuit.gates:
                current_state = self._apply_gate(gate, current_state, circuit.num_qubits)
            
            return current_state
        
        def _apply_gate(self, gate, state: np.ndarray, num_qubits: int) -> np.ndarray:
            """Apply single gate to quantum state"""
            gate_matrix = gate.to_matrix()
            gate_qubits = gate.qubits
            
            if len(gate_qubits) == 1:
                return self._apply_single_qubit_gate(gate_matrix, gate_qubits[0], state, num_qubits)
            elif len(gate_qubits) == 2:
                return self._apply_two_qubit_gate(gate_matrix, gate_qubits, state, num_qubits)
            else:
                return self._apply_multi_qubit_gate(gate_matrix, gate_qubits, state, num_qubits)
        
        def _validate_circuit(self, circuit: Circuit):
            """Validate circuit for this device"""
            if circuit.num_qubits > self.num_qubits:
                raise ValueError(
                    f"Circuit requires {circuit.num_qubits} qubits, "
                    f"device supports maximum {self.num_qubits}"
                )

Hardware Device Integration
==========================

.. code-block:: python

    import requests
    import time
    from typing import Dict, Any
    
    class RemoteHardwareDevice(QuantumDevice):
        """Integration with remote quantum hardware"""
        
        def __init__(self, api_endpoint: str, api_key: str, device_name: str):
            super().__init__()
            self.api_endpoint = api_endpoint.rstrip('/')
            self.api_key = api_key
            self.device_name = device_name
            self._device_info = None
            self._status = "unknown"
            
            # Initialize connection
            self._initialize_connection()
        
        @property
        def name(self) -> str:
            return self.device_name
        
        @property
        def num_qubits(self) -> int:
            return self._device_info.get('num_qubits', 0) if self._device_info else 0
        
        def get_capabilities(self) -> Dict[str, Any]:
            if not self._device_info:
                return {}
            
            return {
                'max_qubits': self._device_info.get('num_qubits', 0),
                'native_gates': self._device_info.get('native_gates', []),
                'coupling_map': self._device_info.get('coupling_map', []),
                'gate_times': self._device_info.get('gate_times', {}),
                'coherence_times': self._device_info.get('coherence_times', {}),
                'error_rates': self._device_info.get('error_rates', {}),
                'max_shots': self._device_info.get('max_shots', 1024)
            }
        
        def execute(self, circuit: Circuit, shots: int = 1024, **kwargs) -> Result:
            """Execute circuit on remote hardware"""
            # Validate inputs
            self._validate_circuit(circuit)
            
            # Submit job
            job_id = self._submit_job(circuit, shots, **kwargs)
            
            # Wait for completion
            result_data = self._wait_for_job_completion(job_id)
            
            # Process results
            return self._process_results(result_data, circuit, shots)
        
        def _initialize_connection(self):
            """Initialize connection to remote device"""
            try:
                headers = {'Authorization': f'Bearer {self.api_key}'}
                response = requests.get(
                    f"{self.api_endpoint}/devices/{self.device_name}",
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                self._device_info = response.json()
                self._status = "online" if self._device_info.get('available', False) else "offline"
            except Exception as e:
                self._status = "error"
                raise ConnectionError(f"Failed to connect to device: {e}")
        
        def _submit_job(self, circuit: Circuit, shots: int, **kwargs) -> str:
            """Submit job to remote device"""
            job_data = {
                'circuit': self._serialize_circuit(circuit),
                'shots': shots,
                'device': self.device_name,
                'options': kwargs
            }
            
            headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
            response = requests.post(
                f"{self.api_endpoint}/jobs",
                json=job_data,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()['job_id']
        
        def _wait_for_job_completion(self, job_id: str) -> Dict[str, Any]:
            """Wait for job completion and return results"""
            headers = {'Authorization': f'Bearer {self.api_key}'}
            
            while True:
                response = requests.get(
                    f"{self.api_endpoint}/jobs/{job_id}",
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                
                job_status = response.json()
                
                if job_status['status'] == 'completed':
                    return job_status['results']
                elif job_status['status'] == 'failed':
                    error_msg = job_status.get('error', 'Unknown error')
                    raise RuntimeError(f"Job {job_id} failed: {error_msg}")
                
                time.sleep(2)  # Poll every 2 seconds
        
        def _serialize_circuit(self, circuit: Circuit) -> Dict[str, Any]:
            """Serialize circuit for API submission"""
            serialized_gates = []
            
            for gate in circuit.gates:
                gate_data = {
                    'name': gate.name,
                    'qubits': gate.qubits,
                    'params': getattr(gate, 'params', [])
                }
                serialized_gates.append(gate_data)
            
            return {
                'num_qubits': circuit.num_qubits,
                'gates': serialized_gates,
                'measurements': circuit.get_measurements()
            }
        
        def _process_results(self, result_data: Dict[str, Any], 
                           circuit: Circuit, shots: int) -> Result:
            """Process raw results from device"""
            return Result(
                counts=result_data.get('counts', {}),
                shots=shots,
                circuit=circuit,
                device=self.name,
                execution_time=result_data.get('execution_time', 0.0),
                job_id=result_data.get('job_id'),
                raw_data=result_data
            )
        
        def _validate_circuit(self, circuit: Circuit):
            """Validate circuit for hardware constraints"""
            if circuit.num_qubits > self.num_qubits:
                raise ValueError(
                    f"Circuit requires {circuit.num_qubits} qubits, "
                    f"device has {self.num_qubits}"
                )

Device Testing Framework
=======================

.. code-block:: python

    import unittest
    import numpy as np
    from tyxonq.core import Circuit
    
    class DeviceTestSuite(unittest.TestCase):
        """Comprehensive test suite for quantum devices"""
        
        def setUp(self):
            """Override with device initialization"""
            self.device = None  # Initialize your device here
        
        def test_device_properties(self):
            """Test basic device properties"""
            self.assertIsNotNone(self.device.name)
            self.assertGreater(self.device.num_qubits, 0)
            self.assertIsInstance(self.device.get_capabilities(), dict)
        
        def test_single_qubit_gates(self):
            """Test single-qubit gate execution"""
            circuit = Circuit(1)
            circuit.h(0)
            circuit.measure_all()
            
            result = self.device.execute(circuit, shots=1000)
            
            # Check statistics for H gate
            counts = result.counts
            total = sum(counts.values())
            
            # Should be roughly 50/50 split
            prob_0 = counts.get('0', 0) / total
            prob_1 = counts.get('1', 0) / total
            
            self.assertAlmostEqual(prob_0, 0.5, delta=0.1)
            self.assertAlmostEqual(prob_1, 0.5, delta=0.1)
        
        def test_two_qubit_gates(self):
            """Test two-qubit gate execution"""
            circuit = Circuit(2)
            circuit.h(0)
            circuit.cnot(0, 1)
            circuit.measure_all()
            
            result = self.device.execute(circuit, shots=1000)
            
            # Bell state should give only |00> and |11>
            counts = result.counts
            allowed_states = ['00', '11']
            
            for state in counts:
                self.assertIn(state, allowed_states)
        
        def test_error_handling(self):
            """Test error handling for invalid inputs"""
            # Too many qubits
            circuit = Circuit(self.device.num_qubits + 1)
            circuit.h(0)
            
            with self.assertRaises(ValueError):
                self.device.execute(circuit)

Best Practices
=============

Performance Optimization
-----------------------

1. **Efficient State Representation**: Use appropriate data structures for quantum states
2. **Gate Fusion**: Combine consecutive single-qubit gates when possible
3. **Memory Management**: Clean up large state vectors after use
4. **Parallel Execution**: Support batch job execution for better throughput

Error Handling
-------------

1. **Input Validation**: Thoroughly validate all inputs before execution
2. **Resource Management**: Handle device availability and queue management
3. **Graceful Degradation**: Provide meaningful error messages
4. **Retry Logic**: Implement retry mechanisms for transient failures

Testing Guidelines
-----------------

1. **Unit Tests**: Test individual components thoroughly
2. **Integration Tests**: Test device integration with TyxonQ framework
3. **Performance Tests**: Benchmark device performance
4. **Stress Tests**: Test device under high load conditions

.. note::
   This guide covers the essentials of creating custom devices in TyxonQ.
   For advanced topics like noise modeling and cloud integration, refer to the extended documentation.

.. seealso::
   
   - :doc:`extending_tyxonq` - General framework extension guide
   - :doc:`plugin_system` - Plugin system architecture
   - :doc:`testing_guidelines` - Testing best practices
   - :doc:`architecture_overview` - TyxonQ architecture overview
