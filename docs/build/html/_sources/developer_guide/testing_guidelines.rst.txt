Testing Guidelines
==================

Comprehensive testing is crucial for maintaining the quality and reliability of TyxonQ. This guide covers testing strategies and best practices based on TyxonQ's actual architecture.

.. contents:: Table of Contents
   :local:
   :depth: 3

Testing Philosophy
=================

TyxonQ Testing Approach
----------------------

TyxonQ follows a structured testing approach:

.. code-block:: text

    ┌────────────────────────────────────────────────────────┐
    │                  Integration Tests                       │
    │           (Full workflow validation)                   │
    └────────────────────────────────────────────────────────┘
                            ┌────────────────────────────────────────────────────────────────────────── ┐
                            │                Component Tests                            │
                            │            (Engine/Device specific)                     │
                            └────────────────────────────────────────────────────────────────────────── ┘
                    ┌──────────────────────────────────────────────────────────────────────────────────── ┐
                    │                        Unit Tests                               │
                    │                   (Core functionality)                        │
                    └──────────────────────────────────────────────────────────────────────────────────── ┘

Unit Testing
===========

Core Circuit Testing
-------------------

Testing TyxonQ circuits using the actual API:

.. code-block:: python

    import unittest
    import numpy as np
    import tyxonq as tq
    
    class TestTyxonQCircuit(unittest.TestCase):
        """Unit tests for TyxonQ circuits"""
        
        def setUp(self):
            """Set up test fixtures"""
            tq.set_backend("numpy")
        
        def test_circuit_creation(self):
            """Test basic circuit creation"""
            circuit = tq.Circuit(2)
            self.assertEqual(circuit.num_qubits, 2)
            self.assertEqual(len(circuit.ops), 0)
        
        def test_gate_addition(self):
            """Test adding gates to circuit"""
            circuit = tq.Circuit(2)
            circuit.h(0)
            circuit.cx(0, 1)
            
            self.assertEqual(len(circuit.ops), 2)
            self.assertEqual(circuit.ops[0][0], 'h')
            self.assertEqual(circuit.ops[1][0], 'cx')
        
        def test_bell_state_creation(self):
            """Test Bell state creation"""
            circuit = tq.Circuit(2).h(0).cx(0, 1)
            
            # Execute on statevector simulator
            result = (
                circuit
                .device(provider="simulator", device="statevector", shots=1000)
                .run()
            )
            
            # Check result structure
            self.assertIsInstance(result, (dict, list))
            if isinstance(result, list) and result:
                result = result[0]
            
            counts = result.get("result", {})
            total_counts = sum(counts.values())
            self.assertGreater(total_counts, 0)

Engine Testing
-------------

Testing different simulation engines:

.. code-block:: python

    class TestSimulationEngines(unittest.TestCase):
        """Test different simulation engines"""
        
        def setUp(self):
            tq.set_backend("numpy")
            self.circuit = tq.Circuit(2).h(0).cx(0, 1)
        
        def test_statevector_engine(self):
            """Test statevector simulation engine"""
            result = (
                self.circuit
                .device(provider="simulator", device="statevector", shots=1000)
                .run()
            )
            
            self.assertIsNotNone(result)
        
        def test_density_matrix_engine(self):
            """Test density matrix simulation engine"""
            result = (
                self.circuit
                .device(provider="simulator", device="density_matrix", shots=1000)
                .run()
            )
            
            self.assertIsNotNone(result)
        
        def test_mps_engine(self):
            """Test matrix product state engine"""
            result = (
                self.circuit
                .device(provider="simulator", device="matrix_product_state", shots=1000)
                .run()
            )
            
            self.assertIsNotNone(result)

Numeric Backend Testing
----------------------

.. code-block:: python

    class TestNumericBackends(unittest.TestCase):
        """Test numeric backend switching"""
        
        def test_numpy_backend(self):
            """Test NumPy backend"""
            tq.set_backend("numpy")
            circuit = tq.Circuit(2).h(0).cx(0, 1)
            
            result = (
                circuit
                .device(provider="simulator", device="statevector")
                .run()
            )
            
            self.assertIsNotNone(result)
        
        def test_pytorch_backend(self):
            """Test PyTorch backend if available"""
            try:
                tq.set_backend("pytorch")
                circuit = tq.Circuit(2).h(0).cx(0, 1)
                
                result = (
                    circuit
                    .device(provider="simulator", device="statevector")
                    .run()
                )
                
                self.assertIsNotNone(result)
            except Exception:
                self.skipTest("PyTorch backend not available")

Quantum Chemistry Testing
========================

Testing quantum chemistry algorithms:

.. code-block:: python

    class TestQuantumChemistry(unittest.TestCase):
        """Test quantum chemistry applications"""
        
        def setUp(self):
            tq.set_backend("numpy")
        
        def test_uccsd_algorithm(self):
            """Test UCCSD algorithm"""
            try:
                from tyxonq.applications.chem.algorithms.uccsd import UCCSD
                from tyxonq.applications.chem import molecule
                
                # Create UCCSD instance
                uccsd = UCCSD(molecule.h2)
                
                # Test energy calculation
                params = [0.1, 0.2]  # Simple test parameters
                energy = uccsd.energy(params, runtime="numeric")
                
                self.assertIsInstance(energy, float)
                
            except ImportError:
                self.skipTest("Quantum chemistry module not available")
        
        def test_dual_path_execution(self):
            """Test dual-path execution model"""
            try:
                from tyxonq.applications.chem.algorithms.uccsd import UCCSD
                from tyxonq.applications.chem import molecule
                
                uccsd = UCCSD(molecule.h2)
                params = [0.1, 0.2]
                
                # Numeric path
                energy_numeric = uccsd.energy(params, runtime="numeric")
                
                # Device path
                energy_device = uccsd.energy(
                    params, 
                    runtime="device", 
                    provider="simulator", 
                    device="statevector",
                    shots=1000
                )
                
                # Both should return float values
                self.assertIsInstance(energy_numeric, float)
                self.assertIsInstance(energy_device, float)
                
            except ImportError:
                self.skipTest("Quantum chemistry module not available")

Integration Testing
==================

Workflow Testing
---------------

.. code-block:: python

    class TestWorkflows(unittest.TestCase):
        """Test complete TyxonQ workflows"""
        
        def setUp(self):
            tq.set_backend("numpy")
        
        def test_chain_api_workflow(self):
            """Test complete chain API workflow"""
            # Build circuit
            circuit = tq.Circuit(2).h(0).cx(0, 1)
            
            # Complete workflow
            result = (
                circuit
                .compile()
                .device(provider="simulator", device="statevector", shots=1024)
                .run()
            )
            
            self.assertIsNotNone(result)
        
        def test_compilation_workflow(self):
            """Test compilation pipeline"""
            circuit = tq.Circuit(2).h(0).cx(0, 1)
            
            # Test different compilation options
            compiled_circuit = circuit.compile()
            self.assertIsNotNone(compiled_circuit)
            
            # Test with specific passes
            compiled_circuit = circuit.compile(
                passes=["measurement_rewrite", "shot_scheduler"]
            )
            self.assertIsNotNone(compiled_circuit)

Performance Testing
==================

Benchmarking
-----------

.. code-block:: python

    import time
    
    class TestPerformance(unittest.TestCase):
        """Performance and benchmarking tests"""
        
        def setUp(self):
            tq.set_backend("numpy")
        
        def test_simulation_performance(self):
            """Test simulation performance scaling"""
            qubit_counts = [5, 10, 15]
            times = []
            
            for num_qubits in qubit_counts:
                circuit = tq.Circuit(num_qubits)
                
                # Add gates
                for i in range(num_qubits):
                    circuit.h(i)
                for i in range(num_qubits - 1):
                    circuit.cx(i, i + 1)
                
                # Time execution
                start_time = time.time()
                result = (
                    circuit
                    .device(provider="simulator", device="statevector", shots=100)
                    .run()
                )
                end_time = time.time()
                
                execution_time = end_time - start_time
                times.append(execution_time)
                
                # Verify result
                self.assertIsNotNone(result)
            
            # Check reasonable scaling (exponential but controlled)
            for i in range(1, len(times)):
                scaling_factor = times[i] / times[i-1]
                self.assertLess(scaling_factor, 10.0)  # Should not be too bad
        
        def test_backend_comparison(self):
            """Compare different backend performance"""
            circuit = tq.Circuit(10)
            for i in range(10):
                circuit.h(i)
            
            backends = ["numpy"]
            try:
                tq.set_backend("pytorch")
                backends.append("pytorch")
            except:
                pass
            
            for backend_name in backends:
                tq.set_backend(backend_name)
                
                start_time = time.time()
                result = (
                    circuit
                    .device(provider="simulator", device="statevector", shots=100)
                    .run()
                )
                end_time = time.time()
                
                execution_time = end_time - start_time
                print(f"{backend_name} backend: {execution_time:.4f}s")
                
                self.assertIsNotNone(result)

Test Utilities
=============

Helper Functions
---------------

.. code-block:: python

    def extract_counts(result):
        """Extract counts from TyxonQ result"""
        if isinstance(result, list) and result:
            result = result[0]
        if isinstance(result, dict):
            return result.get("result", {})
        return {}
    
    def check_bell_state_statistics(counts, tolerance=0.1):
        """Check if counts represent Bell state statistics"""
        total = sum(counts.values())
        if total == 0:
            return False
        
        # Bell state should have roughly equal |00⟩ and |11⟩
        prob_00 = counts.get('00', 0) / total
        prob_11 = counts.get('11', 0) / total
        
        return (abs(prob_00 - 0.5) < tolerance and 
                abs(prob_11 - 0.5) < tolerance)
    
    def setup_test_environment():
        """Setup standard test environment"""
        tq.set_backend("numpy")
        # Add any other standard setup

Best Practices
=============

Testing Guidelines
-----------------

1. **Use Real APIs**: Test with actual TyxonQ interfaces, not mocked versions
2. **Backend Agnostic**: Test across different numeric backends when possible
3. **Error Handling**: Include tests for error conditions and edge cases
4. **Resource Cleanup**: Properly clean up resources after tests
5. **Deterministic Results**: Use fixed random seeds for reproducible tests
6. **Performance Monitoring**: Include performance regression tests

Test Organization
----------------

.. code-block:: text

    tests/
    ├── unit/
    │   ├── test_core_circuit.py
    │   ├── test_engines.py
    │   └── test_backends.py
    ├── integration/
    │   ├── test_workflows.py
    │   └── test_chemistry.py
    ├── performance/
    │   └── test_benchmarks.py
    └── conftest.py

Continuous Integration
=====================

Test Configuration
-----------------

.. code-block:: yaml

    # Example CI configuration
    name: TyxonQ Tests
    
    on: [push, pull_request]
    
    jobs:
      test:
        runs-on: ubuntu-latest
        strategy:
          matrix:
            python-version: [3.8, 3.9, '3.10', 3.11]
        
        steps:
        - uses: actions/checkout@v3
        
        - name: Set up Python ${{ matrix.python-version }}
          uses: actions/setup-python@v3
          with:
            python-version: ${{ matrix.python-version }}
        
        - name: Install dependencies
          run: |
            python -m pip install --upgrade pip
            pip install -e .
            pip install pytest pytest-cov
        
        - name: Run tests
          run: |
            pytest tests/ -v --cov=tyxonq

.. note::
   This guide is based on TyxonQ's actual architecture and APIs.
   Always refer to the current codebase for the most up-to-date interfaces.

.. seealso::
   
   - :doc:`architecture_overview` - TyxonQ architecture
   - :doc:`contributing` - Contribution guidelines
   - :doc:`custom_devices` - Device development pyramid approach:

.. code-block:: text

    ┌────────────────────────────────────────────────────────┐
    │                  End-to-End Tests                       │
    │              (Few, High-level)                        │
    └────────────────────────────────────────────────────────┘
                            ┌──────────────────────────────────────────────────────────────────────────── ┐
                            │                  Integration Tests                              │
                            │               (Some, Component-level)                          │
                            └──────────────────────────────────────────────────────────────────────────── ┘
                    ┌────────────────────────────────────────────────────────────────────────────────────────── ┐
                    │                          Unit Tests                                    │
                    │                      (Many, Function-level)                           │
                    └────────────────────────────────────────────────────────────────────────────────────────── ┘

Unit Testing
===========

Test Framework Setup
-------------------

.. code-block:: python

    import unittest
    import numpy as np
    from tyxonq.core import Circuit, QuantumState
    from tyxonq.devices import StatevectorSimulator
    
    class TestQuantumCircuit(unittest.TestCase):
        """Unit tests for quantum circuits"""
        
        def setUp(self):
            """Set up test fixtures"""
            self.circuit = Circuit(2)
            self.simulator = StatevectorSimulator()
        
        def tearDown(self):
            """Clean up after tests"""
            self.circuit = None
            self.simulator = None
        
        def test_circuit_creation(self):
            """Test circuit creation"""
            self.assertEqual(self.circuit.num_qubits, 2)
            self.assertEqual(len(self.circuit.gates), 0)
        
        def test_gate_addition(self):
            """Test adding gates to circuit"""
            self.circuit.h(0)
            self.circuit.cnot(0, 1)
            
            self.assertEqual(len(self.circuit.gates), 2)
            self.assertEqual(self.circuit.gates[0].name, 'h')
            self.assertEqual(self.circuit.gates[1].name, 'cnot')
        
        def test_bell_state_creation(self):
            """Test Bell state creation and measurement"""
            # Create Bell state
            self.circuit.h(0)
            self.circuit.cnot(0, 1)
            self.circuit.measure_all()
            
            # Execute circuit
            result = self.simulator.execute(self.circuit, shots=1000)
            
            # Check Bell state statistics
            counts = result.counts
            total_counts = sum(counts.values())
            
            # Should only have |00> and |11> states
            allowed_states = {'00', '11'}
            for state in counts.keys():
                self.assertIn(state, allowed_states)
            
            # Check roughly equal probabilities
            if '00' in counts and '11' in counts:
                prob_00 = counts['00'] / total_counts
                prob_11 = counts['11'] / total_counts
                self.assertAlmostEqual(prob_00, 0.5, delta=0.1)
                self.assertAlmostEqual(prob_11, 0.5, delta=0.1)

Quantum-Specific Testing
-----------------------

.. code-block:: python

    import numpy as np
    from numpy.testing import assert_allclose
    
    class TestQuantumGates(unittest.TestCase):
        """Test quantum gate implementations"""
        
        def test_pauli_x_gate(self):
            """Test Pauli-X gate matrix"""
            from tyxonq.core.gates import X
            
            x_gate = X(0)
            expected_matrix = np.array([[0, 1], [1, 0]])
            
            assert_allclose(x_gate.to_matrix(), expected_matrix)
        
        def test_hadamard_gate(self):
            """Test Hadamard gate matrix"""
            from tyxonq.core.gates import H
            
            h_gate = H(0)
            expected_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            
            assert_allclose(h_gate.to_matrix(), expected_matrix)
        
        def test_cnot_gate(self):
            """Test CNOT gate matrix"""
            from tyxonq.core.gates import CNOT
            
            cnot_gate = CNOT(0, 1)
            expected_matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ])
            
            assert_allclose(cnot_gate.to_matrix(), expected_matrix)
        
        def test_rotation_gates(self):
            """Test parameterized rotation gates"""
            from tyxonq.core.gates import RX, RY, RZ
            
            theta = np.pi / 4
            
            # Test RX gate
            rx_gate = RX(0, theta)
            expected_rx = np.array([
                [np.cos(theta/2), -1j*np.sin(theta/2)],
                [-1j*np.sin(theta/2), np.cos(theta/2)]
            ])
            assert_allclose(rx_gate.to_matrix(), expected_rx)

Mocking and Test Doubles
-----------------------

.. code-block:: python

    from unittest.mock import Mock, MagicMock, patch
    
    class TestDeviceIntegration(unittest.TestCase):
        """Test device integration with mocking"""
        
        def test_remote_device_execution(self):
            """Test remote device execution with mocked API"""
            from tyxonq.devices import RemoteDevice
            
            # Mock HTTP responses
            mock_response = Mock()
            mock_response.json.return_value = {
                'job_id': 'test-job-123',
                'status': 'completed',
                'results': {'counts': {'00': 500, '11': 500}}
            }
            mock_response.raise_for_status.return_value = None
            
            with patch('requests.post', return_value=mock_response), \
                 patch('requests.get', return_value=mock_response):
                
                device = RemoteDevice('http://test-api.com', 'test-key', 'test-device')
                
                circuit = Circuit(2)
                circuit.h(0)
                circuit.cnot(0, 1)
                circuit.measure_all()
                
                result = device.execute(circuit, shots=1000)
                
                self.assertEqual(result.counts, {'00': 500, '11': 500})
        
        def test_optimizer_with_mock_objective(self):
            """Test optimizer with mocked objective function"""
            from tyxonq.optimization import SPSA
            
            # Create mock objective function
            mock_objective = Mock(side_effect=[
                1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05
            ])
            
            optimizer = SPSA(max_iterations=6)
            result = optimizer.minimize(mock_objective, [0.5, 0.5])
            
            # Verify objective was called multiple times
            self.assertGreater(mock_objective.call_count, 6)
            
            # Verify convergence
            self.assertLess(result.optimal_value, 0.1)

Integration Testing
==================

Component Integration
--------------------

.. code-block:: python

    class TestCircuitCompilation(unittest.TestCase):
        """Test circuit compilation pipeline"""
        
        def test_optimization_pipeline(self):
            """Test complete optimization pipeline"""
            from tyxonq.compiler import CompilerPipeline
            from tyxonq.devices import IBMDevice
            
            # Create circuit with redundant gates
            circuit = Circuit(3)
            circuit.h(0)
            circuit.x(1)
            circuit.x(1)  # Redundant - should be removed
            circuit.cnot(0, 1)
            circuit.cnot(0, 2)
            
            # Mock device with coupling constraints
            mock_device = Mock()
            mock_device.get_coupling_map.return_value = [(0, 1), (1, 2)]
            mock_device.get_native_gates.return_value = ['x', 'sx', 'rz', 'cnot']
            
            # Compile circuit
            compiler = CompilerPipeline(optimization_level=2)
            compiled_circuit = compiler.compile(circuit, mock_device)
            
            # Verify optimizations
            self.assertLess(len(compiled_circuit.gates), len(circuit.gates))
            
            # Verify all gates are native
            native_gates = mock_device.get_native_gates()
            for gate in compiled_circuit.gates:
                self.assertIn(gate.name, native_gates)
        
        def test_vqe_algorithm_integration(self):
            """Test VQE algorithm integration"""
            from tyxonq.algorithms import VQE
            from tyxonq.optimization import COBYLA
            from tyxonq.chemistry import H2Molecule
            
            # Create simple molecule
            molecule = H2Molecule(bond_length=0.74)
            hamiltonian = molecule.get_hamiltonian()
            
            # Create VQE instance
            optimizer = COBYLA(max_iterations=10)
            vqe = VQE(
                hamiltonian=hamiltonian,
                ansatz='hardware_efficient',
                optimizer=optimizer,
                device=StatevectorSimulator()
            )
            
            # Run VQE
            result = vqe.run()
            
            # Verify result structure
            self.assertIsNotNone(result.optimal_energy)
            self.assertIsNotNone(result.optimal_parameters)
            self.assertLessEqual(result.num_iterations, 10)
            
            # Verify energy is reasonable for H2
            self.assertLess(result.optimal_energy, 0)  # Binding energy

End-to-End Testing
=================

Workflow Testing
---------------

.. code-block:: python

    class TestCompleteWorkflows(unittest.TestCase):
        """End-to-end workflow tests"""
        
        def test_quantum_chemistry_workflow(self):
            """Test complete quantum chemistry workflow"""
            import tyxonq as tq
            
            # Define molecule
            molecule = tq.chemistry.Molecule(
                atoms=['H', 'H'],
                coordinates=[[0, 0, 0], [0, 0, 0.74]],
                basis='sto-3g'
            )
            
            # Build Hamiltonian
            hamiltonian = molecule.get_electronic_hamiltonian()
            
            # Create ansatz
            ansatz = tq.ansatz.UCCSD(molecule.num_electrons, molecule.num_orbitals)
            
            # Set up VQE
            optimizer = tq.optimization.SPSA(max_iterations=50)
            device = tq.devices.StatevectorSimulator()
            
            vqe = tq.algorithms.VQE(
                hamiltonian=hamiltonian,
                ansatz=ansatz,
                optimizer=optimizer,
                device=device
            )
            
            # Run calculation
            result = vqe.run()
            
            # Verify result
            self.assertIsInstance(result.optimal_energy, float)
            self.assertLess(result.optimal_energy, -1.1)  # H2 ground state energy
            
        def test_optimization_workflow(self):
            """Test optimization workflow"""
            import tyxonq as tq
            
            # Define QAOA problem
            graph = [(0, 1), (1, 2), (2, 3), (3, 0)]  # 4-node cycle
            
            # Create QAOA circuit
            qaoa = tq.algorithms.QAOA(
                graph=graph,
                layers=3,
                optimizer=tq.optimization.COBYLA(),
                device=tq.devices.StatevectorSimulator()
            )
            
            # Run optimization
            result = qaoa.run()
            
            # Verify result structure
            self.assertIsNotNone(result.optimal_parameters)
            self.assertIsNotNone(result.optimal_value)
            self.assertGreater(len(result.history), 0)

Performance Testing
==================

Benchmarking
-----------

.. code-block:: python

    import time
    import psutil
    import numpy as np
    
    class TestPerformance(unittest.TestCase):
        """Performance and benchmarking tests"""
        
        def test_statevector_simulation_performance(self):
            """Test statevector simulation performance"""
            from tyxonq.devices import StatevectorSimulator
            
            device = StatevectorSimulator()
            
            # Test different circuit sizes
            qubit_counts = [10, 15, 20]
            times = []
            
            for num_qubits in qubit_counts:
                circuit = Circuit(num_qubits)
                
                # Add random gates
                np.random.seed(42)
                for _ in range(num_qubits * 10):
                    qubit = np.random.randint(0, num_qubits)
                    circuit.h(qubit)
                
                # Time execution
                start_time = time.time()
                result = device.execute(circuit, shots=1000)
                end_time = time.time()
                
                execution_time = end_time - start_time
                times.append(execution_time)
                
                # Verify result
                self.assertIsNotNone(result.counts)
            
            # Verify reasonable scaling
            for i in range(1, len(times)):
                # Each doubling should be less than 4x slower (2^n scaling)
                scaling_factor = times[i] / times[i-1]
                self.assertLess(scaling_factor, 8.0)
        
        def test_memory_usage(self):
            """Test memory usage during simulation"""
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Create large circuit
            circuit = Circuit(20)
            for i in range(20):
                circuit.h(i)
            
            device = StatevectorSimulator()
            result = device.execute(circuit)
            
            peak_memory = process.memory_info().rss
            memory_increase = peak_memory - initial_memory
            
            # Clean up
            del circuit, device, result
            gc.collect()
            
            final_memory = process.memory_info().rss
            memory_cleaned = initial_memory - final_memory
            
            # Verify reasonable memory usage (< 1GB for 20 qubits)
            self.assertLess(memory_increase, 1024 * 1024 * 1024)

Continuous Integration
=====================

Test Configuration
-----------------

.. code-block:: yaml

    # .github/workflows/tests.yml
    name: Tests
    
    on:
      push:
        branches: [ main, develop ]
      pull_request:
        branches: [ main ]
    
    jobs:
      test:
        runs-on: ubuntu-latest
        strategy:
          matrix:
            python-version: [3.8, 3.9, '3.10', 3.11]
        
        steps:
        - uses: actions/checkout@v3
        
        - name: Set up Python ${{ matrix.python-version }}
          uses: actions/setup-python@v3
          with:
            python-version: ${{ matrix.python-version }}
        
        - name: Install dependencies
          run: |
            python -m pip install --upgrade pip
            pip install -e .[dev]
        
        - name: Run unit tests
          run: |
            python -m pytest tests/unit/ -v --cov=tyxonq --cov-report=xml
        
        - name: Run integration tests
          run: |
            python -m pytest tests/integration/ -v
        
        - name: Upload coverage to Codecov
          uses: codecov/codecov-action@v3
          with:
            file: ./coverage.xml

Test Organization
================

Directory Structure
------------------

.. code-block:: text

    tests/
    ├── unit/
    │   ├── test_core.py
    │   ├── test_devices.py
    │   ├── test_compiler.py
    │   └── test_optimization.py
    ├── integration/
    │   ├── test_vqe_workflow.py
    │   ├── test_qaoa_workflow.py
    │   └── test_device_integration.py
    ├── e2e/
    │   ├── test_chemistry_workflows.py
    │   └── test_optimization_workflows.py
    ├── performance/
    │   ├── test_benchmarks.py
    │   └── test_memory_usage.py
    └── conftest.py

Best Practices
=============

1. **Test Naming**: Use descriptive test names that explain what is being tested
2. **Test Independence**: Each test should be independent and repeatable
3. **Data-Driven Testing**: Use parameterized tests for multiple scenarios
4. **Assertion Messages**: Include meaningful assertion messages
5. **Test Coverage**: Aim for high test coverage but focus on critical paths
6. **Performance Monitoring**: Include performance regression tests

.. note::
   This guide provides comprehensive testing strategies for TyxonQ development.
   Regular testing and continuous integration ensure code quality and reliability.

.. seealso::
   
   - :doc:`contributing` - Contribution guidelines
   - :doc:`architecture_overview` - TyxonQ architecture
   - :doc:`extending_tyxonq` - Framework extension guide
