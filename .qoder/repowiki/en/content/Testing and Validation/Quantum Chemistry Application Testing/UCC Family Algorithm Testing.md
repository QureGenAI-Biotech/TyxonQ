# UCC Family Algorithm Testing

<cite>
**Referenced Files in This Document**   
- [test_uccsd_device_smoke.py](file://tests_applications_chem/test_uccsd_device_smoke.py)
- [test_ucc_device_runtime_smoke.py](file://tests_applications_chem/test_ucc_device_runtime_smoke.py)
- [uccsd.py](file://src/tyxonq/applications/chem/algorithms/uccsd.py)
- [ucc.py](file://src/tyxonq/applications/chem/algorithms/ucc.py)
- [ucc_device_runtime.py](file://src/tyxonq/applications/chem/runtimes/ucc_device_runtime.py)
- [hamiltonian_builders.py](file://src/tyxonq/applications/chem/chem_libs/hamiltonians_chem_library/hamiltonian_builders.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Smoke Test for UCCSD Device Execution](#smoke-test-for-uccsd-device-execution)
3. [UCC Device Runtime Compatibility](#ucc-device-runtime-compatibility)
4. [Hamiltonian Construction and Fermion-to-Qubit Mapping](#hamiltonian-construction-and-fermion-to-qubit-mapping)
5. [Parameterized Circuit Generation](#parameterized-circuit-generation)
6. [Expected Failure Modes](#expected-failure-modes)
7. [Configuration Requirements](#configuration-requirements)
8. [Integration with Compiler Pipeline and Error Mitigation](#integration-with-compiler-pipeline-and-error-mitigation)

## Introduction
This document provides a comprehensive analysis of the UCC (Unitary Coupled Cluster) family algorithm testing framework, focusing on smoke tests that validate end-to-end execution of UCCSD implementations. The testing strategy ensures compatibility between UCC algorithms and both device-based and numeric runtimes. Key aspects include proper Hamiltonian construction, fermion-to-qubit mapping via chem_libs, and parameterized circuit generation. The document also covers expected failure modes, configuration requirements for backend selection, shot settings, gradient computation, and integration points with the compiler pipeline and postprocessing error mitigation.

**Section sources**
- [test_uccsd_device_smoke.py](file://tests_applications_chem/test_uccsd_device_smoke.py#L1-L40)
- [test_ucc_device_runtime_smoke.py](file://tests_applications_chem/test_ucc_device_runtime_smoke.py#L1-L13)

## Smoke Test for UCCSD Device Execution

The `test_uccsd_device_smoke.py` file contains a smoke test that verifies the end-to-end execution of the UCCSD algorithm from molecular input to energy convergence on quantum devices. This test uses the H2 molecule as a simple example to validate the basic functionality of the UCCSD implementation.

The test begins by importing the necessary modules and defining a helper function `_has_pyscf()` to check if the PySCF library is installed. If PySCF is not available, the test is skipped. The main test function `test_uccsd_energy_device_h2_smoke()` creates a UCCSD instance using the predefined H2 molecule and evaluates the energy using the device chain API with a simulator backend.

Key aspects of this test include:
- Energy evaluation via device chain API using counts-based measurement on a simulator
- Gradient computation using the parameter-shift method through the `energy_and_grad` function
- Kernel execution with device engine configuration to verify optimization convergence
- Basic sanity checks to ensure the computed energy is a finite float within a reasonable chemical range

The test demonstrates the complete workflow from molecular input to energy convergence, ensuring that the UCCSD implementation can successfully execute on quantum devices.

**Section sources**
- [test_uccsd_device_smoke.py](file://tests_applications_chem/test_uccsd_device_smoke.py#L1-L40)
- [uccsd.py](file://src/tyxonq/applications/chem/algorithms/uccsd.py#L17-L229)
- [ucc.py](file://src/tyxonq/applications/chem/algorithms/ucc.py#L51-L582)

## UCC Device Runtime Compatibility

The `test_ucc_device_runtime_smoke.py` file validates the compatibility of UCC algorithms with both device-based and numeric runtimes. This test focuses on the `UCCDeviceRuntime` class, which provides a device runtime for UCC energy and gradient calculations via counts with a parameterized ansatz.

The test creates a simple 2-qubit Hamiltonian (H = Z0 + Z1) and initializes a `UCCDeviceRuntime` instance with this Hamiltonian. It then computes the energy using the device runtime with a simulator backend and verifies that the result is a finite float.

The `UCCDeviceRuntime` class supports:
- HF initial state preparation
- Configurable excitation operators and parameter mappings
- Parameter-shift gradient computation (using π/2 shift)
- Precomputation and caching of measurement groupings to optimize performance across multiple energy and gradient evaluations

The runtime groups qubit operator terms once per instance and reuses these groupings across energy and gradient evaluations. It also caches measurement prefixes to avoid redundant computation. This design ensures efficient execution of UCC algorithms on quantum devices.

**Section sources**
- [test_ucc_device_runtime_smoke.py](file://tests_applications_chem/test_ucc_device_runtime_smoke.py#L1-L13)
- [ucc_device_runtime.py](file://src/tyxonq/applications/chem/runtimes/ucc_device_runtime.py#L0-L304)

## Hamiltonian Construction and Fermion-to-Qubit Mapping

Proper Hamiltonian construction is critical for accurate UCC calculations. The `hamiltonian_builders.py` module in the `chem_libs` package provides functions for constructing Hamiltonians from molecular integrals and mapping them to qubit operators.

Key functions include:
- `get_integral_from_hf`: Extracts one- and two-electron integrals from a Hartree-Fock calculation
- `get_hop_from_integral`: Constructs a FermionOperator from molecular integrals
- `get_h_sparse_from_integral`: Converts FermionOperator to sparse matrix representation
- `get_h_fcifunc_from_integral`: Creates a CI-space Hamiltonian apply function using PySCF

The module supports multiple fermion-to-qubit mappings:
- Jordan-Wigner transformation for "fermion" and "qubit" modes
- Hard-core boson (HCB) mapping for "hcb" mode

The Hamiltonian construction process involves:
1. Computing molecular integrals in the atomic orbital basis
2. Transforming to the molecular orbital basis using the Hartree-Fock coefficients
3. Applying active space approximation if specified
4. Mapping the fermionic Hamiltonian to qubit operators using the selected encoding

This modular approach allows for flexible Hamiltonian construction and supports various quantum chemistry methods.

**Section sources**
- [hamiltonian_builders.py](file://src/tyxonq/applications/chem/chem_libs/hamiltonians_chem_library/hamiltonian_builders.py#L0-L297)

## Parameterized Circuit Generation

The UCC algorithm generates parameterized quantum circuits based on excitation operators derived from the molecular Hamiltonian. The circuit generation process is handled by the `build_ucc_circuit` function in the `circuits_library`, which is abstracted through the `UCCDeviceRuntime` and `UCCNumericRuntime` classes.

Key aspects of parameterized circuit generation include:
- Excitation operator selection based on initial amplitude guesses (MP2, CCSD, or FE)
- Operator screening and sorting to reduce circuit complexity
- Support for different excitation types (single, double, or both)
- Configuration options for circuit decomposition and Trotterization

The `UCCSD` class inherits from the base `UCC` class and implements specific methods for generating UCCSD ansatz circuits. The `get_ex_ops` method combines single and double excitation operators, applying screening and sorting based on the initial amplitude guesses. The `pick_and_sort` method filters out excitation operators with amplitudes below a threshold and sorts the remaining operators by amplitude magnitude.

This approach ensures that the generated circuits are both chemically meaningful and computationally efficient.

**Section sources**
- [uccsd.py](file://src/tyxonq/applications/chem/algorithms/uccsd.py#L17-L229)
- [ucc.py](file://src/tyxonq/applications/chem/algorithms/ucc.py#L51-L582)

## Expected Failure Modes

Several failure modes can occur during UCC algorithm execution, and the testing framework includes checks to identify these issues:

1. **Non-convergent optimizers**: The optimization process may fail to converge due to poor initial parameter guesses, insufficient maximum iterations, or inappropriate optimization tolerances. The `kernel` method in the `UCC` class includes default options for the L-BFGS-B optimizer, but these may need adjustment for specific molecules.

2. **Incorrect excitation operators**: Errors in excitation operator generation can lead to incorrect ansatz circuits. This can occur due to:
   - Incorrect active space specification
   - Errors in molecular orbital coefficient transformation
   - Bugs in excitation operator screening and sorting

3. **Hamiltonian construction errors**: Issues in Hamiltonian construction can result in incorrect energy calculations. Common problems include:
   - Incorrect integral transformation from atomic to molecular orbital basis
   - Errors in fermion-to-qubit mapping
   - Incorrect handling of nuclear repulsion energy

4. **Runtime configuration issues**: Misconfiguration of runtime parameters can cause execution failures:
   - Invalid backend selection
   - Insufficient shot counts for accurate expectation value estimation
   - Incompatible gradient computation methods

The smoke tests include basic sanity checks to detect these failure modes, such as verifying that computed energies are finite floats within expected chemical ranges.

**Section sources**
- [test_uccsd_device_smoke.py](file://tests_applications_chem/test_uccsd_device_smoke.py#L1-L40)
- [test_ucc_device_runtime_smoke.py](file://tests_applications_chem/test_ucc_device_runtime_smoke.py#L1-L13)
- [uccsd.py](file://src/tyxonq/applications/chem/algorithms/uccsd.py#L17-L229)

## Configuration Requirements

Proper configuration is essential for successful UCC algorithm execution. Key configuration requirements include:

### Backend Selection
The UCC algorithms support multiple backends through the `provider` and `device` parameters:
- **Simulator backends**: "statevector", "density_matrix", "matrix_product_state"
- **Hardware backends**: IBM, TyxonQ, and other quantum hardware providers

### Shot Settings
Shot configuration affects the accuracy of expectation value estimation:
- **Device runtime**: Default shots=2048 for hardware execution, shots=0 for analytic simulation
- **Numeric runtime**: Shots are ignored as calculations are exact

### Gradient Computation
Gradient methods are configurable through the `grad` parameter:
- **Parameter-shift**: Default method using finite differences with π/2 shifts
- **Free gradient**: Uses COBYLA optimizer without explicit gradient computation

### Classical Chemistry Configuration
The UCC algorithms integrate with classical quantum chemistry software:
- **Local execution**: Uses PySCF for Hartree-Fock, MP2, CCSD, and FCI calculations
- **Cloud execution**: Offloads classical chemistry calculations to cloud services

### Active Space Configuration
Active space approximation can be specified through:
- `active_space`: Tuple of (number of electrons, number of spatial orbitals)
- `active_orbital_indices`: List of orbital indices to include in the active space

These configuration options provide flexibility for different computational scenarios, from small molecule simulations to large-scale quantum chemistry calculations.

**Section sources**
- [uccsd.py](file://src/tyxonq/applications/chem/algorithms/uccsd.py#L17-L229)
- [ucc.py](file://src/tyxonq/applications/chem/algorithms/ucc.py#L51-L582)
- [ucc_device_runtime.py](file://src/tyxonq/applications/chem/runtimes/ucc_device_runtime.py#L0-L304)

## Integration with Compiler Pipeline and Error Mitigation

The UCC algorithms integrate with the compiler pipeline and postprocessing error mitigation through several key components:

### Compiler Integration
The `UCCDeviceRuntime` class uses the compiler API to prepare circuits for execution:
- Circuit compilation through the `compile_api` function
- Hamiltonian term grouping using `group_qubit_operator_terms`
- Measurement basis transformation and circuit optimization

### Postprocessing and Error Mitigation
The runtime integrates with postprocessing functions for expectation value computation and error mitigation:
- `apply_postprocessing` function for expectation value calculation from measurement counts
- Support for various postprocessing methods including "expval_pauli_sum"
- Error mitigation techniques such as readout error correction and zero-noise extrapolation

### Device Integration
The UCC algorithms interface with quantum devices through the devices module:
- Unified device API through `device_base.run`
- Support for multiple device types (simulators and hardware)
- Noise model integration for realistic simulation

This integration ensures that UCC calculations can leverage advanced compiler optimizations and error mitigation techniques to improve accuracy and efficiency.

**Section sources**
- [ucc_device_runtime.py](file://src/tyxonq/applications/chem/runtimes/ucc_device_runtime.py#L0-L304)
- [ucc.py](file://src/tyxonq/applications/chem/algorithms/ucc.py#L51-L582)