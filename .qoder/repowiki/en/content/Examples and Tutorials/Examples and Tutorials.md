# Examples and Tutorials

<cite>
**Referenced Files in This Document**   
- [circuit_chain_demo.py](file://examples/circuit_chain_demo.py)
- [cloud_uccsd_hea_demo.py](file://examples/cloud_uccsd_hea_demo.py)
- [vqe_extra.py](file://examples/vqe_extra.py)
- [vqe_parallel_pmap.py](file://examples/vqe_parallel_pmap.py)
- [run_circuit_on_lqcloud.py](file://examples/run_circuit_on_lqcloud.py)
- [run_circuit_on_guodun.py](file://examples/run_circuit_on_guodun.py)
- [run_circuit_on_quafu.py](file://examples/run_circuit_on_quafu.py)
- [h2o_sqd.py](file://examples/h2o_sqd.py)
- [md_qmmm_device_hardware.py](file://examples/qmmm/md_qmmm_device_hardware.py)
- [md_qmmm_sqd_device.py](file://examples/qmmm/md_qmmm_sqd_device.py)
- [md_ase_optimize_and_md.py](file://examples/qmmm/md_ase_optimize_and_md.py)
- [md_mdi_qmmm_embedded.py](file://examples/qmmm/md_mdi_qmmm_embedded.py)
- [md_openmm_pure_qm.py](file://examples/qmmm/md_openmm_pure_qm.py)
- [gradient_benchmark.py](file://examples/gradient_benchmark.py)
- [profile_chem.py](file://scripts/profile_chem.py)
- [demo_hea_homo_lumo_gap.py](file://examples/demo_hea_homo_lumo_gap.py)
- [demo_homo_lumo_gap.py](file://examples/demo_homo_lumo_gap.py)
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py)
- [uccsd.py](file://src/tyxonq/applications/chem/algorithms/uccsd.py)
</cite>

## Update Summary
**Changes Made**
- Added comprehensive hardware provider examples section covering LQCloud, Guodun, and Quafu platforms
- Integrated quantum chemistry algorithms with H2O SQD example demonstrating LUCJ sampling and selected configuration interaction
- Expanded molecular dynamics integration section with QM/MM examples including ASE, OpenMM, MDI, and device execution workflows
- Enhanced practical implementation guidance for real quantum hardware deployment
- Updated common modifications and pitfalls to include hardware-specific considerations

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Circuit Construction](#basic-circuit-construction)
3. [Quantum Chemistry Applications](#quantum-chemistry-applications)
4. [Hardware Provider Integration](#hardware-provider-integration)
5. [Quantum Chemistry Algorithms](#quantum-chemistry-algorithms)
6. [Molecular Dynamics Integration](#molecular-dynamics-integration)
7. [HOMO-LUMO Gap Analysis](#homo-lumo-gap-analysis)
8. [Variational Algorithms](#variational-algorithms)
9. [Advanced Optimization Techniques](#advanced-optimization-techniques)
10. [Next-Generation Examples](#next-generation-examples)
11. [Utility Scripts](#utility-scripts)
12. [Common Modifications and Pitfalls](#common-modifications-and-pitfalls)

## Introduction
This document provides comprehensive tutorials and examples for the TyxonQ quantum computing framework, organized by complexity and application domain. The examples demonstrate practical implementation patterns for quantum circuit construction, quantum chemistry simulations, variational algorithms, advanced optimization techniques, and integration with multiple quantum hardware providers. Each example includes step-by-step walkthroughs of key concepts, with explanations of the underlying quantum computing principles and framework-specific implementation details.

## Basic Circuit Construction

The `circuit_chain_demo.py` example demonstrates fundamental quantum circuit construction and manipulation using TyxonQ's chainable API. This example covers the complete workflow from circuit creation to execution, showcasing the framework's intuitive syntax and powerful features for quantum circuit design.

The demonstration begins with basic circuit construction using various methods: direct gate application, uppercase aliases for quantum gates, and parameterized quantum operations. The chainable API allows for seamless configuration of compilation, device execution, and postprocessing stages through method chaining. This approach provides a clear and readable way to specify the complete quantum computation pipeline.

Key features demonstrated include automatic completion of configuration options, where unspecified parameters are filled with sensible defaults, and global default configuration that can be set for consistent behavior across multiple circuit executions. The example also shows support for multiple numerical backends (NumPy and PyTorch), allowing users to choose the most appropriate computational framework for their needs.

Advanced quantum operations are illustrated, including measurement, reset, and barrier instructions, which are essential for complex quantum algorithms. Circuit composition and qubit remapping capabilities enable the construction of sophisticated circuits from simpler components, while JSON serialization provides a standardized format for circuit storage and exchange.

**Section sources**
- [circuit_chain_demo.py:1-305](file://examples/circuit_chain_demo.py#L1-L305)

## Quantum Chemistry Applications

The `cloud_uccsd_hea_demo.py` example demonstrates quantum chemistry simulations using Unitary Coupled Cluster with Singles and Doubles (UCCSD) and Hardware-Efficient Ansatz (HEA) methods. This example showcases the integration of classical computational chemistry with quantum computing, specifically using PySCF for molecular modeling.

The demonstration begins by constructing a hydrogen molecule (H₂) with specified atomic coordinates and basis set. Two quantum chemistry approaches are compared: UCCSD, which uses a chemically motivated ansatz based on electronic excitations, and HEA, which employs a hardware-efficient parameterized circuit. Both methods are executed with local baseline calculations and cloud-based Hartree-Fock (HF) integral computations, highlighting the hybrid classical-quantum workflow.

For UCCSD, the example shows how to initialize the algorithm with a molecular object and execute the kernel to obtain the ground state energy. The cloud-based approach offloads the classical HF and integral computations to cloud resources while maintaining local control over the quantum variational optimization. This hybrid model optimizes computational efficiency by leveraging cloud resources for intensive classical calculations.

The HEA implementation demonstrates a more flexible approach that can be initialized directly from a molecule object. The ansatz consists of alternating layers of entangling operations (CNOT gates in a linear chain) and single-qubit rotations (RY gates), with the number of layers determining the circuit depth and expressibility. The example shows how to specify the qubit mapping (parity, Jordan-Wigner, or Bravyi-Kitaev) which affects the number of qubits required and the structure of the resulting Hamiltonian.

**Section sources**
- [cloud_uccsd_hea_demo.py:1-57](file://examples/cloud_uccsd_hea_demo.py#L1-L57)

## Hardware Provider Integration

### LQCloud Platform Integration

The `run_circuit_on_lqcloud.py` example demonstrates running quantum circuits on the LQCloud superconducting quantum platform. This example showcases the complete workflow from circuit construction to hardware submission, including offline compilation verification and online execution modes.

The implementation begins with building a four-qubit circuit containing parameterized rotations and two-qubit entanglement operations. The example provides robust argument parsing for device selection, shot counts, and physical qubit mapping. A key feature is the dual-mode operation: offline mode for circuit validation without network access, and online mode for actual hardware execution.

The compilation process converts logical circuits to LQCloud-native format using the `to_lqcloud` compiler, enabling inspection of the final instruction set before submission. The example includes detailed metadata extraction from results, showing how to access platform-specific information like job IDs and task status.

Authentication is handled securely through environment variables (`TYXONQ_LQCLOUD_API_KEY` or `LQCLOUD_API_KEY`) with fallback to interactive input. The example demonstrates proper error handling for invalid configurations and provides clear feedback about the execution status.

**Section sources**
- [run_circuit_on_lqcloud.py:1-114](file://examples/run_circuit_on_lqcloud.py#L1-L114)

### Guodun Platform Integration

The `run_circuit_on_guodun.py` example provides minimal working examples for the Guodun quantum computing platform, supporting both Bell state generation and single-qubit X gate operations. This example emphasizes security best practices by avoiding hardcoded credentials and providing clear authentication requirements.

The implementation supports two circuit types: Bell states for two-qubit entanglement demonstrations and simple X gates for initial platform connectivity testing. The example includes comprehensive argument parsing for device selection, shot counts, experiment naming, and physical qubit specification.

A critical security feature is the requirement for the `TYXONQ_GUODUN_TOKEN` environment variable, preventing accidental credential exposure in source code. The example demonstrates safe token handling and provides helpful error messages when authentication fails.

The compilation process generates QCIS (Guodun's quantum instruction set) format, with detailed instruction counting and output inspection capabilities. The example includes both offline compilation verification and online submission modes, with the latter designed for quick connectivity testing rather than full result retrieval.

**Section sources**
- [run_circuit_on_guodun.py:1-125](file://examples/run_circuit_on_guodun.py#L1-L125)

### Quafu Platform Integration

The `run_circuit_on_quafu.py` example demonstrates integration with the BAQIS Quafu superconducting quantum cloud platform. This example focuses on the specific requirements for Quafu compatibility, including explicit QASM2 compilation and device discovery mechanisms.

The implementation begins with environment variable checking for `TYXONQ_QUAFU_TOKEN`, providing clear setup instructions for users. The example demonstrates device listing functionality, showing how to discover available quantum chips and select appropriate targets for circuit execution.

A key architectural difference is the requirement for explicit QASM2 compilation using `compile(compile_engine="qiskit", output="qasm2")`. This differs from other providers and highlights the importance of understanding platform-specific compilation requirements. The example shows how the Chain API's no-argument compile becomes a no-op when QASM2 has already been generated.

The workflow includes circuit construction, explicit compilation, device selection, and result processing with JSON formatting for easy inspection. The example serves as a template for integrating with other cloud platforms that require similar compilation steps.

**Section sources**
- [run_circuit_on_quafu.py:1-56](file://examples/run_circuit_on_quafu.py#L1-L56)

## Quantum Chemistry Algorithms

### H2O SQD Implementation

The `h2o_sqd.py` example demonstrates the Selected Configuration Interaction with Determinantal Quantum Monte Carlo (SQD) algorithm applied to water molecule calculations. This example showcases advanced quantum chemistry techniques combining Local Unitary Coupled Cluster (LUCJ) sampling with selected configuration interaction methods.

The implementation begins with active space preparation for a CAS(4,4) calculation on water molecules, including natural orbital selection using UMP2 density matrices. The example constructs Fermion operators from integrals and prepares reference wavefunctions using restricted Hartree-Fock calculations with natural orbitals.

Key features include LUCJ circuit construction with CCSD-initialized parameters, noisy sampling simulation with depolarizing noise models, and bitstring reversal for proper spin-orbital ordering. The example demonstrates the complete workflow from molecular setup through energy convergence, comparing results against exact Full Configuration Interaction (FCI) references.

The implementation includes sophisticated noise modeling at the probability level, simulating realistic quantum hardware conditions. The example shows how to handle different initialization strategies, from random parameters to CCSD-based starting points, and demonstrates the impact of noise on energy convergence.

**Section sources**
- [h2o_sqd.py:1-376](file://examples/h2o_sqd.py#L1-L376)

## Molecular Dynamics Integration

### QM/MM Device Execution

The `md_qmmm_device_hardware.py` example demonstrates quantum mechanics/molecular mechanics (QM/MM) simulations with direct quantum hardware integration. This tutorial showcases how to run QM/MM calculations on actual quantum devices using the qc_scanner interface, supporting both UCCSD and HEA methods.

The implementation provides three operational modes: numeric evaluation for exact reference calculations, device simulation for testing the complete hardware submission pipeline, and real hardware execution for production runs. The example includes comprehensive logging with cumulative timing to track progress during long-running optimizations.

Key features include electrostatic embedding with MM point charges, gradient computation for molecular geometry optimization, and automated switching between different quantum hardware providers. The example demonstrates proper shot count selection for different algorithms, with recommendations for UCCSD (2048 shots) and HEA (8192 shots) to ensure reliable convergence.

The implementation includes safety checks for sampling-based optimizations, limiting function evaluations to prevent excessive resource consumption. The example shows how to validate device execution by comparing results with numeric references and ensuring consistency across different operational modes.

**Section sources**
- [md_qmmm_device_hardware.py:1-255](file://examples/qmmm/md_qmmm_device_hardware.py#L1-L255)

### SQD Molecular Dynamics

The `md_qmmm_sqd_device.py` example extends molecular dynamics capabilities to the SQD algorithm, demonstrating sample-based quantum diagonalization for potential energy surface calculations. This tutorial covers frozen versus refresh subspace strategies, force noise analysis, and integration with various molecular dynamics engines.

The implementation begins with establishing the frozen subspace strategy, where the first frame determines the determinant subspace that remains fixed throughout the simulation. This approach ensures smooth potential energy surfaces and well-defined analytical gradients suitable for molecular dynamics integration.

Key features include comparison between frozen and refresh subspace strategies, demonstrating the significant force noise (~4.3e-3 Ha/Bohr) introduced by refresh strategies due to discontinuous potential energy surfaces. The example includes finite difference validation of analytical gradients and demonstrates proper NVE ensemble integration with energy conservation diagnostics.

The implementation supports electrostatic embedding with MM point charges, providing accurate treatment of environmental effects on quantum regions. The example shows how to integrate with PySCF's native molecular dynamics engine and demonstrates proper handling of MM forces for bidirectional coupling.

**Section sources**
- [md_qmmm_sqd_device.py:1-318](file://examples/qmmm/md_qmmm_sqd_device.py#L1-L318)

### ASE Integration

The `md_ase_optimize_and_md.py` example demonstrates integration with the Atomic Simulation Environment (ASE) ecosystem, showcasing how TyxonQ can serve as a quantum calculator for molecular structure optimization, molecular dynamics, and vibrational frequency analysis.

The implementation creates a TyxonQCalculator instance that wraps the qc_scanner interface, providing energy and force calculations for any molecular geometry. The example demonstrates BFGS structure optimization, Langevin molecular dynamics at 300K, and vibrational frequency analysis using finite difference methods.

Key features include automatic unit conversion between ASE (Angstrom, eV) and quantum chemistry units (Bohr, Hartree), efficient SCF restarts for consecutive geometries, and seamless integration with ASE's optimization and dynamics engines. The example shows proper initialization of velocities using Maxwell-Boltzmann distribution and stationary condition enforcement.

The implementation includes comprehensive validation checks, verifying energy reduction during optimization, reasonable bond lengths after relaxation, and physically meaningful vibrational frequencies. The example demonstrates how to collect and analyze trajectory data, including potential energy, kinetic energy, and instantaneous temperature monitoring.

**Section sources**
- [md_ase_optimize_and_md.py:1-186](file://examples/qmmm/md_ase_optimize_and_md.py#L1-L186)

### MDI Engine Integration

The `md_mdi_qmmm_embedded.py` example demonstrates the Message Passing Interface (MDI) protocol for distributed quantum-classical molecular dynamics simulations. This tutorial shows how to establish communication between a driver process and a TyxonQ engine process for electrostatically embedded QM/MM calculations.

The implementation establishes a two-process architecture where the driver handles MDI communication and the engine performs quantum calculations. The example demonstrates the complete MDI protocol sequence: coordinate exchange, lattice charge updates, and energy/force computation with proper unit handling (atomic units throughout).

Key features include dynamic MM charge updates during simulation, proper force decomposition between QM and MM regions, and robust error handling for process communication failures. The example shows how to implement the NLATTICE/CLATTICE/LATTICE command sequence for updating MM point charges and COORDS/ENERGY/FORCES for quantum calculations.

The implementation includes validation checks for electrostatic embedding effects, verifying that MM charges properly influence quantum energies and that forces are correctly partitioned between quantum and classical regions. The example demonstrates step-wise charge updates and energy response verification.

**Section sources**
- [md_mdi_qmmm_embedded.py:1-137](file://examples/qmmm/md_mdi_qmmm_embedded.py#L1-L137)

### OpenMM Integration

The `md_openmm_pure_qm.py` example demonstrates direct integration with OpenMM for pure quantum mechanical molecular dynamics simulations. This tutorial shows how to create OpenMM systems with TyxonQ potentials and perform velocity-Verlet integration with proper energy conservation diagnostics.

The implementation begins with topology construction using OpenMM's Topology class, defining atoms, bonds, and residue information. The example then creates a TyxonQ system using the create_tyxonq_system function, which internally constructs a TyxonQCalculator and integrates it with OpenMM's PythonForce mechanism.

Key features include proper velocity initialization with center-of-mass motion removal, energy conservation monitoring during NVE ensemble simulations, and validation against independent qc_scanner calculations. The example demonstrates careful handling of unit conversions between OpenMM (kJ/mol, nm) and quantum chemistry units (Hartree, Bohr).

The implementation includes comprehensive energy conservation analysis, identifying drift sources from integration truncation errors and quantum solver numerical noise. The example shows how to achieve sub-microhartree energy conservation with appropriate time steps and validates results against reference calculations.

**Section sources**
- [md_openmm_pure_qm.py:1-159](file://examples/qmmm/md_openmm_pure_qm.py#L1-L159)

## HOMO-LUMO Gap Analysis

The `demo_hea_homo_lumo_gap.py` and `demo_homo_lumo_gap.py` examples demonstrate the new HOMO-LUMO gap calculation functionality in TyxonQ, providing tools for analyzing electronic structure properties of molecular systems. This feature enables researchers to calculate the energy difference between the highest occupied molecular orbital (HOMO) and the lowest unoccupied molecular orbital (LUMO), which is crucial for understanding chemical reactivity, excitation energies, and material properties.

The implementation provides a unified interface for HOMO-LUMO gap calculation across different quantum chemistry algorithms, with consistent API design between UCCSD and HEA classes. The gap calculation is based on Hartree-Fock orbital energies and automatically handles both closed-shell and open-shell systems.

For UCCSD, the gap calculation is a native feature that directly accesses the Hartree-Fock results from the underlying PySCF computation. The method automatically determines HOMO and LUMO indices based on electron count for closed-shell systems, or analyzes orbital occupations for open-shell systems. Users can also manually specify orbital indices for custom analysis.

The HEA implementation delegates HOMO-LUMO gap calculation to an internal UCC object, creating a seamless experience for users who want to analyze electronic structure properties while using hardware-efficient ansatz circuits. This architectural design maintains code reuse and consistency while extending functionality to different algorithm types.

Key features demonstrated include:
- Automatic HOMO-LUMO determination based on electron count and orbital occupations
- Support for both closed-shell and open-shell systems
- Manual specification of HOMO/LUMO orbitals for custom analysis
- eV unit conversion for convenient interpretation
- Active space considerations in gap calculations
- Numerical validation against PySCF reference data

The examples show how to compare HOMO-LUMO gaps between different algorithms (HEA vs UCC) and across multiple molecules, providing insights into the accuracy and consistency of different quantum chemistry approaches. Error handling is also demonstrated, showing appropriate behavior when attempting gap calculations on HEA instances constructed from integrals rather than molecular objects.

**Section sources**
- [demo_hea_homo_lumo_gap.py:1-174](file://examples/demo_hea_homo_lumo_gap.py#L1-L174)
- [demo_homo_lumo_gap.py:1-200](file://examples/demo_homo_lumo_gap.py#L1-L200)
- [hea.py:729-788](file://src/tyxonq/applications/chem/algorithms/hea.py#L729-L788)
- [uccsd.py:1089-1219](file://src/tyxonq/applications/chem/algorithms/uccsd.py#L1089-L1219)

## Variational Algorithms

The `vqe_extra.py` example demonstrates Variational Quantum Eigensolver (VQE) implementation with a focus on hardware-efficient ansatz and parameter-shift gradient computation. This example illustrates the complete VQE workflow for estimating the ground state energy of a transverse field Ising model (TFIM)-like Hamiltonian.

The implementation uses a chain API approach where circuit configuration, device execution, and postprocessing are specified through method chaining. The hardware-efficient ansatz is constructed using the circuits library, featuring a sequence of two-qubit entangling gates (CNOT) followed by single-qubit rotations (RX, RZ). The energy estimation employs a counts-based approach, where measurements are performed in different bases to estimate the expectation values of various Hamiltonian terms.

For the TFIM Hamiltonian, two measurement settings are required: Z-basis measurements for ZZ interaction terms and X-basis measurements (after applying Hadamard gates) for X-field terms. The example shows how to construct separate circuits for each measurement basis and combine the results to compute the total energy. This approach accurately reflects the constraints of real quantum hardware, where measurements can only be performed in a single basis per circuit execution.

The parameter-shift rule is implemented for gradient computation, which is essential for variational optimization. For each parameter in the ansatz, two additional circuit evaluations are performed with the parameter shifted by ±π/2. The gradient is then computed as half the difference between these forward and backward evaluations. This method provides exact gradients in the absence of noise and is compatible with both quantum hardware and simulators.

The example also includes a direct numeric path using PyTorch autograd for comparison, demonstrating how the same variational algorithm can be implemented using exact statevector simulation with automatic differentiation. This dual approach allows users to validate their hardware-based results against exact numerical calculations and provides a smooth transition between simulation and real device execution.

**Section sources**
- [vqe_extra.py:1-196](file://examples/vqe_extra.py#L1-L196)

## Advanced Optimization Techniques

The `vqe_parallel_pmap.py` example demonstrates advanced optimization techniques using PyTorch's parallel mapping capabilities for batched parameter updates in VQE. This example showcases how to efficiently evaluate multiple parameter sets simultaneously, significantly accelerating the optimization process for variational quantum algorithms.

The implementation leverages PyTorch's functional programming interface (`torch.func`) to perform vectorized operations across batches of parameters. The core innovation is the use of `vmap` (vectorized map) and `grad` (automatic differentiation) to compute energy values and gradients for multiple parameter sets in parallel. When these functions are available, the example achieves substantial performance improvements compared to sequential evaluation.

The Hamiltonian for the transverse field Ising model is represented in a compact form using Pauli string codes, where each qubit's measurement basis is encoded as integers (0=I, 1=X, 2=Y, 3=Z). This representation allows efficient processing of multiple Pauli terms and facilitates the grouping of commuting terms for simultaneous measurement.

Two evaluation methods are provided: an exact path using statevector simulation and a counts-based path that simulates finite-shot measurements. The exact path uses direct statevector kernels to compute expectation values without sampling noise, making it ideal for gradient computation and optimization. The counts-based path simulates the behavior of real quantum devices by sampling measurement outcomes according to the Born rule, providing a more realistic assessment of algorithm performance.

The batch update function demonstrates how to compute both energy values and gradients for a batch of parameter sets simultaneously. This approach is particularly valuable for second-order optimization methods, population-based optimizers, or when evaluating the landscape of the cost function. The example shows how to handle cases where the functional API is not available by falling back to explicit loops with PyTorch's autograd, ensuring compatibility across different environments.

**Section sources**
- [vqe_parallel_pmap.py:1-164](file://examples/vqe_parallel_pmap.py#L1-L164)

## Next-Generation Examples

### JIT Compilation with Scan Optimization

The `hea_scan_jit_acc.py` example demonstrates how to reduce JIT (Just-In-Time) compilation time using scan operations for hardware-efficient ansatz circuits. This technique is particularly valuable for deep circuits with many layers, where traditional compilation approaches can become prohibitively slow.

The core idea is to use a scan operation to iteratively apply circuit layers, rather than constructing the entire circuit at once. This approach significantly reduces the size of the computational graph that needs to be compiled, leading to faster compilation times and lower memory usage. The example compares a plain implementation with a scan-based implementation, showing the performance benefits across different scan chunk sizes.

The implementation uses a loop function that applies a specified number of circuit layers to a quantum state, with the scan operation handling the iteration. By varying the number of layers processed in each scan iteration (controlled by the `each` parameter), users can trade off between compilation speed and runtime performance. Smaller chunk sizes lead to faster compilation but potentially slower execution, while larger chunks approach the performance of the plain implementation at the cost of longer compilation times.

This technique is especially effective with the PyTorch backend, where the scan methodology can outperform other backends in both compilation time and runtime performance. The example includes benchmarking functionality to measure and compare the performance of different implementations, allowing users to optimize their choice based on their specific requirements.

### Hybrid GPU Pipeline

The `hybrid_gpu_pipeline.py` example demonstrates a hybrid quantum-classical pipeline with both quantum and classical components running on GPU. This example showcases the integration of quantum circuits with classical neural networks for machine learning applications.

The implementation uses PyTorch as the backend for both quantum and classical computations, enabling seamless data transfer between quantum circuits and neural network layers. The example processes the MNIST dataset, converting images to quantum circuits through a binarization and encoding process. Each pixel value is mapped to a rotation angle in the quantum circuit, creating a quantum embedding of the classical data.

The quantum neural network layer is defined using TyxonQ's `TorchLayer`, which wraps a quantum circuit function and integrates it into a PyTorch neural network. The layer computes expectation values of Pauli-Z operators for each qubit, which are then passed to a classical fully connected layer for final classification. The entire model, including both quantum and classical components, is trained end-to-end using standard PyTorch optimization procedures.

Key features include GPU acceleration for both quantum simulation and classical neural network operations, JIT compilation for improved performance, and DLPack integration for efficient tensor transfer between frameworks. The example demonstrates how quantum circuits can be treated as differentiable layers in a deep learning pipeline, opening possibilities for hybrid quantum-classical machine learning models.

### Large-Scale Simulations

The `rem_super_large_scale.py` example demonstrates the limitations of readout error mitigation (REM) when the number of qubits becomes much larger than the inverse of the error probability. This example is particularly relevant for near-term quantum devices with significant readout errors.

The simulation models a simple scenario where all qubits are either in the |0⟩ state or |1⟩ state, and readout errors flip the measurement outcome with probability p. The example shows how the effectiveness of REM degrades as the number of qubits increases relative to 1/p, highlighting the challenges of error mitigation in large-scale quantum computations.

The implementation includes a simulator function that applies readout errors to ideal measurement results and a REM calibration process that characterizes single-qubit readout errors. The example demonstrates how to compute expectation values from mitigated measurement results and shows the accuracy of REM for different qubit counts and error probabilities.

This example serves as a cautionary demonstration of the scalability challenges in quantum error mitigation and emphasizes the need for improved quantum hardware with lower error rates or alternative error mitigation strategies for large-scale applications.

## Utility Scripts

### Gradient Benchmarking

The `gradient_benchmark.py` example provides tools for benchmarking gradient computation methods in variational quantum algorithms. This utility script allows users to compare the performance and accuracy of different gradient estimation techniques, such as parameter-shift, finite differences, and analytic methods.

The script includes functionality for measuring computation time, memory usage, and numerical accuracy of gradient calculations. It supports various circuit depths, parameter counts, and Hamiltonian complexities, enabling comprehensive performance evaluation across different scenarios. The benchmarking results help users select the most appropriate gradient method for their specific application, balancing accuracy, computational cost, and hardware constraints.

### Chemistry Profiling

The `profile_chem.py` script provides profiling capabilities for quantum chemistry applications, specifically focusing on UCCSD and HEA algorithms. This utility uses Python's cProfile module to analyze the performance of different components in quantum chemistry workflows, identifying bottlenecks and optimization opportunities.

The profiling covers various scenarios, including numeric simulations with exact statevector computation and device-based executions with finite shots. It examines both the classical preprocessing stages (such as integral computation and HF calculations) and the quantum variational optimization loops. The results are presented in a structured format, showing the cumulative time spent in different functions and highlighting the most time-consuming operations.

This profiling information is invaluable for optimizing quantum chemistry simulations, guiding decisions about algorithm selection, parameter tuning, and hardware configuration. The script can be easily adapted to profile custom implementations and compare the performance of different ansatz circuits or optimization strategies.

**Section sources**
- [gradient_benchmark.py](file://examples/gradient_benchmark.py)
- [profile_chem.py:1-120](file://scripts/profile_chem.py#L1-L120)

## Common Modifications and Pitfalls

When working with the examples in this repository, several common modifications and potential pitfalls should be considered:

1. **Backend Selection**: The choice of numerical backend (NumPy, PyTorch, CuPy) significantly impacts performance and functionality. PyTorch enables automatic differentiation and GPU acceleration but may have longer compilation times. Users should select the backend that best matches their computational requirements and hardware availability.

2. **Parameter Initialization**: The initial parameter values for variational algorithms can greatly affect convergence. While random initialization is common, domain-specific knowledge can inform better starting points. For quantum chemistry applications, MP2 amplitudes often provide good initial guesses for UCCSD parameters.

3. **Circuit Depth and Expressibility**: Deeper circuits with more layers can represent more complex quantum states but are more susceptible to noise and require more optimization parameters. Users should balance circuit expressibility with the limitations of their target hardware.

4. **Measurement Strategy**: The choice of measurement basis and grouping strategy affects both accuracy and computational cost. Commuting terms can be measured simultaneously, reducing the number of required circuit executions. However, finding optimal groupings is an NP-hard problem, and heuristic approaches are typically used.

5. **Error Mitigation Trade-offs**: Error mitigation techniques like readout error correction improve result accuracy but require additional calibration circuits and computational overhead. As demonstrated in the large-scale simulation example, these methods have scalability limits and may become impractical for very large systems.

6. **Hardware Provider Specifics**: Different quantum hardware providers have unique compilation requirements, authentication methods, and execution models. Always consult provider-specific documentation and examples when switching between platforms. Pay special attention to compilation flags, token management, and result format differences.

7. **Quantum Chemistry Algorithm Selection**: Choose algorithms based on your specific requirements: UCCSD for chemical accuracy with deeper circuits, HEA for hardware efficiency with shallower circuits, and SQD for sampling-based approaches with frozen subspace strategies. Consider the trade-offs between accuracy, circuit depth, and computational cost.

8. **Molecular Dynamics Integration**: When integrating with molecular dynamics engines, ensure proper unit conversions, force conservation, and energy conservation diagnostics. Different engines (ASE, OpenMM, MDI) have different integration schemes and require specific setup procedures.

9. **JIT Compilation Overheads**: While JIT compilation can accelerate repeated circuit executions, the initial compilation time can be substantial, especially for deep circuits. Techniques like scan operations can mitigate this issue but may introduce other trade-offs in runtime performance.

10. **HOMO-LUMO Gap Calculation Limitations**: The HOMO-LUMO gap functionality is only available for HEA instances constructed from molecular objects, not from integrals directly. Users should ensure they use the appropriate constructor method when they need electronic structure analysis.

By understanding these common considerations, users can effectively adapt the provided examples to their specific research questions and computational environments, avoiding common pitfalls and maximizing the effectiveness of their quantum algorithms.

**Section sources**
- [circuit_chain_demo.py:1-305](file://examples/circuit_chain_demo.py#L1-L305)
- [cloud_uccsd_hea_demo.py:1-57](file://examples/cloud_uccsd_hea_demo.py#L1-L57)
- [vqe_extra.py:1-196](file://examples/vqe_extra.py#L1-L196)
- [vqe_parallel_pmap.py:1-164](file://examples/vqe_parallel_pmap.py#L1-L164)
- [run_circuit_on_lqcloud.py:1-114](file://examples/run_circuit_on_lqcloud.py#L1-L114)
- [run_circuit_on_guodun.py:1-125](file://examples/run_circuit_on_guodun.py#L1-L125)
- [run_circuit_on_quafu.py:1-56](file://examples/run_circuit_on_quafu.py#L1-L56)
- [h2o_sqd.py:1-376](file://examples/h2o_sqd.py#L1-L376)
- [md_qmmm_device_hardware.py:1-255](file://examples/qmmm/md_qmmm_device_hardware.py#L1-L255)
- [md_qmmm_sqd_device.py:1-318](file://examples/qmmm/md_qmmm_sqd_device.py#L1-L318)
- [md_ase_optimize_and_md.py:1-186](file://examples/qmmm/md_ase_optimize_and_md.py#L1-L186)
- [md_mdi_qmmm_embedded.py:1-137](file://examples/qmmm/md_mdi_qmmm_embedded.py#L1-L137)
- [md_openmm_pure_qm.py:1-159](file://examples/qmmm/md_openmm_pure_qm.py#L1-L159)
- [gradient_benchmark.py](file://examples/gradient_benchmark.py)
- [profile_chem.py:1-120](file://scripts/profile_chem.py#L1-L120)
- [demo_hea_homo_lumo_gap.py:1-174](file://examples/demo_hea_homo_lumo_gap.py#L1-L174)
- [demo_homo_lumo_gap.py:1-200](file://examples/demo_homo_lumo_gap.py#L1-L200)