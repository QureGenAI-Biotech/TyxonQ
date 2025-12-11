# Quantum Chemistry Applications

<cite>
**Referenced Files in This Document**   
- [uccsd.py](file://src/tyxonq/applications/chem/algorithms/uccsd.py) - *Updated with HOMO-LUMO gap method*
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py) - *Updated with HOMO-LUMO gap method*
- [kupccgsd.py](file://src/tyxonq/applications/chem/algorithms/kupccgsd.py)
- [puccd.py](file://src/tyxonq/applications/chem/algorithms/puccd.py)
- [ucc_device_runtime.py](file://src/tyxonq/applications/chem/runtimes/ucc_device_runtime.py)
- [ucc_numeric_runtime.py](file://src/tyxonq/applications/chem/runtimes/ucc_numeric_runtime.py)
- [hea_device_runtime.py](file://src/tyxonq/applications/chem/runtimes/hea_device_runtime.py)
- [hea_numeric_runtime.py](file://src/tyxonq/applications/chem/runtimes/hea_numeric_runtime.py)
- [hamiltonian_builders.py](file://src/tyxonq/applications/chem/chem_libs/hamiltonians_chem_library/hamiltonian_builders.py)
- [molecule.py](file://src/tyxonq/applications/chem/molecule.py)
- [cloud_uccsd_hea_demo.py](file://examples/cloud_uccsd_hea_demo.py)
- [ucc.py](file://src/tyxonq/applications/chem/algorithms/ucc.py) - *Added HOMO-LUMO gap calculation functionality*
</cite>

## Update Summary
- **Added**: HOMO-LUMO gap calculation functionality to HEA and UCC/UCCSD methods
- **Updated**: Computational Chemistry Algorithms section to include HOMO-LUMO gap explanation
- **Added**: New section on HOMO-LUMO Gap Calculation
- **Updated**: Section sources to include ucc.py where HOMO-LUMO gap functionality is implemented
- **Enhanced**: Examples to demonstrate HOMO-LUMO gap usage in quantum chemistry workflows

## Table of Contents
1. [Introduction](#introduction)
2. [Molecular Representation and Hamiltonian Construction](#molecular-representation-and-hamiltonian-construction)
3. [Computational Chemistry Algorithms](#computational-chemistry-algorithms)
   - [UCC/UCCSD Method](#uccuccsd-method)
   - [HEA Method](#hea-method)
   - [k-UpCCGSD Method](#k-upccgsd-method)
   - [pUCCD Method](#puccd-method)
4. [HOMO-LUMO Gap Calculation](#homo-lumo-gap-calculation)
5. [Dual Runtime Systems](#dual-runtime-systems)
   - [Device-Based Execution](#device-based-execution)
   - [Numeric Runtime for Exact Simulation](#numeric-runtime-for-exact-simulation)
6. [End-to-End Quantum Chemistry Workflows](#end-to-end-quantum-chemistry-workflows)
7. [Configuration Options](#configuration-options)
   - [Convergence Criteria](#convergence-criteria)
   - [Orbital Optimization](#orbital-optimization)
   - [Active Space Selection](#active-space-selection)
8. [Chemical Accuracy and Classical Method Comparison](#chemical-accuracy-and-classical-method-comparison)
9. [Conclusion](#conclusion)

## Introduction

Quantum chemistry applications leverage quantum computing to solve molecular energy calculations with high precision. This document details the implementation of various quantum computational chemistry algorithms including UCC/UCCSD, HEA, k-UpCCGSD, and pUCCD methods. It explains molecule representation and Hamiltonian construction through chem_libs, describes dual runtime systems for both device-based execution and numeric simulation, and provides examples from cloud_uccsd_hea_demo.py demonstrating end-to-end workflows. The documentation also covers configuration options for convergence criteria, orbital optimization, and active space selection, along with guidance on interpreting chemical accuracy and comparing results with classical methods like PySCF. Additionally, this update introduces the HOMO-LUMO gap calculation functionality, which provides critical insights into electronic structure and chemical reactivity.

## Molecular Representation and Hamiltonian Construction

The foundation of quantum chemistry calculations lies in accurate molecular representation and Hamiltonian construction. The framework uses PySCF for molecular input handling and integral computation. Molecular data is represented using the Mole class from PySCF, which encapsulates atomic coordinates, basis sets, charge, and spin information. The Hamiltonian is constructed through a series of transformations from molecular integrals to qubit operators.

The process begins with obtaining one- and two-electron integrals from Hartree-Fock (HF) calculations. These integrals are then transformed into a fermionic Hamiltonian, which is subsequently mapped to a qubit Hamiltonian using various encoding schemes such as Jordan-Wigner, Bravyi-Kitaev, or parity mapping. The parity mapping is particularly efficient as it reduces the number of qubits by two through conservation of particle number and spin.

Hamiltonian construction is handled by the `hamiltonian_builders.py` module, which provides functions to convert molecular integrals into different Hamiltonian representations including sparse matrices, MPOs (Matrix Product Operators), and callable functions for CI (Configuration Interaction) space operations. The module supports active space approximation, allowing users to specify the number of active electrons and orbitals for more efficient calculations on larger molecules.

**Section sources**
- [hamiltonian_builders.py](file://src/tyxonq/applications/chem/chem_libs/hamiltonians_chem_library/hamiltonian_builders.py#L1-L298)
- [molecule.py](file://src/tyxonq/applications/chem/molecule.py#L1-L311)

## Computational Chemistry Algorithms

### UCC/UCCSD Method

The Unitary Coupled Cluster (UCC) method is a quantum algorithm for calculating molecular ground state energies. The UCCSD (Unitary Coupled Cluster Singles and Doubles) variant specifically includes single and double excitation operators in the ansatz. The implementation follows the standard UCC approach where the wavefunction is prepared by applying a unitary operator to a reference Hartree-Fock state.

The UCCSD class inherits from a base UCC class and implements the specific excitation operator selection for singles and doubles. It supports various initialization methods including MP2 (second-order Møller-Plesset perturbation theory), CCSD (Coupled Cluster Singles and Doubles), and zero initialization. The algorithm can screen out two-body excitations based on amplitude thresholds and sort excitations by initial guess amplitude.

Key features include:
- Automatic generation of excitation operators based on molecular symmetry
- Support for active space approximation with customizable orbital selection
- Integration with classical providers for HF and integral computation
- Configuration options for convergence criteria and optimization parameters

The UCCSD method provides high accuracy for molecular energy calculations, typically achieving chemical accuracy (within 1 kcal/mol of experimental values) for small to medium-sized molecules.

**Section sources**
- [uccsd.py](file://src/tyxonq/applications/chem/algorithms/uccsd.py#L1-L337)

### HEA Method

The Hardware-Efficient Ansatz (HEA) is designed to be compatible with near-term quantum devices by using a parameterized circuit structure that alternates between single-qubit rotations and entangling layers. The HEA implementation in this framework uses a RY-only structure with initial RY layer followed by multiple layers of CNOT chains and RY rotations.

The HEA class provides multiple construction pathways:
- Direct initialization from a molecule object
- Construction from molecular integrals and active space specification
- Integration with external circuit templates such as Qiskit's RealAmplitudes

The ansatz structure is defined by the number of layers and qubits, with the total number of parameters being (layers + 1) * n_qubits. The implementation supports various fermion-to-qubit mappings including parity, Jordan-Wigner, and Bravyi-Kitaev. The HEA method is particularly suitable for variational quantum eigensolver (VQE) algorithms and can be optimized using gradient-based or gradient-free methods.

**Section sources**
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py#L1-L659)

### k-UpCCGSD Method

The k-UpCCGSD (k-layer Unitary Pair Coupled Cluster Generalized Singles and Doubles) method is a compact ansatz that combines the efficiency of pair-coupled cluster with a layered structure. This approach reduces the number of parameters compared to full UCCSD while maintaining good accuracy for many molecular systems.

The k-UpCCGSD implementation allows users to specify the number of layers (k) and the number of different initial points for VQE calculations. The excitation operators are generalized and restricted to paired two-body excitations, making the ansatz more hardware-efficient. The method supports multiple optimization attempts with different initial parameters to improve convergence to the global minimum.

Key advantages of k-UpCCGSD include:
- Reduced circuit depth compared to UCCSD
- Fewer parameters to optimize
- Good transferability across different molecular geometries
- Efficient implementation suitable for near-term quantum devices

**Section sources**
- [kupccgsd.py](file://src/tyxonq/applications/chem/algorithms/kupccgsd.py#L1-L293)

### pUCCD Method

The paired UCCD (pUCCD) method is a specialized variant of UCC that focuses on paired double excitations, making it particularly suitable for systems with strong pairing correlations. The pUCCD implementation aligns with the new UCC base class and uses a hard-core boson (HCB) mapping for improved efficiency.

The pUCCD ansatz is constructed using Givens rotation circuits, which provide an efficient decomposition of the excitation operators. The method supports RDM (Reduced Density Matrix) calculation in both MO (Molecular Orbital) and AO (Atomic Orbital) bases, enabling detailed analysis of electronic structure. The implementation includes specialized functions for making one- and two-body RDMs that take advantage of the paired structure of the ansatz.

pUCCD is particularly effective for:
- Systems with strong electron correlation
- Superconducting materials
- Molecular dissociation curves
- Cases where full UCCSD is computationally prohibitive

**Section sources**
- [puccd.py](file://src/tyxonq/applications/chem/algorithms/puccd.py#L1-L188)

## HOMO-LUMO Gap Calculation

The framework now includes HOMO-LUMO (Highest Occupied Molecular Orbital - Lowest Unoccupied Molecular Orbital) gap calculation functionality, which provides critical insights into electronic structure, chemical reactivity, and material properties. This feature has been implemented in both the UCC and HEA classes, enabling researchers to analyze fundamental electronic properties of molecular systems.

The HOMO-LUMO gap is calculated using the Hartree-Fock orbital energies obtained during the initial quantum chemistry calculations. For closed-shell systems (spin=0), the HOMO index is determined as (n_electrons // 2) - 1 and the LUMO index as n_electrons // 2. For open-shell systems (spin≠0), the calculation uses orbital occupation analysis to identify the highest occupied and lowest unoccupied orbitals.

The implementation provides both a detailed method `get_homo_lumo_gap()` and a convenient property `homo_lumo_gap` for accessing the gap value. The detailed method returns a comprehensive dictionary containing:
- HOMO and LUMO orbital energies (in Hartree)
- HOMO-LUMO energy gap (in Hartree)
- Option to include eV conversion of the gap
- Orbital indices for HOMO and LUMO
- System type classification (closed-shell or open-shell)

For HEA instances, the HOMO-LUMO gap calculation is delegated to an internal UCC object created during molecule construction, ensuring consistency with the underlying quantum chemistry framework. This approach allows HEA to leverage the same robust HOMO-LUMO analysis while maintaining its hardware-efficient ansatz structure.

**Section sources**
- [ucc.py](file://src/tyxonq/applications/chem/algorithms/ucc.py#L1089-L1219)
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py#L729-L788)

## Dual Runtime Systems

### Device-Based Execution

The device-based execution runtime enables quantum chemistry calculations on actual quantum hardware or simulators that mimic hardware constraints. This runtime system is designed to handle the practical aspects of quantum computation including shot-based measurements, noise models, and device-specific constraints.

The device runtime for UCC methods (UCCDeviceRuntime) and HEA methods (HEADeviceRuntime) follows a common pattern:
- Construction of parameterized quantum circuits based on the chosen ansatz
- Grouping of Hamiltonian terms by measurement basis to minimize circuit executions
- Implementation of parameter shift rules for gradient computation
- Integration with quantum device providers through a unified API

Key features include:
- Support for various quantum device providers including simulators and real hardware
- Configurable shot numbers for energy estimation
- Noise modeling and error mitigation techniques
- Circuit compilation and optimization for specific device architectures
- Batch execution of multiple circuits for efficiency

The device runtime is essential for testing and validating quantum algorithms under realistic conditions and for running calculations on available quantum hardware.

**Section sources**
- [ucc_device_runtime.py](file://src/tyxonq/applications/chem/runtimes/ucc_device_runtime.py#L1-L305)
- [hea_device_runtime.py](file://src/tyxonq/applications/chem/runtimes/hea_device_runtime.py#L1-L193)

### Numeric Runtime for Exact Simulation

The numeric runtime provides exact simulation capabilities for quantum chemistry calculations, enabling precise benchmarking and algorithm development without the noise and sampling errors inherent in device-based execution. This runtime system uses statevector simulation and exact linear algebra operations to compute molecular energies and gradients.

The numeric runtime for UCC methods (UCCNumericRuntime) and HEA methods (HEANumericRuntime) offers several advantages:
- Exact energy evaluation without sampling noise
- Analytical gradient computation for efficient optimization
- Support for various numerical backends including NumPy, CuPy, and PyTorch
- Integration with classical optimization libraries such as SciPy

The numeric runtime is particularly useful for:
- Algorithm development and testing
- Convergence studies
- Comparison with classical quantum chemistry methods
- High-precision energy calculations

It serves as a reference implementation against which device-based results can be compared, helping to distinguish between algorithmic performance and hardware-induced errors.

**Section sources**
- [ucc_numeric_runtime.py](file://src/tyxonq/applications/chem/runtimes/ucc_numeric_runtime.py#L1-L293)
- [hea_numeric_runtime.py](file://src/tyxonq/applications/chem/runtimes/hea_numeric_runtime.py#L1-L101)

## End-to-End Quantum Chemistry Workflows

The framework provides comprehensive end-to-end workflows for quantum chemistry calculations, demonstrated in the cloud_uccsd_hea_demo.py example. These workflows integrate all components from molecular input to final energy calculation, showcasing the complete process of quantum computational chemistry.

A typical workflow includes:
1. Molecular definition using PySCF's Mole class or predefined molecules
2. Ansatz selection and initialization (UCCSD, HEA, etc.)
3. Runtime configuration (device or numeric)
4. Energy calculation through VQE optimization
5. Result analysis and comparison

The demo example illustrates both local and cloud-based execution paths. The local baseline uses PySCF for HF and integral computation, while the cloud path delegates these classical computations to remote servers. This hybrid approach allows researchers to leverage powerful classical computing resources for the most demanding parts of the calculation while focusing quantum resources on the variational optimization.

The workflows now support HOMO-LUMO gap analysis as part of the result interpretation phase. After optimization, users can access both the ground state energy and electronic structure properties through methods like `get_homo_lumo_gap()` and the `homo_lumo_gap` property. This enables comprehensive analysis of chemical reactivity and electronic properties alongside energy calculations.

**Section sources**
- [cloud_uccsd_hea_demo.py](file://examples/cloud_uccsd_hea_demo.py#L1-L57)

## Configuration Options

### Convergence Criteria

The framework provides extensive configuration options for controlling the convergence of quantum chemistry calculations. These include:
- Maximum number of optimization iterations
- Energy convergence thresholds
- Gradient convergence criteria
- Step size parameters for optimization algorithms
- Multiple restart options for global optimization

The convergence criteria can be set through the kernel method's options parameter, allowing fine-tuning of the optimization process. For methods like k-UpCCGSD that perform multiple optimization attempts, the number of tries can be specified to balance computational cost and solution quality.

### Orbital Optimization

Orbital optimization is supported through integration with classical quantum chemistry packages like PySCF. Users can specify whether to run HF calculations for orbital determination and can provide custom molecular orbital coefficients. The framework also supports different initialization methods for amplitudes, including MP2, CCSD, and FE (fermionic evolution), which influence the starting point of the optimization.

### Active Space Selection

Active space selection is a crucial configuration option for managing computational complexity in quantum chemistry calculations. The framework allows users to specify:
- The number of active electrons and orbitals
- Custom orbital indices for active space
- Methods for orbital sorting and selection

This enables researchers to focus computational resources on the most chemically relevant parts of the system, making calculations on larger molecules feasible. The active space approximation is particularly important for reducing the number of qubits and parameters in the quantum circuit.

**Section sources**
- [uccsd.py](file://src/tyxonq/applications/chem/algorithms/uccsd.py#L1-L337)
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py#L1-L659)
- [kupccgsd.py](file://src/tyxonq/applications/chem/algorithms/kupccgsd.py#L1-L293)
- [puccd.py](file://src/tyxonq/applications/chem/algorithms/puccd.py#L1-L188)

## Chemical Accuracy and Classical Method Comparison

Assessing chemical accuracy is essential for validating quantum chemistry methods. The framework provides tools for comparing quantum computational results with classical methods like PySCF, enabling researchers to evaluate the performance of quantum algorithms.

Key comparison metrics include:
- Total energy differences from FCI (Full Configuration Interaction) reference values
- Relative energies for reaction pathways and isomerization
- Dipole moments and other molecular properties
- Potential energy surface accuracy
- HOMO-LUMO gaps for electronic structure analysis

The implementation includes reference calculations such as HF, MP2, CCSD, and FCI through integration with PySCF, providing a comprehensive benchmarking suite. Chemical accuracy is typically defined as agreement within 1 kcal/mol (approximately 1.6 mHa) of experimental or high-level theoretical values.

The framework also supports analysis of error sources, including:
- Approximations in the ansatz (e.g., truncation of excitation operators)
- Optimization convergence issues
- Sampling noise in device-based execution
- Errors from finite shot numbers

These comparison capabilities are crucial for understanding the strengths and limitations of quantum computational methods and for guiding the development of improved algorithms.

**Section sources**
- [uccsd.py](file://src/tyxonq/applications/chem/algorithms/uccsd.py#L1-L337)
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py#L1-L659)

## Conclusion

The Quantum Chemistry Applications framework provides a comprehensive suite of tools for quantum computational chemistry, implementing state-of-the-art algorithms like UCC/UCCSD, HEA, k-UpCCGSD, and pUCCD for molecular energy calculations. The system integrates molecular representation and Hamiltonian construction through chem_libs, supports dual runtime systems for both device-based execution and exact numeric simulation, and provides end-to-end workflows demonstrated in practical examples.

Key strengths of the framework include its modular design, support for various ansatz types, integration with classical quantum chemistry software, and comprehensive configuration options for convergence criteria, orbital optimization, and active space selection. The ability to compare results with classical methods like PySCF enables rigorous assessment of chemical accuracy and guides the development of improved quantum algorithms.

This update introduces HOMO-LUMO gap calculation functionality, enhancing the framework's capabilities for electronic structure analysis. The HOMO-LUMO gap provides critical insights into chemical reactivity, material properties, and electronic transitions, making it an essential tool for quantum chemistry research. This feature is seamlessly integrated across both UCC and HEA methods, ensuring consistent analysis capabilities regardless of the chosen ansatz.

As quantum hardware continues to advance, this framework provides a solid foundation for exploring the potential of quantum computing in chemistry, from fundamental research to practical applications in materials science and drug discovery.