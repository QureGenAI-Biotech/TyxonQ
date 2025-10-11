# Utility Scripts

<cite>
**Referenced Files in This Document**   
- [diag_pyscf_numeric_tyxonq.py](file://scripts/diag_pyscf_numeric_tyxonq.py)
- [numeric_vs_device_convergence_study.py](file://scripts/numeric_vs_device_convergence_study.py)
- [profile_chem.py](file://scripts/profile_chem.py)
- [uccsd.py](file://src/tyxonq/applications/chem/algorithms/uccsd.py)
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py)
- [hamiltonian_builders.py](file://src/tyxonq/applications/chem/chem_libs/hamiltonians_chem_library/hamiltonian_builders.py)
- [pyscf_civector.py](file://src/tyxonq/applications/chem/chem_libs/quantum_chem_library/pyscf_civector.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [PySCF Numeric Diagnostics](#pyscf-numeric-diagnostics)
3. [Numeric vs Device Convergence Study](#numeric-vs-device-convergence-study)
4. [Chemical Simulation Profiling](#chemical-simulation-profiling)
5. [Integration into Development Lifecycle](#integration-into-development-lifecycle)
6. [Conclusion](#conclusion)

## Introduction
The TyxonQ framework provides a suite of utility scripts designed to support development, testing, and performance analysis of quantum chemistry simulations. These scripts serve as critical tools for validating algorithm correctness, benchmarking performance across different execution paths, and identifying computational bottlenecks. The three primary utility scripts—`diag_pyscf_numeric_tyxonq.py`, `numeric_vs_device_convergence_study.py`, and `profile_chem.py`—address distinct aspects of the quantum computing workflow, from numerical validation to convergence analysis and performance profiling. These utilities are essential for ensuring the reliability and efficiency of quantum algorithms, particularly in the context of variational quantum eigensolvers (VQE) and unitary coupled cluster (UCC) methods. By providing detailed diagnostics, convergence comparisons, and performance benchmarks, these scripts enable developers and researchers to make informed decisions during algorithm design, implementation, and optimization phases.

## PySCF Numeric Diagnostics

The `diag_pyscf_numeric_tyxonq.py` script provides comprehensive numerical diagnostics by interfacing with PySCF to validate the correctness of quantum chemistry simulations. This script establishes a bridge between classical quantum chemistry calculations (via PySCF) and quantum circuit simulations (via TyxonQ), enabling direct comparison of results. The core functionality revolves around constructing a full configuration interaction (FCI) Hamiltonian from molecular integrals and comparing energy calculations and gradients between the classical PySCF implementation and the quantum circuit-based approach.

The script begins by setting up a UCCSD (Unitary Coupled Cluster Singles and Doubles) instance for the H4 molecule, extracting molecular integrals through the `get_integral_from_hf` function. It then constructs an FCI Hamiltonian function using PySCF's direct spin-1 module, which serves as the reference for subsequent comparisons. The diagnostic process involves several key steps: calculating the CI vector from circuit parameters using `get_civector_pyscf`, computing the CI energy through matrix-vector multiplication, and comparing analytical gradients obtained via automatic differentiation with those from finite difference methods.

Critical diagnostic outputs include the number of qubits, electron configuration, excitation operators, and parameter mappings used in the UCCSD ansatz. The script also computes and displays the top configuration interaction coefficients, providing insight into the dominant electronic configurations. Additionally, it calculates the Hartree-Fock baseline energy in the CI space, serving as a reference point for assessing correlation energy. The agreement between the CI energy calculated through the quantum circuit approach (`E_ci`) and the analytical gradient energy (`E_ag`) serves as a primary validation metric, with exact agreement expected in the absence of numerical errors.

This diagnostic capability is essential for verifying the mathematical correctness of the quantum circuit implementation against well-established classical quantum chemistry methods. By ensuring that the quantum circuit reproduces the same energy and gradient values as the classical FCI calculation, developers can have confidence in the accuracy of their quantum algorithm implementation before proceeding to more complex analyses or hardware execution.

**Section sources**
- [diag_pyscf_numeric_tyxonq.py](file://scripts/diag_pyscf_numeric_tyxonq.py#L1-L130)
- [hamiltonian_builders.py](file://src/tyxonq/applications/chem/chem_libs/hamiltonians_chem_library/hamiltonian_builders.py#L42-L64)
- [pyscf_civector.py](file://src/tyxonq/applications/chem/chem_libs/quantum_chem_library/pyscf_civector.py#L140-L196)
- [uccsd.py](file://src/tyxonq/applications/chem/algorithms/uccsd.py#L17-L229)

## Numeric vs Device Convergence Study

The `numeric_vs_device_convergence_study.py` script conducts a systematic comparison between numerical simulation and device-based execution paths, focusing on convergence behavior, energy consistency, and gradient accuracy. This comprehensive study is crucial for validating that the quantum circuit simulation results are consistent across different execution backends and for understanding the impact of sampling noise on optimization trajectories.

The script implements a multi-faceted comparison framework that evaluates three key aspects: energy consistency, gradient consistency, and optimization trajectory alignment. For energy consistency, it compares results between the "numeric" runtime (which uses statevector simulation) and the "device" runtime (which can use either statevector simulation or shot-based sampling). When shots=0, the device path uses analytical expectation values, allowing for exact comparison with the numeric path. When shots>0, the comparison reveals the statistical fluctuations introduced by sampling.

The convergence study employs the H4 molecule with UCCSD ansatz as the benchmark system, ensuring consistency with existing test cases. It evaluates multiple simulation engines including statevector, matrix product state (MPS), and density matrix simulators, providing a comprehensive assessment of backend consistency. For each engine, the script measures both energy and gradient agreement between numeric and device paths, with gradient differences quantified using the L2 norm.

A critical component of the study is the short optimization routine that compares convergence trajectories over 10 steps using identical initial parameters and optimization settings. This comparison reveals whether the different execution paths lead to similar optimization behavior and final energies. The script uses a simple gradient descent approach with fixed learning rate to avoid confounding factors from different optimizer implementations.

The script also includes optional comparison with third-party quantum chemistry frameworks such as TenCirChem and PennyLane, enabling cross-platform validation of results. This external validation capability strengthens confidence in the correctness of the TyxonQ implementation by demonstrating agreement with established quantum computing frameworks.

Results are exported to a JSON file (`convergence_study_results.json`) for further analysis and potential inclusion in publications. The script also generates convergence plots when matplotlib is available, providing visual comparison of optimization trajectories across different execution modes. This comprehensive approach to convergence analysis supports both development validation and scientific publication by providing detailed, reproducible comparisons between different computational approaches.

**Section sources**
- [numeric_vs_device_convergence_study.py](file://scripts/numeric_vs_device_convergence_study.py#L1-L391)
- [uccsd.py](file://src/tyxonq/applications/chem/algorithms/uccsd.py#L17-L229)
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py#L27-L439)

## Chemical Simulation Profiling

The `profile_chem.py` script provides performance profiling capabilities for quantum chemistry simulations, helping identify computational bottlenecks and optimize resource utilization. This profiling tool leverages Python's built-in `cProfile` module to collect detailed timing information about various components of the quantum chemistry workflow, with a focus on both numerical simulation and device-based execution paths.

The script defines multiple profiling functions that target specific computational scenarios:
- `run_uccsd_numeric`: Profiles UCCSD calculations using the numeric runtime with statevector simulation
- `run_uccsd_device_shots`: Profiles UCCSD calculations with shot-based sampling (counts path)
- `run_hea_device`: Profiles HEA (Hardware-Efficient Ansatz) calculations on device
- `run_hea_device_shots`: Profiles HEA calculations with shot-based sampling
- `run_hea_open_shell_shots0`: Profiles open-shell HEA calculations
- `run_uccsd_open_shell_shots0`: Profiles open-shell UCCSD calculations

Each profiling function encapsulates a complete quantum chemistry workflow, from molecule definition to energy calculation, allowing for end-to-end performance measurement. The profiling results are sorted by cumulative time, highlighting the most time-consuming functions and methods in the call stack. This information is invaluable for identifying optimization opportunities and understanding the computational cost distribution across different components of the simulation.

The script includes environment variable configuration for controlling BLAS threading (`OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS`), ensuring consistent profiling conditions across different systems. This control is essential for obtaining reproducible performance measurements that can be compared across different runs and environments.

The profiling output provides detailed information about function calls, execution times, and call frequencies, enabling developers to pinpoint specific bottlenecks in the computational workflow. For example, the results can reveal whether performance limitations stem from quantum circuit simulation, Hamiltonian construction, gradient calculation, or other components of the quantum chemistry pipeline.

This profiling capability is particularly valuable during the development and optimization phases, as it provides empirical data to guide performance improvements. By identifying the most computationally expensive components, developers can focus their optimization efforts where they will have the greatest impact on overall performance.

**Section sources**
- [profile_chem.py](file://scripts/profile_chem.py#L1-L119)
- [uccsd.py](file://src/tyxonq/applications/chem/algorithms/uccsd.py#L17-L229)
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py#L27-L439)

## Integration into Development Lifecycle

The utility scripts provided by TyxonQ integrate seamlessly into the development lifecycle, supporting activities from algorithm design through to production deployment. These tools serve as critical components in a comprehensive development and validation workflow, ensuring both correctness and performance across different stages of quantum algorithm development.

During the algorithm design phase, `diag_pyscf_numeric_tyxonq.py` provides essential validation by comparing quantum circuit results with classical quantum chemistry calculations. This early validation ensures that the mathematical formulation of the quantum algorithm is correct before proceeding to implementation. The ability to verify energy and gradient calculations against PySCF's well-established FCI implementation gives developers confidence in their algorithm design.

In the implementation and testing phase, `numeric_vs_device_convergence_study.py` becomes a primary tool for ensuring consistency across different execution paths. This script verifies that the numeric simulation path produces identical results to the device simulation path when using analytical expectations (shots=0), establishing a baseline for correctness. The convergence comparison also validates that optimization trajectories are consistent across different backends, which is essential for reliable algorithm development.

For performance optimization and scaling analysis, `profile_chem.py` provides detailed profiling information that guides optimization efforts. By identifying computational bottlenecks in both UCCSD and HEA implementations, this tool helps developers make informed decisions about algorithmic improvements, parallelization strategies, and resource allocation.

These utility scripts can be integrated into continuous integration (CI) workflows to automate validation and performance monitoring. For example:
- The PySCF diagnostic script can be run as a pre-commit check to ensure algorithmic correctness
- The convergence study can be executed nightly to detect regressions in numerical consistency
- Performance profiling can be run weekly to monitor computational efficiency trends

The JSON output from the convergence study facilitates automated result analysis and comparison across different versions, enabling quantitative assessment of improvements or regressions. This data-driven approach to development supports evidence-based decision making and ensures that changes to the codebase maintain or improve both correctness and performance.

Together, these utility scripts form a comprehensive toolkit that supports the entire development lifecycle, from initial algorithm conception to production deployment, ensuring that quantum chemistry simulations are both mathematically correct and computationally efficient.

**Section sources**
- [diag_pyscf_numeric_tyxonq.py](file://scripts/diag_pyscf_numeric_tyxonq.py#L1-L130)
- [numeric_vs_device_convergence_study.py](file://scripts/numeric_vs_device_convergence_study.py#L1-L391)
- [profile_chem.py](file://scripts/profile_chem.py#L1-L119)

## Conclusion
The utility scripts provided by TyxonQ—`diag_pyscf_numeric_tyxonq.py`, `numeric_vs_device_convergence_study.py`, and `profile_chem.py`—form a comprehensive toolkit for developing, validating, and optimizing quantum chemistry simulations. These tools address critical aspects of the quantum computing workflow, from algorithmic correctness verification to performance profiling and convergence analysis. By providing interfaces to classical quantum chemistry methods (via PySCF), enabling detailed comparison between different execution paths, and offering comprehensive performance profiling, these scripts support a rigorous development process that ensures both mathematical correctness and computational efficiency. Their integration into continuous integration workflows enables automated validation and monitoring, making them essential components of a robust quantum software development lifecycle. As quantum computing applications continue to evolve, these utility scripts will remain vital for ensuring the reliability and performance of quantum chemistry simulations across different hardware platforms and algorithmic approaches.