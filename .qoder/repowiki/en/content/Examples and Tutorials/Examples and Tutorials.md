# Examples and Tutorials

<cite>
**Referenced Files in This Document**   
- [circuit_chain_demo.py](file://examples/circuit_chain_demo.py)
- [cloud_uccsd_hea_demo.py](file://examples/cloud_uccsd_hea_demo.py)
- [vqe_extra.py](file://examples/vqe_extra.py)
- [vqe_parallel_pmap.py](file://examples/vqe_parallel_pmap.py)
- [hea_scan_jit_acc.py](file://examples-ng/hea_scan_jit_acc.py)
- [hybrid_gpu_pipeline.py](file://examples-ng/hybrid_gpu_pipeline.py)
- [rem_super_large_scale.py](file://examples-ng/rem_super_large_scale.py)
- [gradient_benchmark.py](file://examples/gradient_benchmark.py)
- [profile_chem.py](file://scripts/profile_chem.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Circuit Construction](#basic-circuit-construction)
3. [Quantum Chemistry Applications](#quantum-chemistry-applications)
4. [Variational Algorithms](#variational-algorithms)
5. [Advanced Optimization Techniques](#advanced-optimization-techniques)
6. [Next-Generation Examples](#next-generation-examples)
7. [Utility Scripts](#utility-scripts)
8. [Common Modifications and Pitfalls](#common-modifications-and-pitfalls)

## Introduction
This document provides comprehensive tutorials and examples for the TyxonQ quantum computing framework, organized by complexity and application domain. The examples demonstrate practical implementation patterns for quantum circuit construction, quantum chemistry simulations, variational algorithms, and advanced optimization techniques. Each example includes step-by-step walkthroughs of key concepts, with explanations of the underlying quantum computing principles and framework-specific implementation details.

## Basic Circuit Construction

The `circuit_chain_demo.py` example demonstrates fundamental quantum circuit construction and manipulation using TyxonQ's chainable API. This example covers the complete workflow from circuit creation to execution, showcasing the framework's intuitive syntax and powerful features for quantum circuit design.

The demonstration begins with basic circuit construction using various methods: direct gate application, uppercase aliases for quantum gates, and parameterized quantum operations. The chainable API allows for seamless configuration of compilation, device execution, and postprocessing stages through method chaining. This approach provides a clear and readable way to specify the complete quantum computation pipeline.

Key features demonstrated include automatic completion of configuration options, where unspecified parameters are filled with sensible defaults, and global default configuration that can be set for consistent behavior across multiple circuit executions. The example also shows support for multiple numerical backends (NumPy and PyTorch), allowing users to choose the most appropriate computational framework for their needs.

Advanced quantum operations are illustrated, including measurement, reset, and barrier instructions, which are essential for complex quantum algorithms. Circuit composition and qubit remapping capabilities enable the construction of sophisticated circuits from simpler components, while JSON serialization provides a standardized format for circuit storage and exchange.

**Section sources**
- [circuit_chain_demo.py](file://examples/circuit_chain_demo.py#L1-L305)

## Quantum Chemistry Applications

The `cloud_uccsd_hea_demo.py` example demonstrates quantum chemistry simulations using Unitary Coupled Cluster with Singles and Doubles (UCCSD) and Hardware-Efficient Ansatz (HEA) methods. This example showcases the integration of classical computational chemistry with quantum computing, specifically using PySCF for molecular modeling.

The demonstration begins by constructing a hydrogen molecule (H₂) with specified atomic coordinates and basis set. Two quantum chemistry approaches are compared: UCCSD, which uses a chemically motivated ansatz based on electronic excitations, and HEA, which employs a hardware-efficient parameterized circuit. Both methods are executed with local baseline calculations and cloud-based Hartree-Fock (HF) integral computations, highlighting the hybrid classical-quantum workflow.

For UCCSD, the example shows how to initialize the algorithm with a molecular object and execute the kernel to obtain the ground state energy. The cloud-based approach offloads the classical HF and integral computations to cloud resources while maintaining local control over the quantum variational optimization. This hybrid model optimizes computational efficiency by leveraging cloud resources for intensive classical calculations.

The HEA implementation demonstrates a more flexible approach that can be initialized directly from a molecule object. The ansatz consists of alternating layers of entangling operations (CNOT gates in a linear chain) and single-qubit rotations (RY gates), with the number of layers determining the circuit depth and expressibility. The example shows how to specify the qubit mapping (parity, Jordan-Wigner, or Bravyi-Kitaev) which affects the number of qubits required and the structure of the resulting Hamiltonian.

**Section sources**
- [cloud_uccsd_hea_demo.py](file://examples/cloud_uccsd_hea_demo.py#L1-L57)

## Variational Algorithms

The `vqe_extra.py` example demonstrates Variational Quantum Eigensolver (VQE) implementation with a focus on hardware-efficient ansatz and parameter-shift gradient computation. This example illustrates the complete VQE workflow for estimating the ground state energy of a transverse field Ising model (TFIM)-like Hamiltonian.

The implementation uses a chain API approach where circuit configuration, device execution, and postprocessing are specified through method chaining. The hardware-efficient ansatz is constructed using the circuits library, featuring a sequence of two-qubit entangling gates (CNOT) followed by single-qubit rotations (RX, RZ). The energy estimation employs a counts-based approach, where measurements are performed in different bases to estimate the expectation values of various Hamiltonian terms.

For the TFIM Hamiltonian, two measurement settings are required: Z-basis measurements for ZZ interaction terms and X-basis measurements (after applying Hadamard gates) for X-field terms. The example shows how to construct separate circuits for each measurement basis and combine the results to compute the total energy. This approach accurately reflects the constraints of real quantum hardware, where measurements can only be performed in a single basis per circuit execution.

The parameter-shift rule is implemented for gradient computation, which is essential for variational optimization. For each parameter in the ansatz, two additional circuit evaluations are performed with the parameter shifted by ±π/2. The gradient is then computed as half the difference between these forward and backward evaluations. This method provides exact gradients in the absence of noise and is compatible with both quantum hardware and simulators.

The example also includes a direct numeric path using PyTorch autograd for comparison, demonstrating how the same variational algorithm can be implemented using exact statevector simulation with automatic differentiation. This dual approach allows users to validate their hardware-based results against exact numerical calculations and provides a smooth transition between simulation and real device execution.

**Section sources**
- [vqe_extra.py](file://examples/vqe_extra.py#L1-L196)

## Advanced Optimization Techniques

The `vqe_parallel_pmap.py` example demonstrates advanced optimization techniques using PyTorch's parallel mapping capabilities for batched parameter updates in VQE. This example showcases how to efficiently evaluate multiple parameter sets simultaneously, significantly accelerating the optimization process for variational quantum algorithms.

The implementation leverages PyTorch's functional programming interface (`torch.func`) to perform vectorized operations across batches of parameters. The core innovation is the use of `vmap` (vectorized map) and `grad` (automatic differentiation) to compute energy values and gradients for multiple parameter sets in parallel. When these functions are available, the example achieves substantial performance improvements compared to sequential evaluation.

The Hamiltonian for the transverse field Ising model is represented in a compact form using Pauli string codes, where each qubit's measurement basis is encoded as integers (0=I, 1=X, 2=Y, 3=Z). This representation allows efficient processing of multiple Pauli terms and facilitates the grouping of commuting terms for simultaneous measurement.

Two evaluation methods are provided: an exact path using statevector simulation and a counts-based path that simulates finite-shot measurements. The exact path uses direct statevector kernels to compute expectation values without sampling noise, making it ideal for gradient computation and optimization. The counts-based path simulates the behavior of real quantum devices by sampling measurement outcomes according to the Born rule, providing a more realistic assessment of algorithm performance.

The batch update function demonstrates how to compute both energy values and gradients for a batch of parameter sets simultaneously. This approach is particularly valuable for second-order optimization methods, population-based optimizers, or when evaluating the landscape of the cost function. The example shows how to handle cases where the functional API is not available by falling back to explicit loops with PyTorch's autograd, ensuring compatibility across different environments.

**Section sources**
- [vqe_parallel_pmap.py](file://examples/vqe_parallel_pmap.py#L1-L164)

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

**Section sources**
- [hea_scan_jit_acc.py](file://examples-ng/hea_scan_jit_acc.py#L1-L79)
- [hybrid_gpu_pipeline.py](file://examples-ng/hybrid_gpu_pipeline.py#L1-L124)
- [rem_super_large_scale.py](file://examples-ng/rem_super_large_scale.py#L1-L59)

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
- [profile_chem.py](file://scripts/profile_chem.py#L1-L120)

## Common Modifications and Pitfalls

When working with the examples in this repository, several common modifications and potential pitfalls should be considered:

1. **Backend Selection**: The choice of numerical backend (NumPy, PyTorch, CuPy) significantly impacts performance and functionality. PyTorch enables automatic differentiation and GPU acceleration but may have longer compilation times. Users should select the backend that best matches their computational requirements and hardware availability.

2. **Parameter Initialization**: The initial parameter values for variational algorithms can greatly affect convergence. While random initialization is common, domain-specific knowledge can inform better starting points. For quantum chemistry applications, MP2 amplitudes often provide good initial guesses for UCCSD parameters.

3. **Circuit Depth and Expressibility**: Deeper circuits with more layers can represent more complex quantum states but are more susceptible to noise and require more optimization parameters. Users should balance circuit expressibility with the limitations of their target hardware.

4. **Measurement Strategy**: The choice of measurement basis and grouping strategy affects both accuracy and computational cost. Commuting terms can be measured simultaneously, reducing the number of required circuit executions. However, finding optimal groupings is an NP-hard problem, and heuristic approaches are typically used.

5. **Error Mitigation Trade-offs**: Error mitigation techniques like readout error correction improve result accuracy but require additional calibration circuits and computational overhead. As demonstrated in the large-scale simulation example, these methods have scalability limits and may become impractical for very large systems.

6. **Hybrid Classical-Quantum Workflows**: When combining classical and quantum computations, careful consideration should be given to data transfer costs and synchronization. Offloading classical computations to the cloud can accelerate certain workflows but introduces network latency and potential data privacy concerns.

7. **JIT Compilation Overheads**: While JIT compilation can accelerate repeated circuit executions, the initial compilation time can be substantial, especially for deep circuits. Techniques like scan operations can mitigate this issue but may introduce other trade-offs in runtime performance.

By understanding these common considerations, users can effectively adapt the provided examples to their specific research questions and computational environments, avoiding common pitfalls and maximizing the effectiveness of their quantum algorithms.

**Section sources**
- [circuit_chain_demo.py](file://examples/circuit_chain_demo.py#L1-L305)
- [cloud_uccsd_hea_demo.py](file://examples/cloud_uccsd_hea_demo.py#L1-L57)
- [vqe_extra.py](file://examples/vqe_extra.py#L1-L196)
- [vqe_parallel_pmap.py](file://examples/vqe_parallel_pmap.py#L1-L164)
- [hea_scan_jit_acc.py](file://examples-ng/hea_scan_jit_acc.py#L1-L79)
- [hybrid_gpu_pipeline.py](file://examples-ng/hybrid_gpu_pipeline.py#L1-L124)
- [rem_super_large_scale.py](file://examples-ng/rem_super_large_scale.py#L1-L59)
- [gradient_benchmark.py](file://examples/gradient_benchmark.py)
- [profile_chem.py](file://scripts/profile_chem.py#L1-L120)