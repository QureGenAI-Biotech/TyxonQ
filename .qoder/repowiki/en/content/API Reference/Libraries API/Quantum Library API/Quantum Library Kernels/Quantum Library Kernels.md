# Quantum Library Kernels

<cite>
**Referenced Files in This Document**   
- [statevector.py](file://src/tyxonq/libs/quantum_library/kernels/statevector.py)
- [density_matrix.py](file://src/tyxonq/libs/quantum_library/kernels/density_matrix.py)
- [matrix_product_state.py](file://src/tyxonq/libs/quantum_library/kernels/matrix_product_state.py)
- [unitary.py](file://src/tyxonq/libs/quantum_library/kernels/unitary.py)
- [pauli.py](file://src/tyxonq/libs/quantum_library/kernels/pauli.py)
- [gates.py](file://src/tyxonq/libs/quantum_library/kernels/gates.py)
- [common.py](file://src/tyxonq/libs/quantum_library/kernels/common.py)
- [timeevolution_trotter.py](file://examples/timeevolution_trotter.py)
- [sample_value_gradient.py](file://examples/sample_value_gradient.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)

## Introduction
This document provides comprehensive API documentation for the quantum simulation kernels in TyxonQ, focusing on state representations (statevector, density matrix, matrix product state), unitary operations, Pauli algebra, and supporting utilities. The kernels are designed for integration with multiple numerical backends (NumPy, PyTorch, CuPyNumeric) and are used within device simulators for quantum circuit execution. Practical examples demonstrate time evolution and gradient computation workflows.

## Project Structure
The quantum library kernels reside in `src/tyxonq/libs/quantum_library/kernels/` and provide low-level computational primitives for quantum state manipulation. These modules are utilized by higher-level simulators in `src/tyxonq/devices/simulators/` and application examples in the `examples/` directory.

```mermaid
graph TD
subgraph "Kernels"
statevector["statevector.py"]
density_matrix["density_matrix.py"]
mps["matrix_product_state.py"]
unitary["unitary.py"]
pauli["pauli.py"]
gates["gates.py"]
common["common.py"]
end
subgraph "Simulators"
sv_sim["statevector/engine.py"]
dm_sim["density_matrix/engine.py"]
mps_sim["matrix_product_state/engine.py"]
end
subgraph "Examples"
trotter["timeevolution_trotter.py"]
gradient["sample_value_gradient.py"]
end
statevector --> sv_sim
density_matrix --> dm_sim
mps --> mps_sim
gates --> statevector
gates --> density_matrix
gates --> mps
unitary --> gates
pauli --> trotter
common --> gradient
```

**Diagram sources**
- [statevector.py](file://src/tyxonq/libs/quantum_library/kernels/statevector.py)
- [density_matrix.py](file://src/tyxonq/libs/quantum_library/kernels/density_matrix.py)
- [matrix_product_state.py](file://src/tyxonq/libs/quantum_library/kernels/matrix_product_state.py)
- [unitary.py](file://src/tyxonq/libs/quantum_library/kernels/unitary.py)
- [pauli.py](file://src/tyxonq/libs/quantum_library/kernels/pauli.py)
- [gates.py](file://src/tyxonq/libs/quantum_library/kernels/gates.py)
- [common.py](file://src/tyxonq/libs/quantum_library/kernels/common.py)
- [timeevolution_trotter.py](file://examples/timeevolution_trotter.py)
- [sample_value_gradient.py](file://examples/sample_value_gradient.py)

**Section sources**
- [statevector.py](file://src/tyxonq/libs/quantum_library/kernels/statevector.py)
- [density_matrix.py](file://src/tyxonq/libs/quantum_library/kernels/density_matrix.py)
- [matrix_product_state.py](file://src/tyxonq/libs/quantum_library/kernels/matrix_product_state.py)
- [unitary.py](file://src/tyxonq/libs/quantum_library/kernels/unitary.py)
- [pauli.py](file://src/tyxonq/libs/quantum_library/kernels/pauli.py)
- [gates.py](file://src/tyxonq/libs/quantum_library/kernels/gates.py)
- [common.py](file://src/tyxonq/libs/quantum_library/kernels/common.py)
- [timeevolution_trotter.py](file://examples/timeevolution_trotter.py)
- [sample_value_gradient.py](file://examples/sample_value_gradient.py)

## Core Components
The quantum library kernels provide five core state representations: statevector, density matrix, matrix product state (MPS), unitary, and Pauli operators. These are supported by common utilities and gate operations that enable quantum circuit simulation across different numerical backends.

**Section sources**
- [statevector.py](file://src/tyxonq/libs/quantum_library/kernels/statevector.py)
- [density_matrix.py](file://src/tyxonq/libs/quantum_library/kernels/density_matrix.py)
- [matrix_product_state.py](file://src/tyxonq/libs/quantum_library/kernels/matrix_product_state.py)
- [unitary.py](file://src/tyxonq/libs/quantum_library/kernels/unitary.py)
- [pauli.py](file://src/tyxonq/libs/quantum_library/kernels/pauli.py)

## Architecture Overview
The quantum library architecture follows a modular design where each kernel provides state-specific operations that can be integrated with any supported numerical backend. The system enables both exact simulation (statevector, density matrix) and approximate compressed-state simulation (MPS).

```mermaid
graph TD
Backend["Numerics Backend<br>(NumPy, PyTorch, CuPyNumeric)"]
subgraph "State Kernels"
SV["Statevector Kernel"]
DM["Density Matrix Kernel"]
MPS["MPS Kernel"]
Unitary["Unitary Kernel"]
Pauli["Pauli Kernel"]
end
subgraph "Utilities"
Gates["Gates Module"]
Common["Common Utilities"]
end
Backend --> SV
Backend --> DM
Backend --> MPS
Backend --> Gates
Backend --> Common
Unitary --> Gates
Pauli --> SV
Pauli --> DM
Gates --> SV
Gates --> DM
Gates --> MPS
Common --> SV
Common --> DM
Common --> MPS
subgraph "Simulators"
SVSim["Statevector Simulator"]
DMSim["Density Matrix Simulator"]
MPSSim["MPS Simulator"]
end
SV --> SVSim
DM --> DMSim
MPS --> MPSSim
```

**Diagram sources**
- [statevector.py](file://src/tyxonq/libs/quantum_library/kernels/statevector.py)
- [density_matrix.py](file://src/tyxonq/libs/quantum_library/kernels/density_matrix.py)
- [matrix_product_state.py](file://src/tyxonq/libs/quantum_library/kernels/matrix_product_state.py)
- [unitary.py](file://src/tyxonq/libs/quantum_library/kernels/unitary.py)
- [pauli.py](file://src/tyxonq/libs/quantum_library/kernels/pauli.py)
- [gates.py](file://src/tyxonq/libs/quantum_library/kernels/gates.py)
- [common.py](file://src/tyxonq/libs/quantum_library/kernels/common.py)

## Detailed Component Analysis

### Statevector Kernel Analysis
The statevector kernel provides functions for initializing and manipulating pure quantum states using the full statevector representation.

#### Statevector Class Diagram
```mermaid
classDiagram
class StatevectorKernel {
+init_statevector(num_qubits, backend)
+apply_1q_statevector(backend, state, gate2, qubit, num_qubits)
+apply_2q_statevector(backend, state, gate4, q0, q1, num_qubits)
+expect_z_statevector(state, qubit, num_qubits, backend)
}
```

**Diagram sources**
- [statevector.py](file://src/tyxonq/libs/quantum_library/kernels/statevector.py#L8-L53)

#### Statevector Operation Sequence
```mermaid
sequenceDiagram
participant User
participant Statevector
participant Backend
User->>Statevector : init_statevector(num_qubits)
Statevector->>Backend : Create |0...0⟩ state
Backend-->>Statevector : Statevector array
Statevector-->>User : Initialized state
User->>Statevector : apply_1q_statevector(gate, qubit)
Statevector->>Backend : Reshape state, einsum contraction
Backend-->>Statevector : Transformed state
Statevector-->>User : Updated state
User->>Statevector : expect_z_statevector(qubit)
Statevector->>Backend : Compute Z expectation
Backend-->>Statevector : Expectation value
Statevector-->>User : ⟨Z⟩ result
```

**Diagram sources**
- [statevector.py](file://src/tyxonq/libs/quantum_library/kernels/statevector.py#L8-L53)

**Section sources**
- [statevector.py](file://src/tyxonq/libs/quantum_library/kernels/statevector.py#L8-L53)

### Density Matrix Kernel Analysis
The density matrix kernel provides functions for initializing and manipulating mixed quantum states using the density matrix representation.

#### Density Matrix Class Diagram
```mermaid
classDiagram
class DensityMatrixKernel {
+init_density(num_qubits, backend)
+apply_1q_density(backend, rho, U, q, n)
+apply_2q_density(backend, rho, U4, q0, q1, n)
+exp_z_density(backend, rho, q, n)
}
```

**Diagram sources**
- [density_matrix.py](file://src/tyxonq/libs/quantum_library/kernels/density_matrix.py#L8-L72)

#### Density Matrix Operation Flow
```mermaid
flowchart TD
Start([Initialize Density Matrix]) --> Init["init_density(num_qubits)"]
Init --> Apply1Q["apply_1q_density(U, qubit)"]
Apply1Q --> Apply2Q["apply_2q_density(U4, q0, q1)"]
Apply2Q --> Expect["exp_z_density(qubit)"]
Expect --> End([Return Expectation])
style Start fill:#f9f,stroke:#333
style End fill:#f9f,stroke:#333
```

**Diagram sources**
- [density_matrix.py](file://src/tyxonq/libs/quantum_library/kernels/density_matrix.py#L8-L72)

**Section sources**
- [density_matrix.py](file://src/tyxonq/libs/quantum_library/kernels/density_matrix.py#L8-L72)

### Matrix Product State Kernel Analysis
The MPS kernel provides a compressed-state representation for efficient simulation of certain quantum states, particularly useful for 1D systems.

#### MPS Class Structure
```mermaid
classDiagram
class MPSState {
+tensors : List[Any]
}
class MPSKernel {
+init_product_state(num_qubits, bit)
+apply_1q(mps, U, site)
+apply_2q_nn(mps, U4, left_site, max_bond, svd_cutoff)
+apply_2q(mps, U4, q0, q1, max_bond, svd_cutoff)
+to_statevector(mps)
+bond_dims(mps)
}
MPSKernel --> MPSState
```

**Diagram sources**
- [matrix_product_state.py](file://src/tyxonq/libs/quantum_library/kernels/matrix_product_state.py#L36-L223)

#### MPS Two-Qubit Gate Flow
```mermaid
flowchart TD
Start([Apply 2Q Gate]) --> CheckNN{"Nearest Neighbor?"}
CheckNN --> |Yes| ApplyNN["apply_2q_nn(U4, left_site)"]
CheckNN --> |No| Route["Route qubits via SWAPs"]
Route --> ApplyNN
ApplyNN --> Merge["Merge site tensors θ = A ⊗ B"]
Merge --> ApplyGate["Apply U: θ' = U·θ"]
ApplyGate --> Reshape["Reshape to matrix"]
Reshape --> SVD["SVD with truncation"]
SVD --> Reconstruct["Reconstruct A', B'"]
Reconstruct --> Update["Update mps.tensors"]
Update --> End([Complete])
```

**Diagram sources**
- [matrix_product_state.py](file://src/tyxonq/libs/quantum_library/kernels/matrix_product_state.py#L101-L148)

**Section sources**
- [matrix_product_state.py](file://src/tyxonq/libs/quantum_library/kernels/matrix_product_state.py#L36-L223)

### Unitary and Pauli Kernels Analysis
The unitary and Pauli kernels provide gate definitions and Pauli algebra operations for quantum circuit construction and Hamiltonian simulation.

#### Unitary and Pauli Functionality
```mermaid
classDiagram
class UnitaryKernel {
+get_unitary(name, *params)
+_u_h()
+_u_rz(theta)
+_u_cx()
}
class PauliKernel {
+ps2xyz(ps)
+xyz2ps(xyz, n)
+pauli_string_to_matrix(ps)
+pauli_string_sum_dense(ls, weights)
+heisenberg_hamiltonian(num_qubits, edges, ...)
}
PauliKernel --> UnitaryKernel
```

**Diagram sources**
- [unitary.py](file://src/tyxonq/libs/quantum_library/kernels/unitary.py#L8-L82)
- [pauli.py](file://src/tyxonq/libs/quantum_library/kernels/pauli.py#L8-L174)

**Section sources**
- [unitary.py](file://src/tyxonq/libs/quantum_library/kernels/unitary.py#L8-L82)
- [pauli.py](file://src/tyxonq/libs/quantum_library/kernels/pauli.py#L8-L174)

### Gates and Common Utilities Analysis
The gates and common modules provide gate matrix implementations and shared utilities for quantum simulation.

#### Gates Module Structure
```mermaid
classDiagram
class GatesKernel {
+gate_h()
+gate_rz(theta)
+gate_rx(theta)
+gate_ry(theta)
+gate_cx_4x4()
+gate_cx_rank4()
+gate_cz_4x4()
+gate_x()
+gate_s()
+gate_sd()
+gate_t()
+gate_td()
+gate_rxx(theta)
+gate_ryy(theta)
+gate_rzz(theta)
+zz_matrix()
+gate_cry_4x4(theta)
+build_controlled_unitary(U, num_controls, ctrl_state)
}
```

**Diagram sources**
- [gates.py](file://src/tyxonq/libs/quantum_library/kernels/gates.py#L8-L205)

#### Common Utilities Flow
```mermaid
sequenceDiagram
participant User
participant Common
participant Backend
User->>Common : parameter_shift_gradient(energy_fn, params)
Common->>User : energy_fn(params + π/2)
User-->>Common : E+
Common->>User : energy_fn(params - π/2)
User-->>Common : E-
Common->>Common : g[i] = 0.5*(E+ - E-)
Common-->>User : Gradient array
```

**Diagram sources**
- [common.py](file://src/tyxonq/libs/quantum_library/kernels/common.py#L8-L26)

**Section sources**
- [gates.py](file://src/tyxonq/libs/quantum_library/kernels/gates.py#L8-L205)
- [common.py](file://src/tyxonq/libs/quantum_library/kernels/common.py#L8-L26)

## Dependency Analysis
The quantum library kernels have a layered dependency structure where higher-level operations depend on lower-level primitives and the numerical backend abstraction.

```mermaid
graph TD
Backend["Numerics Backend"]
Unitary --> Gates
Pauli --> Statevector
Pauli --> DensityMatrix
Gates --> Statevector
Gates --> DensityMatrix
Gates --> MPS
Common --> Statevector
Common --> DensityMatrix
Common --> MPS
Backend --> Statevector
Backend --> DensityMatrix
Backend --> MPS
Backend --> Gates
Backend --> Common
class Statevector,DensityMatrix,MPS,Gates,Common,Unitary,Pauli UnitaryClass;
classDef UnitaryClass fill:#e6f3ff,stroke:#333;
```

**Diagram sources**
- [statevector.py](file://src/tyxonq/libs/quantum_library/kernels/statevector.py)
- [density_matrix.py](file://src/tyxonq/libs/quantum_library/kernels/density_matrix.py)
- [matrix_product_state.py](file://src/tyxonq/libs/quantum_library/kernels/matrix_product_state.py)
- [unitary.py](file://src/tyxonq/libs/quantum_library/kernels/unitary.py)
- [pauli.py](file://src/tyxonq/libs/quantum_library/kernels/pauli.py)
- [gates.py](file://src/tyxonq/libs/quantum_library/kernels/gates.py)
- [common.py](file://src/tyxonq/libs/quantum_library/kernels/common.py)

**Section sources**
- [statevector.py](file://src/tyxonq/libs/quantum_library/kernels/statevector.py)
- [density_matrix.py](file://src/tyxonq/libs/quantum_library/kernels/density_matrix.py)
- [matrix_product_state.py](file://src/tyxonq/libs/quantum_library/kernels/matrix_product_state.py)
- [unitary.py](file://src/tyxonq/libs/quantum_library/kernels/unitary.py)
- [pauli.py](file://src/tyxonq/libs/quantum_library/kernels/pauli.py)
- [gates.py](file://src/tyxonq/libs/quantum_library/kernels/gates.py)
- [common.py](file://src/tyxonq/libs/quantum_library/kernels/common.py)

## Performance Considerations
Different state representations have distinct performance characteristics:

- **Statevector**: O(2^N) memory, O(2^N) per gate operation for 1Q, O(4^N) for 2Q
- **Density Matrix**: O(4^N) memory, O(8^N) per gate operation
- **MPS**: O(d*D^2) memory where D is bond dimension, O(D^3) per NN 2Q gate
- **Backend Impact**: PyTorch enables GPU acceleration, CuPyNumeric provides distributed array support, NumPy is CPU-only

The choice of representation should consider system size, entanglement structure, and available hardware resources.

## Troubleshooting Guide
Common issues and solutions:

- **Memory exhaustion**: Use MPS representation for large 1D systems with limited entanglement
- **Numerical instability**: Ensure proper normalization of states; use svd_cutoff in MPS operations
- **Backend errors**: Verify backend compatibility; use set_backend() before kernel operations
- **Gate application errors**: Check qubit indices are within bounds; ensure proper gate matrix dimensions
- **Gradient noise**: Increase shot count in sampling-based gradient estimation

**Section sources**
- [statevector.py](file://src/tyxonq/libs/quantum_library/kernels/statevector.py)
- [density_matrix.py](file://src/tyxonq/libs/quantum_library/kernels/density_matrix.py)
- [matrix_product_state.py](file://src/tyxonq/libs/quantum_library/kernels/matrix_product_state.py)
- [common.py](file://src/tyxonq/libs/quantum_library/kernels/common.py)

## Conclusion
The TyxonQ quantum library kernels provide a comprehensive set of tools for quantum state simulation with support for multiple representations and numerical backends. The modular design enables efficient implementation of quantum algorithms while maintaining flexibility for different use cases. The integration of exact and approximate methods allows users to select the appropriate trade-off between accuracy and computational resources.