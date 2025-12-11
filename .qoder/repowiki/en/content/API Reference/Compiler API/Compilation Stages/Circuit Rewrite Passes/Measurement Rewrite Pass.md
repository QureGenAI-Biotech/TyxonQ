# Measurement Rewrite Pass

<cite>
**Referenced Files in This Document**   
- [measurement.py](file://src/tyxonq/compiler/stages/rewrite/measurement.py)
- [hamiltonian_grouping.py](file://src/tyxonq/compiler/utils/hamiltonian_grouping.py)
- [hamiltonian_grouping.py](file://src/tyxonq/libs/hamiltonian_encoding/hamiltonian_grouping.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Core Functionality](#core-functionality)
3. [Integration with Hamiltonian Grouping Utilities](#integration-with-hamiltonian-grouping-utilities)
4. [Metadata Injection Process](#metadata-injection-process)
5. [Greedy Grouping Algorithm](#greedy-grouping-algorithm)
6. [Step-by-Step Example](#step-by-step-example)
7. [Role in Variational Algorithms](#role-in-variational-algorithms)
8. [Configuration and Performance Benefits](#configuration-and-performance-benefits)

## Introduction
The MeasurementRewritePass module is a critical component in TyxonQ's compiler infrastructure, designed to optimize observable measurement in quantum circuits. It achieves this by grouping compatible Pauli terms to minimize circuit configurations during measurement, thereby reducing shot scheduling overhead. This pass plays a pivotal role in variational quantum algorithms like VQE (Variational Quantum Eigensolver), where efficient energy estimation is essential. By leveraging the hamiltonian_grouping utilities, it analyzes Hamiltonian terms and generates measurement groups that are compatible with product-basis measurement constraints.

**Section sources**
- [measurement.py](file://src/tyxonq/compiler/stages/rewrite/measurement.py#L9-L88)

## Core Functionality
The MeasurementRewritePass rewrites measurement-related constructs in quantum circuits by grouping compatible measurement items. It supports two primary modes: grouping arbitrary measurement items provided via the `measurements` option or deriving them from the circuit's IR (Intermediate Representation). When no explicit measurements are provided, it scans the circuit operations for `measure_z` instructions and constructs corresponding Expectation objects. The core advantage lies in storing explicit grouping metadata—such as basis, basis_map, and wires—in `Circuit.metadata`, which enhances observability and facilitates downstream scheduling decisions.

This pass ensures that measurement groupings are product-basis-safe, meaning that overlapping qubits can be measured together only if their measurement bases are consistent. This approach maintains circuit semantics while enabling safe shot reuse across grouped terms. The implementation is extensible, allowing future enhancements such as commuting-set-based grouping or cost-aware packing strategies without modifying device-specific code.

**Section sources**
- [measurement.py](file://src/tyxonq/compiler/stages/rewrite/measurement.py#L29-L88)

## Integration with Hamiltonian Grouping Utilities
The MeasurementRewritePass integrates with the `hamiltonian_grouping` utilities to process Hamiltonian-like inputs for Pauli-sum energy calculations. It accepts two types of inputs: `hamiltonian_terms`, which is a list of tuples containing coefficients and Pauli operator sequences, and `qubit_operator`, which follows the OpenFermion QubitOperator interface. For `hamiltonian_terms`, it invokes `group_hamiltonian_pauli_terms`, while for `qubit_operator`, it uses `group_qubit_operator_terms`.

These utility functions are located in `src/tyxonq/libs/hamiltonian_encoding/hamiltonian_grouping.py` and are re-exported through `src/tyxonq/compiler/utils/hamiltonian_grouping.py`. They perform the actual grouping of Pauli terms into product-basis-compatible sets by constructing a basis tuple for each group. The grouping logic ensures that all terms within a group can be measured simultaneously under the same basis configuration, thus minimizing the number of distinct measurement settings required.

**Section sources**
- [measurement.py](file://src/tyxonq/compiler/stages/rewrite/measurement.py#L52-L68)
- [hamiltonian_grouping.py](file://src/tyxonq/compiler/utils/hamiltonian_grouping.py#L15-L20)
- [hamiltonian_grouping.py](file://src/tyxonq/libs/hamiltonian_encoding/hamiltonian_grouping.py#L11-L65)

## Metadata Injection Process
After grouping measurements and Hamiltonian terms, the MeasurementRewritePass injects metadata into the circuit's `metadata` dictionary. The primary metadata key is `measurement_groups`, which contains a list of dictionaries, each representing a measurement group. Each group includes:
- `items`: The list of measurement items (Pauli terms) in the group.
- `wires`: The sorted tuple of qubit indices involved in the group.
- `basis`: The measurement basis, currently fixed as "pauli".
- `basis_map`: A dictionary mapping qubit indices to their respective Pauli bases (X, Y, Z).
- `source`: Indicates whether the group originated from "hamiltonian" input or other measurements.

Additionally, if Hamiltonian terms are processed, the identity constant (scalar offset) is stored under `circuit.metadata["measurement_context"]["identity_const"]`. This value is crucial for accurate energy calculation, as it represents the contribution from identity terms in the Hamiltonian. The metadata structure is designed to be consumed by downstream components, particularly the shot scheduler, which uses it to plan efficient measurement execution.

**Section sources**
- [measurement.py](file://src/tyxonq/compiler/stages/rewrite/measurement.py#L70-L88)

## Greedy Grouping Algorithm
The greedy grouping algorithm implemented in `_group_measurements` processes measurement items sequentially and attempts to place each item into an existing group if no basis conflict arises. A conflict occurs when a qubit in the current measurement item is already assigned to a different basis in the target group. If no compatible group is found, a new group is created.

The algorithm operates in linear time relative to the number of measurement items, making it efficient for large-scale problems. It uses a product-basis-safe merge strategy: overlapping qubits are allowed within a group only if their measurement bases agree. This ensures that all terms in a group can be measured in a single circuit configuration. After grouping, each group is annotated with `estimated_settings` (set to 1, as a consistent product basis exists) and `estimated_shots_per_group`, which is heuristically calculated based on the number of items and qubits in the group.

**Section sources**
- [measurement.py](file://src/tyxonq/compiler/stages/rewrite/measurement.py#L117-L162)

## Step-by-Step Example
Consider a Hamiltonian with the following Pauli terms: Z₀, X₁, and Z₀Z₁. The MeasurementRewritePass processes these terms as follows:

1. **Parse Terms**: Each term is parsed into its constituent Pauli operators and associated qubits.
2. **Initialize Groups**: Start with an empty list of groups.
3. **Process Z₀**: Create the first group with basis_map {0: 'Z'} and wires (0,).
4. **Process X₁**: Attempt to merge with the first group. Since qubit 1 is not in the group and there is no basis conflict, add it to the same group, updating basis_map to {0: 'Z', 1: 'X'} and wires to (0, 1).
5. **Process Z₀Z₁**: Check against the existing group. Both qubits 0 and 1 are present, and their required bases (Z for both) match the group's basis_map. Therefore, this term is added to the same group.
6. **Final Group**: One group contains all three terms with basis_map {0: 'Z', 1: 'X'}, enabling simultaneous measurement under a single circuit configuration.

This grouping reduces the number of required measurement settings from three to one, significantly improving efficiency.

**Section sources**
- [measurement.py](file://src/tyxonq/compiler/stages/rewrite/measurement.py#L117-L162)

## Role in Variational Algorithms
The MeasurementRewritePass is essential for variational algorithms like VQE, where the expectation value of a molecular Hamiltonian must be estimated repeatedly during optimization. In VQE, the Hamiltonian is typically expressed as a sum of Pauli terms, each requiring measurement. Without grouping, each term would necessitate a separate circuit execution, leading to high overhead.

By grouping compatible terms, the pass minimizes the number of distinct measurement configurations, thereby reducing the total number of circuit executions and associated shot scheduling overhead. This optimization is critical for achieving practical runtimes on near-term quantum devices, where measurement and reset operations are time-consuming. The injected metadata, including `basis_map` and `estimated_shots_per_group`, enables the shot scheduler to allocate resources efficiently, further enhancing performance.

**Section sources**
- [measurement.py](file://src/tyxonq/compiler/stages/rewrite/measurement.py#L9-L88)

## Configuration and Performance Benefits
The MeasurementRewritePass is configured through the `hamiltonian_terms` input, which provides the Pauli-sum representation of the observable to be measured. This input is typically generated by quantum chemistry libraries or problem-specific Hamiltonian builders. The pass automatically handles both explicit measurement lists and derived measurements from circuit IR.

Performance benefits include:
- **Reduced Shot Scheduling Overhead**: By minimizing the number of measurement settings, the pass reduces the complexity of shot allocation and execution planning.
- **Linear Complexity**: The greedy grouping algorithm ensures that processing time scales linearly with the number of terms, making it suitable for large Hamiltonians.
- **Extensibility**: The design allows for future enhancements, such as variance-aware grouping or integration with commuting sets, without altering the core infrastructure.
- **Interoperability**: The metadata format is standardized and consumed by downstream components like the shot scheduler, ensuring seamless integration across the compiler pipeline.

These benefits make the MeasurementRewritePass a cornerstone of efficient quantum measurement in TyxonQ.

**Section sources**
- [measurement.py](file://src/tyxonq/compiler/stages/rewrite/measurement.py#L9-L88)
- [hamiltonian_grouping.py](file://src/tyxonq/libs/hamiltonian_encoding/hamiltonian_grouping.py#L11-L65)