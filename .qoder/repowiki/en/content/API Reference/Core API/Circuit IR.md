# Circuit IR

<cite>
**Referenced Files in This Document**   
- [circuit.py](file://src/tyxonq/core/ir/circuit.py)
- [api.py](file://src/tyxonq/compiler/api.py)
- [base.py](file://src/tyxonq/devices/base.py)
- [parameter_shift.py](file://examples/parameter_shift.py)
- [jsonio.py](file://examples/jsonio.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Core Attributes](#core-attributes)
3. [Dataclass Design and Validation](#dataclass-design-and-validation)
4. [Chainable Configuration Methods](#chainable-configuration-methods)
5. [Circuit Construction Helpers](#circuit-construction-helpers)
6. [Analysis Utilities](#analysis-utilities)
7. [Serialization and Provider Adapters](#serialization-and-provider-adapters)
8. [Execution Methods](#execution-methods)
9. [Usage Examples](#usage-examples)
10. [Integration with Compiler, Device, and Postprocessing Layers](#integration-with-compiler-device-and-postprocessing-layers)
11. [Error Conditions and Validation Rules](#error-conditions-and-validation-rules)

## Introduction
The `Circuit` class in TyxonQ's core IR system serves as the primary intermediate representation for quantum circuits. It provides a flexible and extensible framework for defining, manipulating, and executing quantum circuits. This documentation details the structure, functionality, and usage of the `Circuit` class, focusing on its role in the build → compile → execute workflow.

**Section sources**
- [circuit.py](file://src/tyxonq/core/ir/circuit.py#L48-L727)

## Core Attributes
The `Circuit` class is defined with several key attributes that define its structure and behavior:

- **num_qubits**: The number of qubits in the circuit.
- **ops**: A sequence of operation descriptors, which can be interpreted by backends or compilers (e.g., gate tuples, IR node objects).
- **metadata**: A dictionary containing additional metadata about the circuit.
- **instructions**: A list of instructions, each represented as a tuple of (name, (indices,)).

These attributes are designed to be flexible, allowing the IR to evolve while maintaining a consistent structural contract.

**Section sources**
- [circuit.py](file://src/tyxonq/core/ir/circuit.py#L48-L727)

## Dataclass Design and Validation
The `Circuit` class is implemented using Python's `dataclass` decorator, which provides a clean and concise way to define classes with attributes. The `__post_init__` method ensures structural validation, checking that:

- `num_qubits` is non-negative.
- Each operation is a tuple or list with a string name and valid qubit indices.
- Instructions are valid tuples with string names and integer indices within the qubit range.

This validation ensures that the circuit is structurally sound before any operations are performed.

**Section sources**
- [circuit.py](file://src/tyxonq/core/ir/circuit.py#L142-L169)

## Chainable Configuration Methods
The `Circuit` class supports chainable configuration methods, allowing for a fluent API style. These methods include:

- **device**: Sets device options for the circuit.
- **postprocessing**: Sets postprocessing options for the circuit.
- **compile**: Configures compilation options or triggers compilation.

These methods return the `Circuit` instance, enabling method chaining for a more readable and concise API.

**Section sources**
- [circuit.py](file://src/tyxonq/core/ir/circuit.py#L171-L200)

## Circuit Construction Helpers
The `Circuit` class provides a variety of helper methods for constructing circuits, including:

- **h, H**: Applies a Hadamard gate to a qubit.
- **rz, RZ**: Applies a rotation around the Z-axis to a qubit.
- **rx, RX**: Applies a rotation around the X-axis to a qubit.
- **cx, CX, cnot, CNOT**: Applies a controlled-NOT gate between two qubits.
- **measure_z, MEASURE_Z**: Measures a qubit in the Z-basis.
- **reset, RESET**: Resets a qubit to the |0⟩ state (simulation-only).

These methods modify the circuit in place and return the `Circuit` instance, allowing for method chaining.

**Section sources**
- [circuit.py](file://src/tyxonq/core/ir/circuit.py#L576-L612)

## Analysis Utilities
The `Circuit` class includes several utility methods for analyzing the circuit:

- **gate_count**: Counts the number of gates by name.
- **gate_summary**: Returns a dictionary mapping gate names to their frequencies.
- **count_flop**: Provides a heuristic FLOP estimate for statevector simulation.
- **get_circuit_summary**: Returns a comprehensive summary of the circuit, including qubit count, gate count, CNOT count, multicontrol count, depth, and FLOP estimate.

These utilities are lightweight and backend-agnostic, making them useful for quick analysis and debugging.

**Section sources**
- [circuit.py](file://src/tyxonq/core/ir/circuit.py#L191-L256)

## Serialization and Provider Adapters
The `Circuit` class supports serialization to JSON and conversion to OpenQASM:

- **to_json_obj**: Returns a dictionary representation of the circuit suitable for JSON serialization.
- **to_json_str**: Returns a JSON string representation of the circuit.
- **to_openqasm**: Serializes the circuit to OpenQASM 2 using the compiler facade.

These methods facilitate interoperability with other tools and systems.

**Section sources**
- [circuit.py](file://src/tyxonq/core/ir/circuit.py#L344-L371)

## Execution Methods
The `Circuit` class provides methods for executing the circuit:

- **run**: Executes the circuit on a specified device, handling both direct source submission and compilation.
- **submit_task**: An alias for `run` with identical semantics.
- **get_task_details**: Retrieves details about a submitted task.
- **get_result**: Fetches the result of a submitted task.
- **cancel**: Cancels a submitted task.

The `run` method integrates with the device layer, handling both simulator and hardware execution, and supports postprocessing of results.

**Section sources**
- [circuit.py](file://src/tyxonq/core/ir/circuit.py#L404-L515)

## Usage Examples
The `examples` directory contains several scripts demonstrating the use of the `Circuit` class:

- **parameter_shift.py**: Demonstrates the use of the `Circuit` class for parameter shift gradient estimation.
- **jsonio.py**: Shows how to serialize and deserialize circuits using JSON.

These examples provide practical insights into the usage of the `Circuit` class in real-world scenarios.

**Section sources**
- [parameter_shift.py](file://examples/parameter_shift.py#L32-L61)
- [jsonio.py](file://examples/jsonio.py#L56-L79)

## Integration with Compiler, Device, and Postprocessing Layers
The `Circuit` class integrates seamlessly with the compiler, device, and postprocessing layers:

- **Compiler**: The `compile` method delegates to the compiler API, allowing for flexible compilation strategies.
- **Device**: The `run` method uses the device layer to execute the circuit, supporting both simulator and hardware execution.
- **Postprocessing**: The `run` method applies postprocessing to the results, using the `apply_postprocessing` function from the postprocessing module.

This integration ensures a smooth workflow from circuit creation to execution and result analysis.

**Section sources**
- [circuit.py](file://src/tyxonq/core/ir/circuit.py#L404-L515)
- [api.py](file://src/tyxonq/compiler/api.py#L0-L65)
- [base.py](file://src/tyxonq/devices/base.py#L0-L403)

## Error Conditions and Validation Rules
The `Circuit` class enforces several validation rules to ensure the integrity of the circuit:

- **num_qubits**: Must be non-negative.
- **ops**: Each operation must be a tuple or list with a string name and valid qubit indices.
- **instructions**: Each instruction must be a valid tuple with a string name and integer indices within the qubit range.

Violations of these rules raise appropriate exceptions, such as `ValueError` or `TypeError`, providing clear feedback to the user.

**Section sources**
- [circuit.py](file://src/tyxonq/core/ir/circuit.py#L142-L169)