# Expectation Value Calculation

<cite>
**Referenced Files in This Document**   
- [counts_expval.py](file://src/tyxonq/postprocessing/counts_expval.py)
- [__init__.py](file://src/tyxonq/postprocessing/__init__.py)
- [readout.py](file://src/tyxonq/postprocessing/readout.py)
- [vqe.py](file://src/tyxonq/libs/circuits_library/vqe.py)
- [hamiltonian_grouping.py](file://src/tyxonq/libs/hamiltonian_encoding/hamiltonian_grouping.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Core Functions Overview](#core-functions-overview)
3. [Pauli Term Input Format](#pauli-term-input-format)
4. [expval_pauli_term: Single Term Expectation](#expval_pauli_term-single-term-expectation)
5. [expval_pauli_terms: Multiple Term Expectations](#expval_pauli_terms-multiple-term-expectations)
6. [expval_pauli_sum: Aggregated Hamiltonian Energy](#expval_pauli_sum-aggregated-hamiltonian-energy)
7. [Analytic Evaluation Path](#analytic-evaluation-path)
8. [Readout Mitigation Integration](#readout-mitigation-integration)
9. [Usage in VQE and QAOA Workflows](#usage-in-vqe-and-qaoa-workflows)
10. [Implementation Details](#implementation-details)

## Introduction
This document details the expectation value computation system in TyxonQ for Pauli-based Hamiltonians. The framework supports both shot-based estimation from measurement counts and analytic evaluation using statevector-derived expectations. The core functionality is implemented in the `postprocessing` module, specifically through the `expval_pauli_term`, `expval_pauli_terms`, and `expval_pauli_sum` functions, which enable flexible energy estimation for variational quantum algorithms such as VQE and QAOA.

## Core Functions Overview
TyxonQ provides three primary functions for computing expectation values from measurement outcomes:
- `expval_pauli_term`: Computes the expectation of a single Pauli-Z product term from bitstring counts.
- `expval_pauli_terms`: Returns a list of expectations for multiple Pauli terms under Z-basis counts.
- `expval_pauli_sum`: Aggregates energy for a full Pauli-sum Hamiltonian, supporting both shot-based and analytic evaluation paths.

These functions are accessible through the postprocessing router via the `apply_postprocessing` interface, which routes calls based on the specified method.

**Section sources**
- [counts_expval.py](file://src/tyxonq/postprocessing/counts_expval.py#L33-L111)

## Pauli Term Input Format
Pauli terms are represented as sequences of (qubit, operator) tuples, where the operator is one of "X", "Y", or "Z". Each term may optionally include a coefficient. Inputs can be provided in two formats:
- Term-only: `Tuple[(qubit, operator), ...]`
- With coefficient: `Tuple[Tuple[(qubit, operator), ...], coefficient]`

The `_normalize_term_entry` utility function handles both formats, extracting the term and optional coefficient for processing.

**Section sources**
- [counts_expval.py](file://src/tyxonq/postprocessing/counts_expval.py#L22-L30)

## expval_pauli_term: Single Term Expectation
The `expval_pauli_term` function computes the expectation value of a single Pauli-Z product from bitstring counts. It serves as a thin wrapper around `term_expectation_from_counts`, which accumulates the sign of each bitstring based on the specified qubit indices. The expectation is calculated as the weighted average of the product of ±1 values corresponding to the measurement outcomes on the selected qubits.

**Section sources**
- [counts_expval.py](file://src/tyxonq/postprocessing/counts_expval.py#L33-L35)

## expval_pauli_terms: Multiple Term Expectations
The `expval_pauli_terms` function computes expectations for a list of Pauli terms under Z-basis counts. It iterates over the input terms, normalizes each entry using `_normalize_term_entry`, and applies `term_expectation_from_counts` to compute individual expectations. Coefficients are ignored in this function; users should use `expval_pauli_sum` when energy aggregation with coefficients is required.

**Section sources**
- [counts_expval.py](file://src/tyxonq/postprocessing/counts_expval.py#L38-L51)

## expval_pauli_sum: Aggregated Hamiltonian Energy
The `expval_pauli_sum` function aggregates energy for a full Pauli-sum Hamiltonian from either counts or analytic expectations. It accepts a list of items, each being a term or (term, coefficient) pair, and an optional identity constant. When counts are provided, it computes each term's expectation and accumulates the weighted sum. If analytic expectations are available (e.g., from a statevector simulation with `shots=0`), it automatically falls back to the analytic path using `_expval_pauli_sum_analytic`.

The function returns a dictionary containing the total energy and individual term expectations.

```mermaid
flowchart TD
Start([expval_pauli_sum]) --> CheckAnalytic{"expectations provided?<br/>counts empty?"}
CheckAnalytic --> |Yes| AnalyticPath[Use _expval_pauli_sum_analytic]
CheckAnalytic --> |No| ShotPath[Use counts with term_expectation_from_counts]
AnalyticPath --> Aggregate[Sum coeff * expval]
ShotPath --> Aggregate
Aggregate --> Return{"energy": float, "expvals": List[float]}
```

**Diagram sources **
- [counts_expval.py](file://src/tyxonq/postprocessing/counts_expval.py#L86-L111)

**Section sources**
- [counts_expval.py](file://src/tyxonq/postprocessing/counts_expval.py#L86-L111)

## Analytic Evaluation Path
When `shots=0` and a statevector simulator is used, the system automatically falls back to analytic evaluation. The `expval_pauli_sum` function detects the presence of per-qubit Z expectations or statevector data and routes to `_expval_pauli_sum_analytic`. This function computes exact expectation values using either the full probability distribution or the product of individual qubit expectations. The analytic path ensures precise energy estimation without sampling noise, making it ideal for algorithm development and verification.

**Section sources**
- [counts_expval.py](file://src/tyxonq/postprocessing/counts_expval.py#L54-L83)

## Readout Mitigation Integration
Readout mitigation can be integrated within the aggregation process when calibrations are provided. During `expval_pauli_sum` execution, if readout calibration data (`cals` or `readout_cals`) is present in the options, the system applies mitigation using the `ReadoutMit` class. The mitigation is applied to the raw counts before expectation computation, with the default method being matrix inversion ("inverse"). This feature allows for error-corrected energy estimation within the same postprocessing pipeline.

**Section sources**
- [__init__.py](file://src/tyxonq/postprocessing/__init__.py#L96-L117)
- [readout.py](file://src/tyxonq/postprocessing/readout.py#L1-L142)

## Usage in VQE and QAOA Workflows
The expectation computation system is critical for VQE and QAOA workflows where energy estimation drives the optimization loop. In VQE, the `expval_pauli_sum` function is used to evaluate the Hamiltonian expectation for a given ansatz circuit. In QAOA, it computes the cost function value from measurement counts. Examples such as `vqe_extra.py` and `simple_qaoa.py` demonstrate how these functions are chained with circuit execution and parameter shifts to implement complete variational algorithms.

**Section sources**
- [vqe.py](file://src/tyxonq/libs/circuits_library/vqe.py#L0-L152)

## Implementation Details
The core implementation relies on bitstring parity accumulation to compute Pauli term expectations. For a given set of qubit indices, the sign of each bitstring is determined by the product of ±1 values corresponding to the measurement outcomes. The expectation is the average sign weighted by count frequency. Coefficient weighting is applied during aggregation in `expval_pauli_sum`. The system supports both big-endian bitstring indexing and proper handling of identity terms. Hamiltonian grouping, as implemented in `hamiltonian_grouping.py`, enables efficient measurement reduction by grouping commuting terms.

**Section sources**
- [counts_expval.py](file://src/tyxonq/postprocessing/counts_expval.py#L6-L19)
- [hamiltonian_grouping.py](file://src/tyxonq/libs/hamiltonian_encoding/hamiltonian_grouping.py#L0-L65)