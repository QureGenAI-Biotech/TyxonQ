# TyxonQ Refactoring Worklog (Living Document)

This file records the ongoing refactor progress, design rationale, and future optimization directions. It will be incrementally updated and will serve as the basis for a technical report upon completion.

## 1. Completed Work (to date)

- Core/IR
  - Added `core/ir/Circuit`, `Hamiltonian`, and `Pulse` IR skeletons.
  - Introduced `core/ir/CircuitBuilder` for light-weight qfunc-style recording.
  - Added `core/types.py` (Problem, backend name normalization, vectorization policy checks) and `core/errors.py`.
- Core operations and measurements
  - `core/operations`: `GateSpec`, `Operation`, in-memory registry with gradient metadata (`num_params`, `is_shiftable`, `shift_coeffs`, `gradient_method`). Registered defaults: `h`, `rz`, `cx`.
  - `core/measurements`: `Expectation`, `Probability`, `Sample`.
- Numerics
  - `numerics/api.py`: backend protocol + `vectorize_or_fallback()` and `get_backend()` factory.
  - Backends: `numpy_backend` (CPU), `pytorch_backend` (PyTorch), `cupynumeric_backend` (GPU; optional).
  - Vectorization checks: `numerics/vectorization_checks.py` (safe_for_vectorization, warn_as_error).
- Compiler
  - `compiler/api.py`; `compiler/pipeline.py` (stage registry and sequential run).
  - Stages: `decompose`/`layout`/`scheduling` (no-op), `rewrite/measurement` (implemented grouping), `scheduling/shot_scheduler` (implemented shot planning).
  - Gradients: `compiler/gradients/parameter_shift.py` minimal parameter-shift circuit pair generation.
- Measurement grouping and scheduling
  - Grouping strategy: greedy, product-basis-safe packing with per-wire `basis_map` and group metadata (`basis`, `wires`, `estimated_settings`, `estimated_shots_per_group`).
  - Shot scheduler bridges grouping to executable segments: supports explicit shot vectors and group-based weighted allocation; segments carry `basis`/`wires`/`basis_map`.
  - `devices/session.execute_plan()` consumes scheduler plan and aggregates results with per-segment context.
- Targets
  - Qiskit target skeleton: `compiler/targets/qiskit` with defaults and dialect helpers. (Transpile path not yet wired.)
- Postprocessing
  - Readout mitigation refactored under `postprocessing/readout/ReadoutMit` (no legacy imports), supports inverse and constrained least squares with per-qubit 2x2 calibration.
  - `postprocessing/metrics.py` (normalized counts, KL, diagonal expectation) and `postprocessing/io.py` (counts CSV round-trip).
- Plugins
  - Minimal plugin registry with path loading and instance caching.
- Tests & configuration
  - New `tests_refactor/` suite with progress output (`-v -ra`). 100% of new modules covered by TDD.

## 2. Why We Did It This Way (Design Rationale)

- Stabilize IR and decouple concerns, matching the layered architecture in MIGRATION_PLAN (app → compiler → devices → postprocessing → numerics).
- Make transformation paths explicit and testable (stages/pipeline), rather than ad‑hoc embedded logic.
- Keep scheduling and device/session decoupled by exchanging data plans (segments), enabling device-agnostic policies and future device-aware optimizations.
- Provide minimal but coherent numerics abstraction with named backends to support co-design experiments across CPU/GPU/AI stacks.
- Use basis-aware measurement grouping to safely reuse shots without changing circuit semantics, improving performance predictably.

## 3. Future Optimization Directions (co-design oriented)

- Measurement grouping
  - Commutation-graph based grouping beyond simple product-basis; cost models informed by variance targets and operator overlaps.
  - Integrate grouping metadata with shot allocator that minimizes total settings under accuracy constraints.
- Hardware-aware shot scheduling
  - Device-specific constraints (batch size limits, queue policy, parallel lanes, basis-change and reset costs); policy plugins per vendor.
  - Feedback loop from runtime metrics (segment durations, error bars) to refine allocation weights.
- Pulse & hardware alignment
  - Map grouping `basis_map` to concrete basis-change sequences and/or pulse schedules; connect to `core/ir/Pulse`.
- Numerics & AI
  - Extend vectorization safety analysis; integrate heuristic/static analysis for vmap/JIT safety.
  - Autodiff bridges and gradient transforms (adjoint, finite-diff) with backend-native acceleration.
  - Explore AI-assisted tuning (e.g., shot allocation, layout hints) and learning-based cost models.
- Performance & quality gates
  - CI gate on performance baselines and vectorization fallback ratios; include device session metrics.

## 4. MIGRATION_PLAN Alignment: Completed vs Pending

- Completed (first pass)
  - core/ir, core/operations, core/measurements skeletons and tests
  - numerics api/backends; vectorization checks and wrapper; factory
  - compiler api/pipeline; stages (no-op + measurement rewrite + shot scheduler)
  - devices session executor (plan consumption and aggregation)
  - gradients: parameter-shift minimal circuit pair generation
  - postprocessing: readout mitigation (refactored), metrics, io
  - plugins: minimal registry
  - tests: TDD suite with progress; all green

- Pending / Next
  - Targets
    - Qiskit target: real transpile path, basis/dialect mapping, layout/routing and opt-level handling, plus tests.
  - Devices
    - Simulators split (densitymatrix, mps) under `devices/simulators/*` and hardware adapters under `devices/hardware/*`.
    - Session: richer execution policies (async, retry, parallelism), capture runtime metrics.
  - Numerics
    - `autodiff/bridge.py` and more comprehensive linear algebra helpers.
    - Enhanced vectorization safety (static + runtime) and diagnostics.
  - App layer
    - `app/chem`: Mole/RHF/Integrals/ActiveSpace and fermion-to-qubit mappings (JW/Parity/BK) with tests.
  - Config & utils
    - `config/loader.py`; `utils/` split (logging, parallel, cache, rng) with naming aligned to plan.
  - Docs & CI
    - ADRs, API docs, performance baselines, vectorization fallback reporting; example pipelines.

## 5. Notes on Co‑Design (Domain → Hamiltonian → Compile → Hardware)

- Keep IR stable and transformations explicit, so domain front-ends (e.g., chemistry) can emit `Circuit`/`Hamiltonian` that flow through compiler stages and device sessions without rewriting.
- Measurement grouping + scheduler are the key handshake between compile-time structure and hardware-time execution; this is the anchor point for hardware-aware co-design.
- Multiple numerics backends (numpy/pytorch/cupynumeric) enable experimentation with AI methods and accelerated linear algebra; backends are pluggable to swap performance envelopes without changing IR/compile/device layers.

## 6. Changelog Snapshot (for this update)

- Implemented grouping with per-wire basis maps and cost estimates; integrated scheduler with session executor.
- Added parameter-shift gradient circuit generator.
- Refactored readout mitigation and added postprocessing metrics/io.
- Expanded tests; suite all green.

---

This document will be incrementally updated as we complete the next milestones.


## Update Timestamp

- Last update: 2025-08-22 00:26:14 CST
