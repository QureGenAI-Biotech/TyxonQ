## TyxonQ Technical Architecture Report

### 1. Overview

TyxonQ is a modular quantum computing platform designed to mirror real hardware while remaining highly usable for engineers and scientists. The architecture centers on a stable Core IR, a pluggable compilation pipeline, unified device abstraction (hardware and simulators), a single numeric backend interface, and a results-first postprocessing layer. Examples adopt a chain-style API to ensure a uniform path across simulators and real hardware.

A first-class mission of the system is QC^2 — Quantum Computing for Quantum Chemistry (and biopharma). We prioritize chemistry workloads end-to-end, from algorithms and Hamiltonians to device execution and reproducible numeric baselines, under one coherent architecture.

Key capabilities
- Stable Core IR decouples applications, compiler, devices, and postprocessing.
- Pluggable compiler pipeline with measurement rewrite/grouping, simplification, scheduling, and gradient passes.
- Unified device abstraction covering simulators and real hardware, with explicit shots/noise semantics.
- A single numeric backend interface (NumPy, PyTorch, CuPyNumeric) powering simulators and numeric libraries alike.
- Postprocessing owns counts→expectations, readout mitigation, and metrics for provider-neutral results.
- Reusable libraries (circuits_library, quantum_library) to accelerate application development.

Chemistry application objectives (QC^2 in practice)
- PySCF-like user experience for quantum algorithms (UCC, HEA, PUCCD, k‑UpCCGSD) while remaining hardware-realistic.
- Unified device and numeric execution: device runtimes return counts with explicit shots/noise handling and postprocessing; numeric runtimes produce exact statevector/civector baselines for verification and autograd.
- Reproducible, portable results via centralized measurement semantics and mitigation in postprocessing.
- Seamless bridge to classical chemistry tooling (PySCF, OpenFermion) integrated with TyxonQ IR, compilers, and devices.

Architectural principles and unifications
- Dual-path execution model (device and numeric) with shared semantics for measurement, grouping, and expectations.
- Single numeric backend abstraction enabling ML/backend portability without refactoring user code.
- Explicit, opt‑in noise controls and mitigation in device paths; deterministic analytic paths in numeric mode for verification and gradient baselines.
- Typed, provider‑agnostic IR and compiler passes so the same circuit lowers to different targets without changing user code.

Execution model (summary)
1) Build IR (optionally using circuits_library templates).
2) Compile through the pass pipeline (measurement rewrite/grouping, simplification, scheduling, gradients as needed).
3) Execute on a device (simulator or hardware) with counts‑first semantics (shots, noise optionally enabled), or on the numeric path for exact statevector/civector.
4) Postprocess counts to expectations and metrics; mitigation is applied where configured.

Chemistry highlights (integrated)
- Algorithms: HEA and the UCC family (UCC/UCCSD/k‑UpCCGSD/PUCCD) expose consistent energy/gradient/kernel APIs, engineered for both device and numeric paths. PUCCD restores RDM1/2 (with AO basis) and stabilizes H4 via paired excitations.
- Runtimes: device runtimes group Hamiltonians and forward postprocessing; numeric runtimes construct civector/statevector baselines. Dynamics (VQD/pVQD/Trotter) numeric runtime caches dense term matrices and properties to avoid repetitive MPO construction.
- Hamiltonians: unified sparse/MPO/FCI‑function outputs; hcb+fcifunc accept int or (na, nb); MPO exposes a lightweight `eval_matrix()` for test integration.
- Numeric bridging: Pauli I/O bridges via the active Numeric Backend; top‑level `backend/rdtypestr` are lazily exposed to maintain compatibility.


### 2. Architecture Diagram
```mermaid
flowchart LR
  A[Applications] --> B[Core / IR]
  B --> C[Compiler / Pipeline]
  C --> D[Devices]
  D --> E[Postprocessing]

  %% Libraries
  subgraph LIBS [Libraries]
    L1[circuits_library]
    L2[quantum_library]
    L3[hamiltonian_encoding]
  end
  A --> L1
  L1 --> B
  L2 --> D

  %% Numerics
  subgraph NUM [Numerics]
    N1[NumPy / PyTorch / CuPyNumeric]
  end
  N1 --> L2

  %% Chem application (high-level)
  subgraph CHEM [Applications/Chem]
    CH_ALG[algorithms: HEA/UCC/PUCCD/k-UpCCGSD]
    CH_MOL[molecule & chem hamiltonians]
  end
  A --> CHEM
  CH_MOL --> CH_ALG

  %% Runtimes abstraction (device/numeric split)
  subgraph RUNTIMES [runtimes]
    RT_DEV[device runtime]
    RT_NUM[numeric runtime]
  end
  CH_ALG --> RT_DEV
  CH_ALG --> RT_NUM
  RT_DEV --> D
  D --> E
  RT_NUM --> L2
  RT_NUM --> N1

  %% chem_libs as a dedicated domain layer
  subgraph CHEM_LIBS [chem_libs]
    CL_CIRCUIT[circuit_chem_library (ansatz)]
    CL_QCHEM[quantum_chem_library (CI/state ops)]
    CL_HAM[hamiltonians_chem_library]
  end

  %% Wiring between chem libs and the rest
  CL_CIRCUIT --> CH_ALG
  CH_MOL --> CL_HAM
  CL_HAM --> L3
  L3 --> OF[OpenFermion]
  CH_MOL --> PYSCF[PySCF]
  CL_QCHEM --> RT_NUM
```



Module responsibilities:
- Applications (`src/tyxonq/applications/*`): domain modules (e.g., Chemistry) that assemble algorithms, runtimes (device/numeric), and domain libraries over the Core/Compiler/Devices stack.
- Core (`src/tyxonq/core/*`): IR types (`ir/circuit.py`, `ir/pulse.py`), `operations`, `measurements`, error types.
- Compiler (`src/tyxonq/compiler/*`): stages for rewrite, simplify, layout, scheduling, gradients; different compiler engine adapters under `compiler/compile-engine/*`.
- Devices (`src/tyxonq/devices/*`): base device contracts, simulators (statevector, density_matrix, MPS) and hardware drivers (`hardware/<vendor>/driver.py`), sessions.
- Numerics (`src/tyxonq/numerics/*`): backend factory/context; backends for NumPy, PyTorch, CuPyNumeric.
- Postprocessing (`src/tyxonq/postprocessing/*`): metrics, IO, readout mitigation, QEM; consumes counts/samples to produce expectations.
- Libraries (`src/tyxonq/libs/*`): `circuits_library` for reusable circuit templates; `quantum_library` for state evolution and kernels.
- Cloud (`src/tyxonq/cloud/api.py`): façade that routes to devices/hardware sessions.

### 2.1 Source Tree
```text
src/tyxonq/
  about.py                               # meta info
  cloud/                                  # submission façade & provider routing
    api.py
  compiler/                               # pass pipeline & provider engines
    api.py
    native_compiler.py
    pipeline.py
    compile_engine/
      qiskit/
        dialect.py
        qiskit_compiler.py
    stages/
      decompose/rotations.py
      gradients/parameter_shift_pass.py, qng.py
      rewrite/measurement.py, merge_prune.py
      scheduling/shot_scheduler.py
      simplify/lightcone.py
    translation/mpo_converters.py
  config.py                                # global configs
  core/                                    # canonical IR & semantics
    errors.py
    ir/circuit.py, pulse.py
    measurements/
    operations/
    types.py
  devices/                                 # device/session, simulators, hardware
    base.py
    session.py
    hardware/
      config.py
      session.py
      ibm/driver.py
      tyxonq/driver.py
    simulators/
      driver.py
      statevector/engine.py
      density_matrix/engine.py
      matrix_product_state/engine.py
      noise/channels.py
  applications/
    chem/                                  # quantum chemistry application layer
      __init__.py
      constants.py
      molecule.py                           # molecule factories & HF entry
      algorithms/                           # HEA/UCC family (strategy only)
        hea.py
        ucc.py
        uccsd.py
        kupccgsd.py
        puccd.py
      runtimes/                             # device/numeric execution paths
        hea_device_runtime.py
        hea_numeric_runtime.py
        ucc_device_runtime.py
        ucc_numeric_runtime.py
        dynamics_numeric.py
      chem_libs/
        circuit_chem_library/               # chemistry ansatz generators
          ansatz_uccsd.py
          ansatz_kupccgsd.py
          ansatz_puccd.py
        quantum_chem_library/               # CI basis/state ops (device-agnostic)
          ci_state_mapping.py
          ci_operator_tensors.py
          civector_ops.py
          pyscf_civector.py
          statevector_ops.py
        hamiltonians_chem_library/          # HF/integrals→Hamiltonian (sparse/MPO/fcifunc)
          __init__.py
          hamiltonian_builders.py
          pyrazine.py
          sbm.py
  libs/
    circuits_library/                       # reusable blocks & state prep
      blocks.py, hamiltonians.py, qaoa_ising.py, trotter_circuit.py, vqe.py
      qubit_state_preparation.py
    quantum_library/                        # numeric kernels over active backend
      dynamics.py
      kernels/common.py, gates.py, pauli.py, statevector.py, density_matrix.py, unitary.py, matrix_product_state.py
    hamiltonian_encoding/                   # OpenFermion↔sparse I/O, RDM helpers
      pauli_io.py
  numerics/                                 # ArrayBackend interface & backends
    api.py, context.py, vectorization_checks.py
    backends/numpy_backend.py, pytorch_backend.py, cupynumeric_backend.py
  plugins/registry.py
  postprocessing/                           # counts→expectations, mitigation, metrics
    io.py, metrics.py, noise_analysis.py
    readout/
    classical_shadows/
    qem/
  utils.py
  visualization/dot.py
```

Descriptions (Key Modules):
- Core (`src/tyxonq/core`): Canonical IR (circuits/pulses), operation and measurement semantics, error types.
- Compiler (`src/tyxonq/compiler`): Pass pipeline (rewrite, simplify, measurement grouping, scheduling, gradients) and provider-specific engines.
- Devices (`src/tyxonq/devices`): Uniform device/session contracts; simulators (statevector/density_matrix/MPS) and hardware drivers.
- Applications/Chem (`src/tyxonq/applications/chem`): Domain layer for quantum chemistry.
  - `algorithms/`: HEA, UCC, UCCSD, k‑UpCCGSD, pUCCD implementations; only strategy/entrypoints, no device specifics.
  - `runtimes/`: Device and numeric execution paths; device side handles shots/noise/postprocessing; numeric side produces exact statevector/civector baselines.
  - `chem_libs/hamiltonians_chem_library/`: Chemistry Hamiltonian builders from HF/integrals; exports sparse/MPO/FCI-function forms; domain models (pyrazine, SBM).
  - `chem_libs/quantum_chem_library/`: CI basis utilities, state/civector mapping, CI-space operator tensors; independent of devices.
  - `chem_libs/circuit_chem_library/`: Ansatz generators for chemistry variants (UCCSD, k‑UpCCGSD, pUCCD).
  - `molecule.py`: Lightweight molecule factories (e.g., H2/H4/CH4/benzene) with HF shortcuts.
- Libraries (`src/tyxonq/libs`): Reusable building blocks for all apps.
  - `circuits_library/`: Parametric blocks, QAOA/Trotter templates, VQE flows, and qubit state preparation (HCB givens-swap).
  - `quantum_library/`: Numeric kernels for state evolution and expectations (statevector, density_matrix, MPS) over the active backend.
  - `hamiltonian_encoding/`: OpenFermion ↔ sparse conversions, Pauli I/O, chemistry helpers (e.g., RDM transforms); bridged via NumericBackend.
- Numerics (`src/tyxonq/numerics`): Single ArrayBackend interface and backends for NumPy/PyTorch/CuPyNumeric; vectorization/dtype policy.
- Postprocessing (`src/tyxonq/postprocessing`): Counts→expectations, readout mitigation/QEM, metrics; normalizes provider outputs.
- Cloud (`src/tyxonq/cloud`): Submission façade and provider routing; harmonized task/result schemas.

### 2.2 Core Module Feature Plan
- Core/IR
  - Circuit/Pulse as unified intermediate representation; metadata for measurement groups, basis maps, and scheduling hints
  - Operation registry with gradient metadata (parameter-shift availability, commutation rules)
  - Measurement semantics: standardize expectation targets, diagonalization rules
- Compiler
  - Pass pipeline orchestration; per-stage contracts and reproducible pass configs
  - Measurement rewrite and grouping; light-cone pruning; scheduling (shot segmentation)
  - Gradient passes (parameter-shift, QNG); different compiler-specific dialect lowering
- Devices
  - Stable Device contract: run/compile interfaces, shot execution, result normalization
  - Simulators share numerics backend; unified noise controls; performance baselines
  - Hardware sessions: submission/polling, error normalization, consistent RunResult schema
- Numerics
  - Single ArrayBackend interface; centralized set_backend/get_backend
  - Vectorization support and dtype policy; minimal capability checks
  - Multiple implementations: NumPy, PyTorch, CuPyNumeric
- Postprocessing
  - Counts-to-expectation with basis metadata; metrics; readout mitigation and QEM
  - IO schemas for device/cloud results; composable pipelines
- Libraries
  - circuits_library: reusable templates (VQE, QAOA, Trotter) with IR-friendly metadata
  - quantum_library: decoupled numeric kernels for state evolution and expectations
- Cloud/Plugins
  - Cloud facade for provider routing; plugin registry for devices/compilers/postprocessing

### 3. Chain-style API and Execution Flow
The chain-style API standardizes the end-to-end path from circuit construction to device execution and postprocessing, especially important for unifying simulators and real hardware.

Important: compiler passes take effect only if they participate in the chain prior to run, i.e., when you call `.compile(...)` before `.device(...).run()`, or when you execute the compiled IR returned by the compiler API.

Typical flow:
1) Build IR (possibly via `circuits_library`).
2) Compile through `compiler` pipeline.
3) Execute on a `device` (simulator or hardware) with shots and scheduling.
4) Postprocess counts to expectations and metrics.

Example sketch (aligned with `examples/circuit_chain_demo.py` and `examples/readout_mitigation.py`):
```python
import tyxonq as tq
from tyxonq.postprocessing import metrics

# Backend selection once (numpy/pytorch/cupynumeric)
tq.set_backend("numpy")

# 1) Build circuit IR
c = tq.Circuit(2).h(0).cx(0, 1).measure_z(0).measure_z(1)

# 2) Simulator chain: use default compiler passes for measurement + shots
sim_res = (
    c
     .compile()  # passes take effect here
     .device(provider="simulator", device="statevector", shots=4096)
     .postprocessing(method=None)
     .run()
)

# 3) Hardware chain: choose provider-appropriate compiler settings
#    (example: qiskit engine + vendor-specific passes if needed)
hw_res = (
    c
     .compile(compile_engine="qiskit")  # per-target config
     .device(provider="tyxonq", device="homebrew_s2", shots=4096)
     .run()  # postprocessing can be added or done later depending on driver output
)

# 4) Extract counts and compute expectations (if needed)
def counts_of(res):
    payload = res if isinstance(res, dict) else (res[0] if res else {})
    return payload.get("result", {})

sim_counts = counts_of(sim_res)
hw_counts = counts_of(hw_res)
ez_sim = metrics.expectation(sim_counts, z=[0, 1])
ez_hw = metrics.expectation(hw_counts, z=[0, 1])
print("E[Z] (sim)", ez_sim)
print("E[Z] (hw) ", ez_hw)
```

This shows: (1) passes are effective only when `.compile(...)` is invoked in the chain, (2) per-device compilation can be configured by selecting different engines/passes for different targets while reusing the same circuit.

This design makes the “counts-first” result format the default, while allowing providers to return expectations directly via standardized `RunResult` metadata when available.


#### 3.1 Simulator vs Hardware Dispatch (Same Circuit)
```python
import tyxonq as tq

# Select numeric backend once
tq.set_backend("numpy")  # or "pytorch" / "cupynumeric"

# Build the same circuit
c = tq.Circuit(2).h(0).cx(0, 1).measure_z(0).measure_z(1)

# Run on local simulator with default compiler
sim_res = (
    c
     .compile(passes=["measurement_rewrite", "shot_scheduler"])  # effective here
     .device(provider="simulator", device="statevector", shots=4096)
     .postprocessing(method=None)
     .run()
)

# Run on real hardware with provider-aware compiler configuration
hw_res = (
    c
     .compile(compile_engine="qiskit", passes=["measurement_rewrite", "shot_scheduler"])  # choose engine per target
     .device(provider="tyxonq", device="homebrew_s2", shots=4096)
     .run()
)

# Extract counts (both follow the same result schema)
def counts_of(res):
    payload = res if isinstance(res, dict) else (res[0] if res else {})
    return payload.get("result", {})

sim_counts = counts_of(sim_res)
hw_counts = counts_of(hw_res)
print("sim counts sample:", list(sim_counts.items())[:3])
print("hw  counts sample:", list(hw_counts.items())[:3])
```

Notes:
- Use different `.compile(...)` options per target; do not attempt to set conflicting compiler configurations in a single chain. Create one chain per target.
- If a provider returns native expectations, drivers can pass them via metadata; otherwise postprocessing computes expectations from counts, uniformly.

Representative example usages from the repo:

- JSON IO and chain execution
```24:28:examples/jsonio.py
res = (
    c.device(provider="simulator", device="statevector", shots=shots)
     .postprocessing(method=None)
     .run()
)
```

- Explicit noise controls at call time
```21:25:examples/noise_controls_demo.py
res_clean = (
    c.device(provider="local", device="statevector", shots=0, use_noise=True)
     .run()
)
```
```29:39:examples/noise_controls_demo.py
res_noisy = (
    c.device(
        provider="local",
        device="statevector",
        shots=0,
        use_noise=True,
        noise={"type": "depolarizing", "p": 0.05},
    )
     .postprocessing(method=None)
     .run()
)
```

- Counts-based expectation and gradients via parameter-shift
```68:72:examples/sample_value_gradient.py
res = (
    cc.device(provider="simulator", device="statevector", shots=shots)
      .postprocessing(method=None)
      .run()
)
```
```93:101:examples/sample_value_gradient.py
def parameter_shift_gradient(param, shots: int = 4096, shift: float = 1.5707963267948966):
    p_np = nb.to_numpy(param)
    grad = nb.zeros_like(p_np)
    for i in range(n):
        for j in range(nlayers):
            for k in (0, 1):
                p_plus = p_np.copy(); p_minus = p_np.copy()
                p_plus[i, j, k] += shift
                p_minus[i, j, k] -= shift
```

These examples illustrate: chain-style composition, explicit device/noise control, counts-first expectations aligned with postprocessing, and numeric backends used for gradient baselines and comparisons.

### 4. Chain-style End-to-End Example (Complete)
```python
import tyxonq as tq
from tyxonq.compiler import api as compiler
from tyxonq.postprocessing import metrics

# 1) Backend selection (once)
tq.set_backend("numpy")  # or "pytorch" / "cupynumeric"

# 2) Build IR (could also reuse circuits_library templates)
c = tq.Circuit(3)
c.h(0).rx(1, theta=0.3).rz(2, theta=-0.5).cx(0, 2)
# Compiler will infer basis rotations; we only mark measurement intent
c.measure_z(0).measure_z(1).measure_z(2)

# 3) Compile to device-agnostic IR with measurement grouping and shot scheduling
compiled = compiler.compile(
    c,
    provider="default",
    passes=[
        "measurement_rewrite",  # generate basis maps/diagonalization
        "lightcone",            # optional simplification
        "shot_scheduler",       # segment shots if needed
    ],
)

# 4) Execute on a simulator or hardware device (noise explicitly controlled)
res = (
    c.device(
        provider="simulator", device="statevector", shots=8192,
        use_noise=False,
    )
     .postprocessing(method=None)  # let postprocessing interpret counts
     .run()
)

# 5) Postprocess counts to expectations using compiler-produced basis metadata
payload = res if isinstance(res, dict) else (res[0] if res else {})
counts = payload.get("result", {})
ez = metrics.expectation(counts, z=[0, 1, 2])
print("E[Z]:", ez)

# Optional: direct-numeric path for baseline comparisons
from tyxonq.numerics import get_backend
from tyxonq.libs.quantum_library import dynamics as qdyn
nb = get_backend(None)
psi_t = qdyn.evolve_state(psi0=nb.zeros([2**3], dtype=nb.complex64), hamiltonian=None, time=0.5, backend=nb)
# In practice supply a Hamiltonian and use qdyn.expectation(...) accordingly
```


#### 4.1 Numeric Backends and Dual Paths
TyxonQ exposes a single numeric backend abstraction to support multiple ecosystems. We illustrate two complementary cases:

- Case A: Chain path (basis rotation + counts → expectations), aligned to devices and postprocessing
```python
import tyxonq as tq
from tyxonq.postprocessing import metrics

# Build parametric circuit
tq.set_backend("numpy")
c = tq.Circuit(3)
for q in range(3):
    c.h(q)
c.rzz(0, 1, theta=0.4).rx(2, theta=0.2)
# measurement intent
for q in range(3):
    c.measure_z(q)

# Execute and compute counts-based expectations
res = c.device(provider="simulator", device="statevector", shots=4096).postprocessing(method=None).run()
counts = res[0]["result"] if isinstance(res, list) else res.get("result", {})
ez = metrics.expectation(counts, z=[0, 1, 2])
```

- Case B: Direct numeric + PyTorch autograd fusion (adapted from `examples/incremental_twoqubit.py`)
```python
import tyxonq as tq
import torch
from tyxonq.libs.quantum_library.kernels.statevector import (
    init_statevector, apply_1q_statevector, apply_2q_statevector, expect_z_statevector
)
from tyxonq.libs.quantum_library.kernels.gates import gate_rx, gate_rzz, gate_h

n, nlayers = 8, 3

tq.set_backend("pytorch")
nb = tq.get_backend("pytorch")

params = torch.randn(2*nlayers, n, dtype=torch.float64, requires_grad=True)
structures = torch.randint(low=0, high=2, size=(nlayers, n-1), dtype=torch.float64)

def energy_autograd(params_t: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
    psi = init_statevector(n, backend=nb)
    for i in range(n):
        psi = apply_1q_statevector(nb, psi, gate_h(), i, n)
    for j in range(nlayers):
        for i in range(n-1):
            theta_eff = (1.0 - s_t[j, i]) * params_t[2*j + 1, i]
            psi = apply_2q_statevector(nb, psi, gate_rzz(2.0 * theta_eff), i, i+1, n)
        for i in range(n):
            psi = apply_1q_statevector(nb, psi, gate_rx(params_t[2*j, i]), i, n)
    e = torch.zeros((), dtype=params_t.dtype)
    for i in range(n):
        psi_x = apply_1q_statevector(nb, psi, gate_h(), i, n)
        e = e + expect_z_statevector(psi_x, i, n, backend=nb)
    return e

loss = energy_autograd(params, structures)
loss.backward()
```

### 5. Reusable Libraries
- circuits_library: reusable circuit templates (e.g., VQE blocks, QAOA Ising, trotter circuits). Encourages standardized circuit construction and metadata.
- quantum_library: numeric kernels for state evolution and expectation calculation, decoupled from devices. Useful for reference baselines and research.

### 6. Postprocessing Principles
- Owns conversion from counts/samples to expectations, matching measurement semantics defined in `core/measurements` and compiler stages (e.g., basis maps, grouping).
- Houses mitigation and readout correction to keep device drivers simple and portable.
- Harmonizes provider result formats: counts-centric but able to consume native expectations when drivers supply them in metadata.

### 7. Devices and Sessions
- Device drivers execute shot plans generated by the compiler, using IR annotations like measurement groups and basis transforms.
- Hardware sessions unify submission and polling logic, consolidate error normalization, and standardize `RunResult`.
- Simulators (statevector, density_matrix, MPS) consume the same IR and numeric backend, enabling like-for-like comparisons and benchmarks.

### 8. Signature Features with Code Snippets
- Chain API: `c.device(...).postprocessing(...).run()` pattern in examples demonstrates modular assembly.
- Parameter-shift gradients in compiler stages align with `operations` metadata, ensuring gradient semantics are consistent between compilers and simulators.
- Dual-path VQE and incremental-twoqubit examples preserve chain-vs-direct numeric comparisons for scientific transparency.

### 5. Comparison with PennyLane
- Execution model
  - PennyLane centers on `QNode`s with measurements returning expectations/stats directly; drivers or interfaces may hide measurement transforms.
  - TyxonQ intentionally standardizes on counts-first results and delegates expectation computation to `postprocessing/`, making provider diversity explicit and portable across hardware.
- Compiler and IR separation
  - PennyLane offers transforms and templates but does not expose a hardware-agnostic IR in the same way; decomposition is often embedded in transform stacks.
  - TyxonQ’s `core/ir` with compiler stages (rewrite, simplify, scheduling, gradients) formalizes contracts between components and devices.
- Numeric backends
  - PennyLane integrates with multiple ML frameworks through interfaces; autograd usually wraps the entire QNode.
  - TyxonQ provides a single ArrayBackend abstraction used consistently by simulators and numeric libraries; autograd via PyTorch backend is a first-class but optional path.
- Postprocessing and mitigation
  - PennyLane provides measurement/shot handling at the QNode level; mitigation toolsets exist but often as plugins or external utilities.
  - TyxonQ centralizes readout mitigation/QEM and metrics in `postprocessing/`, ensuring consistent behavior regardless of device/provider.
- Device/session abstraction
  - PennyLane devices are pluggable; cloud execution is mediated via plugins/providers.
  - TyxonQ adds explicit `devices/hardware/session.py` for submission/polling, error normalization, and a standardized `RunResult`, aiding systems integration for real deployments.

Practical implications for engineers and architects:
- Predictable results pipeline across simulators and hardware (counts → postprocess), simpler to validate and monitor in production.
- Clear separation of concerns enables parallel development (compiler optimizations vs device drivers vs numeric baselines).
- Backend choice is a configuration, not a refactor, unlocking portability and reproducible performance studies.

### 9. Next Steps
- Expand `circuits_library.vqe` reuse across all VQE variants; phase out bespoke implementations.
- Flesh out provider dialect separation in `compiler/providers/*` and normalize device naming.
- Strengthen postprocessing mitigation strategies and IO schemas for cloud/hardware results.
- Performance baselines across backends and devices; vectorization and scheduling improvements.
- Documentation hardening and TDD expansion for `core/`, `numerics/`, `devices/`, and `postprocessing/`.

### 6. Next Steps (Refined)
- Expand `circuits_library.vqe` adoption across VQE variants; provide minimal working examples for each device type.
- Strengthen `postprocessing` APIs for batch evaluation and pluggable mitigation strategies.
- Provider dialect matrices and conformance tests; improve `compiler/providers/*` coverage.
- Performance dashboards comparing backends and devices for representative workloads.

### 10. Architectural Principles for the New System Build
- Shift from direct state manipulation as the primary interface to circuit execution plans reflecting hardware scheduling and measurement optimization.
- Centralize numeric backend selection; unify dtype/policy at the backend; minimize per-function capability checks.
- Adopt counts-first workflows for cross-vendor portability, while allowing explicit expectation passthrough from providers when available.
- Express complex/composite gates via `operations` metadata and compiler rewrite passes to maintain semantic consistency across the stack.
- Place noise mitigation and readout correction in `postprocessing/` to keep device drivers thin and provider-neutral.

This reflects a full system rebuild oriented to real-world hardware integration and systems engineering needs, emphasizing clarity of contracts and practicality for ordinary developers and system architects.

---

## 11. Applications/Chem Module (Integrated)

### 11.1 Purpose & Design Goals
The Chemistry application module provides a complete, production-aligned quantum chemistry stack on top of TyxonQ’s core architecture. It is designed to:
- Offer a PySCF-like user experience for quantum algorithms (UCC, HEA, PUCCD, k-UpCCGSD) while remaining hardware-realistic.
- Unify device and numeric execution paths: device runtimes produce counts with explicit shots/noise handling and postprocessing, numeric runtimes produce exact statevector/civector baselines for verification and autograd.
- Provide reproducible and portable results by centralizing measurement semantics and mitigation in postprocessing.
- Bridge classical chemistry tooling (PySCF, OpenFermion) with TyxonQ IR, devices, and compilers.

This module aims to be the “quantum-computing counterpart” to PySCF in terms of clarity and completeness, but engineered for real-device workflows (shots, noise, mitigation) and for direct numeric baselines in the same API.

### 11.3 Architecture Diagram (Chem scope)
```mermaid
flowchart LR
  subgraph Chem
    ALG[algorithms (HEA/UCC/PUCCD/k-UpCCGSD)] --> RTD[runtimes: device]
    ALG --> RTN[runtimes: numeric]
    MOL[molecule & chem hamiltonians] --> ALG
  end

  RTD --> DEV[devices (simulators/hardware)] --> POST[postprocessing]
  RTN --> QL[libs.quantum_library]
  RTN --> NUM[numerics backends]

  ALG -. uses .-> CL[libs.circuits_library]
  ALG -. encodes .-> HENC[libs.hamiltonian_encoding]
  HENC --> OF[OpenFermion]
  MOL --> PYSCF[PySCF]
```

说明：HEA/UCC 等算法仅负责策略与 API；设备/数值分流发生在 runtime；设备路径输出 counts 由 postprocessing 计算期望和缓解；数值路径直接使用 `quantum_library` 与 NumericBackend 进行向量化计算。

### 11.4 Overview & Key Features
- Dual device/numeric execution paths with consistent semantics:
  - Device path: explicit shots/noise flags, counts-first return, expectations via postprocessing.
  - Numeric path: direct statevector/civector evaluation for baselines, gradients (incl. PyTorch autograd).
- HEA improvements:
  - Path-aware defaults in kernel (analytic path allows higher maxiter), device/local default shots=0, hardware default shots=2048.
  - Noise and postprocessing are transparently forwarded in `hea_device_runtime.py`.
- UCC family:
  - Device runtime groups Hamiltonian terms and delegates expectations to postprocessing; numeric runtime manages civector/statevector.
  - `kupccgsd.py` aligns with TCC: initializes run-time lists (`init_guess_list`, `e_tries_list`, `opt_res_list`) and selects the best result.
  - `puccd.py` reintroduces `make_rdm1/rdm2` (with `basis="AO"`), stabilizes H4 via paired excitations; random-integral tests use civector-based RDM references.
- Hamiltonian builders:
  - Unified sparse/mpo/fcifunc outputs; hcb+fcifunc accepts both int and (na, nb); mpo provides a light wrapper with `eval_matrix()` for tests.
- Numeric backend bridging:
  - `pauli_io.csc_to_coo` bridges through `get_backend(None)`; `tyxonq.__getattr__` lazily exposes `backend/rdtypestr` to maintain compatibility.
- Dynamics:
  - `dynamics_numeric.py` precomputes/lazily caches dense term matrices and property operators to avoid repetitive MPO builds and heavy logging.

### 11.5 Representative Code

初态制备（设备 vs 数值 vs PySCF civector 对齐）：
```20:33:tests_applications_chem/tests_project/test_state_prep_chain_vs_numeric.py
    circ = get_device_init_circuit(n_qubits, n_elec_s, mode, givens_swap=givens_swap)
    eng = StatevectorEngine()
    psi_device = np.asarray(eng.state(circ), dtype=np.complex128)

    psi_numeric = get_numeric_init_circuit(n_qubits, n_elec_s, mode, givens_swap=givens_swap)

    ci_strings = get_ci_strings(n_qubits, n_elec_s, mode == "hcb")
    civ = get_init_civector(len(ci_strings))
    psi_pyscf = get_numeric_init_circuit(
        n_qubits, n_elec_s, mode, civector=civ, givens_swap=givens_swap
    )
```

动态演化（VQD/pVQD/Trotter 数值路径）：
```28:44:tests_applications_chem/tests_project/test_dynamics.py
    te = TimeEvolution(
        ham_terms_spin,
        basis_spin,
        n_layers=n_layers,
        eps=1e-5,
    )
    te.add_property_op("Z", Op("Z", "spin"))
    te.add_property_op("X", Op("X", "spin"))
    te.include_phase = algorithm == "include_phase"
```

哈密顿量多形态（sparse/mpo/fcifunc）验证：
```27:44:tests_applications_chem/tests_project/test_hamiltonian.py
    hamiltonian = get_h_from_hf(hf, mode=mode, htype=htype)
    if htype == "mpo":
        hamiltonian = mpo_to_quoperator(hamiltonian).eval_matrix()
    else:
        hamiltonian = np.array(hamiltonian.todense())

    e_nuc = hf.energy_nuc()
    if mode in ["fermion", "qubit"]:
        fci_e, _ = fci.FCI(hf).kernel()
        np.testing.assert_allclose(np.linalg.eigh(hamiltonian)[0][0] + e_nuc, fci_e, atol=1e-6)
```

### 11.6 Comparison: TyxonQ Chem vs PySCF vs PennyLane
- PySCF：
  - 电子结构参考与数值基线（HF/FCI/RDM），不关心量子设备；我们在 `pyscf_civector` 与 `hamiltonian_builders` 中封装其能力，供数值与设备路径对齐
  - TyxonQ 在 PySCF 之上提供：IR、设备/编译/后处理链、shots/噪声/缓解；化学应用从直接数值扩展到“可上机”的完整系统
- PennyLane（qchem）：
  - PL 偏向期望直出与 QNode 接口；TyxonQ 强化 counts→postprocessing 的设备现实一致性；编译与 IR 分层更清晰（见前文第 5 节）
  - 在化学算子/激发构造方面，我们与 OpenFermion/自研生成器结合，提供 UCCSD/k-UpCCGSD/pUCCD 等变体，利于与设备链配合

### 11.7 Hardware–Simulator Seamless Migration
- 统一的算法 API（`kernel/energy/energy_and_grad`）对上层透明；运行时将计算路由到设备或数值引擎
- shots 策略：本地/模拟器默认 0，真实硬件默认 2048；设备路径支持显式噪声开关与配置
- 后处理层统一负责期望计算和读出缓解，模拟与真机结果对齐
- 数值路径保持与设备路径等价的哈密顿量与测量语义，支持 PyTorch autograd 与直接 statevector/civector 演化，用于快速验证与梯度基线

### 11.8 Design Impact Summary
- 化学模块已从“直连数值”重构为“设备/数值双栈”的工程化形态：更贴近真实硬件、同时保留科研可比性
- 通过 `hamiltonian_builders` 的多形态导出与 `runtimes` 的分层，测试与应用层均能自由切换链路，不牺牲一致性
- HEA 与 UCC 的 shots/噪声/后处理策略为真实设备实验预留默认与参数化接口，减少迁移摩擦
