# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning (https://semver.org/spec/v2.0.0.html).


### [Unreleased]

### Added
- **SpinQ provider**: direct TyxonQ IR → core compile/topology routing → SpinQit native Circuit/executable lowering, with independent credentials, platform availability, counts-first results, offline/mock tests and opt-in live smoke tests. Qiskit/QASM are not used as SpinQ circuit intermediates.
- **SpinQLab pulse feasibility**: JSON-friendly waveform/rectangular-pulse parameter adaptation and fake SDK lifecycle tests. Only native rectangular pulses are accepted by the driver; arbitrary sampled waveforms remain explicitly unsupported pending device validation.

### [1.5.0] - 2026-11-22
### Added
- **Device-agnostic quantum-dynamics validation**: exact evolution, Trotter circuits, kicked response, spectroscopy and finite-shot statistics are exercised by the offline `examples/qd/` workflow. Q8 adds analytic/ED references, discretization and time-window convergence checks, and shot standard errors/confidence intervals.
- **Floquet Ising dynamics** (`applications/chem/dynamics/floquet.py`): reproducible disorder sampling, Néel-state preparation, nearest-neighbor Floquet circuit construction and Z-basis autocorrelators. Observable reduction reuses the common Pauli grouping and counts-first postprocessing rather than introducing provider-specific bit-order logic.
- **Reproducible experiment-pairing examples**: Q9 (single-qubit Rabi) and Q10 (20-qubit Floquet Ising) default to offline analysis and use versioned manifests, circuit/record hashes, source hashes, explicit hardware confirmation, reviewed-pilot gates and append-only JSONL records. Reports can be rebuilt from raw counts; existing valid records support safe resume. The dynamics core remains independent of the hardware provider used for a recorded run.
- **Raw benchmark reports and public-data provenance**: added licensed public experimental datasets, raw paired-point CSVs and formal Markdown reports. The Rabi run contains 101 hardware circuits/10,100 shots and retains its failed independent-validation status. The many-body run contains 8 pilot circuits plus 264 batch circuits/26,400 shots; its unmitigated 20-qubit hardware curve did not agree with the public experimental curve and is reported without claiming quantum advantage.
- **Native CUDA and Apple Metal SBD selected-CI solvers**: `tyxonq.numerics.sbd` is now the common Selected Basis Diagonalization entry with sibling `cuda/` and `metal/` implementations. CUDA retains the native FP64 SBD path; Metal implements alpha/beta same-spin and mixed-spin H@C, S² and Davidson long-vector operations through MLX/MSL, with deterministic FP32 reductions, explicit GPU streams/completion, thick restart and independent residual validation. PR #21 added connected-row/pair-table kernels, projection caching and optional Metal opposite-spin 2-RDM acceleration without changing the CUDA SBD → Metal SBD → PyTorch CPU priority. The Metal GPU implementation is integrated from the implementation provided by the QubeChip organization at [QubeChip/sqd-gpu-metal](https://github.com/QubeChip/sqd-gpu-metal).
- **Single-variable SQD accelerator selection**: `solve_sci_gpu(..., backend="gpu"/"auto")` resolves native CUDA SBD → native MLX/Metal SBD → PyTorch CPU. Explicit `cpu`, `metal` and `cuda`/`sbd` modes are supported. Unavailable hardware or runtimes produce a warned, reported CPU fallback (`used_cpu_fallback`, `selection_reason`); kernel errors, non-convergence and residual/RDM failures still raise and never silently switch backend.
- **Real-device and whole-application SQD benchmarks**: `examples/sqd_metal_scaling.py` keeps fixed 40-cycle FP32 profiles separate from normal convergence, while `examples/sqd_gpu_benchmark.py` measures real-molecule CAS setup, solver, RDM, validation and whole wall time. The post-PR21 six-point, two-round H2O/NH3/CH4 sweep reproduced Metal crossover at 8,281 determinants for all three molecules; 4,356 remains a first-round reversal point. At 23,409 determinants, Metal delivered whole-workflow speedups over PyTorch CPU of 2.57×/2.67× (H2O), 1.98×/2.19× (NH3), and 2.18×/2.24× (CH4), reaching up to 2.67× including setup, solver, RDM and validation. All 36 paired workloads converged and passed residual/RDM acceptance; these single-CAS results do not claim full iterative-SQD or unmeasured CUDA performance.
- **Framework-wide numerical benchmark and optimization report**: added `examples/numerics_acceleration_benchmark.py` and `docs/NUMERICS_OPTIMIZATION_REPORT.md`, covering MPS, `quantum_info`, EWF, LUCJ, SKQD and readout with cold/warm timing, five or more raw samples, stage costs, result checks and memory measurements. Adoption requires at least 5% whole-workload median improvement in two independent processes; `docs/NUMERICS_ACCELERATION_AUDIT.md` now records adopted, deferred and platform-limited candidates.
- **Reproducible SBD research and release documentation**: added detailed Metal optimization history, independent CPU FP64 golden coverage, break-even measurements, rejected kernels and reproduction commands in `docs/SBD_METAL_OPTIMIZATION.md`, `docs/SBD_REPRODUCIBILITY.md` and `docs/SQD_GPU_BACKENDS.md`. The final PR21 SQD/EWF/LUCJ/backend regression set passed 213 tests with 27 hardware/environment-dependent skips; raw post-PR21 samples are archived in `data/sbd_benchmarks/multi_molecule_break_even_pr21_2026-11-22.json`.
- **Cross-platform and multi-machine Apple GPU research**: documented official Mojo/MAX support for custom kernels across Apple/NVIDIA/AMD, experimental AdaptiveCpp and alpha triton-msl routes, and MLX multi-host MPI/Ring/Thunderbolt-RDMA JACCL communication. This is a research route, not a claim that Metal SBD is already distributed; the production native CUDA and single-device Metal paths remain unchanged.

### Changed
- **Numerics acceleration architecture**: renamed local capability discovery to `numerics/accelerator.py`, added MLX to general automatic backend selection, and kept native SBD as an algorithm-specific implementation rather than a global array backend. Metal SQD uses mixed-precision FP32 long vectors with CPU setup and default CPU RDM; optional Metal RDM covers only the opposite-spin 2-RDM. CUDA remains FP64. No end-to-end native Metal RDM, multi-GPU or MPI/NCCL equivalence is claimed.
- **Target-spin and validation behavior**: Metal `spin_tolerance` is keyword-only with default `1e-4`; target roots are filtered by S², inadmissible spaces fail explicitly, and an unconstrained lowest-root solve retries after variational-bound failure. Production factorized RDM remains automatic, while `full_cpu` forces original PySCF spin-block RDMs as an independent golden. POSIX validation uses a persistent isolated worker with current-request inputs, thread/fork/failure recovery and explicit batch cleanup; other platforms retain one-shot CLI validation.
- **MPS backend and SVD execution**: the engine's backend now propagates through initialization, one-/two-qubit gates, truncation, reconstruction and statevector conversion. Broad SVD exception handling no longer silently moves tensors to NumPy; unsupported precision/device operations fail explicitly. Complete NumPy workload medians improved by 8.20%/9.55% (medium) and 13.04%/16.93% (large) in two rounds.
- **EWF DMET fragment-energy reuse**: `compute_dmet_energy` computes each fragment's WF/DMET contribution once and passes it to `dmet_total_energy`, preserving separate global corrections. Medium workloads improved by 5.12%/8.30% and large workloads by 13.31%/13.08%; benchmark energies are unchanged.
- **LUCJ compressed-DF optimization**: SciPy's first objective call now reuses the already recorded initial loss and gradient, eliminating one full PyTorch forward/backward per restricted or αβ optimization. Restricted targets are reused across stages. Tested whole workloads improved by 10.89–20.50%, with value-identical loss histories and unchanged autograd/SciPy CPU boundaries.
- **Unified GPU packaging**: CUDA, Metal, CuPy and MPS optional dependencies and native package data are declared by the root `pyproject.toml`; install with `pip install 'tyxonq[cuda]'` or `pip install 'tyxonq[metal]'`. The separate `packaging/sqd-gpu/pyproject.toml` and standalone `tyxonq-sqd-gpu` distribution were removed.
- **Flattened VQE execution**: removed the VQE `runtimes/` package and the legacy `runtime="numeric"/"device"` option. HEA and concrete UCC algorithms now select direct numerics with `shots=0` and circuit measurement with `shots>0`, with `provider`/`device` supplied at the public call site. The shared UCC base retains chemistry and wavefunction responsibilities only.

### Fixed
- **Hardware-batch provenance and resume guards**: Q9/Q10 now require a complete, valid pilot for the current manifest and reject stale or mixed records based on manifest, circuit, device, shots and (for Rabi) fitted-reference hashes. Retries skip only complete valid records, so failed pilots, incomplete counts or changed configurations cannot silently authorize or contaminate a batch report.

### [1.4.0] - 2026-09-11
### Added
- **Two-layer compiler architecture**: a device-agnostic `gate_compiler` core plus per-hardware `platforms/*` dialects (`tyxonq`/`lqcloud`/`guodun`). Each platform adapter delegates to the core for device-independent compilation, then translates the result into its own dialect — compilation logic is no longer duplicated per vendor
- **Typed step pipeline** (`gate_compiler/steps.py`, `specs.py`): compilation capabilities are expressed as typed specs (`DecomposeSpec`/`RoutingSpec`) that assemble an ordered `CompileStep` sequence (normalize → decompose → route → optimize → simplify), replacing the old string-based `compile_plan` DSL and its `NoOp*` placeholder passes
- **SABRE routing as a first-class step** (`gate_compiler/stages/layout/routing.py`): layout + SWAP insertion delegated to `qiskit.transpile`, executed in a local index space with a reproducible seed, preserving measurement logical order and never rewriting the source IR (routing info flows to the device as a runtime `initial_layout`)
- **Pure QASM2 exporter** (`gate_compiler/exporters/gate_qasm2_exporter.py`): zero-external-dependency IR → OpenQASM 2.0 text, aligned with the existing QASM3 exporter
- **Interop layer** (`gate_compiler/interop/`): `qiskit_interop` (IR ↔ Qiskit `QuantumCircuit`) and `mpo_converters` (tensor-network MPO → dense matrix), shared by routing (SABRE), `draw`/`depth`, QCOS submission and the simulator driver
- **Tests**: `tests_core_module/test_gate_qasm2_exporter.py` (QASM2 exporter coverage); Qiskit provider tests rewritten around `compile(output="qiskit")` + `qiskit_interop`

### Changed
- **Qiskit demoted from pseudo-platform to interop alias**: `platform="qiskit"` now normalizes to the `tyxonq` core (defaulting `output="qiskit"` when unspecified); the standalone `platforms/qiskit` compiler/dialect package was removed. Qiskit is an interoperability target and routing engine, not a hardware dialect
- **Three-layer separation of concerns**: `exporters/` (IR → standard text: QASM2/QASM3), `interop/` (IR ↔ external framework objects/formats: Qiskit, tensor-network MPO), `platforms/` (hardware dialects) are now semantically parallel
- **`output="qiskit"` no longer implies transpilation**: layout/routing is exclusively the `route=True` step's responsibility, making behavior predictable
- **`translation/` merged into `interop/`**: the (dormant) MPO converters moved to `gate_compiler/interop/mpo_converters.py` to consolidate all external-format conversion in one layer
- **LQCloud compiler delegates decomposition to the core**: `platforms/lqcloud` now routes matchgate/rotations decomposition + routing through `gate_compiler`, so the matchgate decomposition exists in exactly one place (resolves the prior duplicated-implementation pitfall)

### Fixed
- **`output="qasm2"` silently falling back to IR**: previously the QASM2 path detoured IR → Qiskit → transpile → QASM2 and could quietly return the IR object; it now emits a real OpenQASM 2.0 string via `GateQASM2Exporter` (benefits `homebrew_s2`/`quafu` submission)
- **`tests_examples` / `tests_mol_valid` regressions**: source-level fixes restored the example and molecule-validation suites to green

### [1.3.0] - 2026-09-01
### Added
- **MD / QM-MM ecosystem integration** (`tyxonq.applications.chem.interfaces`): TyxonQ as a quantum-chemistry force engine; all nuclear gradients reused from PySCF:
  - `qc_scanner`: unified energy/gradient facade (SQD/UCCSD/ROUCCSD/HEA), cluster & periodic-Ewald electrostatic embedding, MM back-reaction forces
  - `TyxonQCalculator` (ASE), `TyxonQDriver` (i-PI), `create_qmmm_ee_system` (OpenMM), `TyxonQMdiEngine` (MDI)
- **HEA/UCCSD real-hardware passthrough**: `qc_scanner(..., solver_kwargs={"runtime": "device", ...})` forwards options via `as_pyscf_solver(device_opts=...)` to `devices.base.run` (TyxonQ/QCOS/Quafu)
- **SQD real-hardware passthrough**: `lucj_sampler(runtime="device", provider=..., device=..., shots>0)` appends `measure_z` and submits the LUCJ circuit via `devices.base.run`, then reverses bitstring halves into PySCF order. SQD device options live on the *sampler* (not `solver_kwargs`) because SQD's quantum step is *sampling a circuit into counts*, not measuring a Hamiltonian
- **New examples** (`examples/qmmm/`): E1-E11 runnable tutorials (E11 = SQD in QM/MM: sampled frozen subspace, frozen-vs-refresh force-noise mechanism, SQD-driven AIMD, electrostatic embedding, device passthrough)
- **Tests**: `tests_applications_chem/` expanded to 57 cases

### Changed
- `pyproject.toml`: `md` extra covers `ase>=3.23`, `openmm>=8.1`, `openmmml>=1.7`
- **Simulator engine strictness (intentional)**: unknown ops raise `ValueError` and unsupported special ops raise `NotImplementedError` (no more silent skip); MPS `state()` returns native `MPSState`; every engine's `run(shots=0)` also returns `probabilities`. No public API signature changed
- **`applications/chem` restructured by domain** (pure code relocation; numerical behavior & public API unchanged): new `dynamics/` (time-evolution `evolution.py` + `models/{pyrazine,sbm}`), consolidated `algorithms/vqe/` (UCC family + HEA, `runtimes/`, and a `wavefunction/` CI/statevector library), and chem-level `hamiltonian_builders.py`; removed the old `chem_libs/` and `runtimes/` packages — only internal full-path imports moved.

### Fixed
- **Simulator engine op-dispatch single source of truth**: the three engines carried 8+ independent dispatch loops, silently dropping `y`/`z`/`t`/`tdg`/`cy` and silently skipping unknown ops (the real cause behind backlog #7 "missing cry"). Now one `_evolve` dispatch per engine (authoritative `gates.resolve_unitary`) shared by `run()`/`state()`/`expval()`; driver shots=0 reuses the same `run()` output; added `DensityMatrixEngine.state()/probabilities()`; `Circuit.state()` returns the true 2D density matrix (density_matrix) and delegates to the engine (MPS); `_expectation_density_matrix` computes true Tr(ρO)
- Device-runtime measurement bit-ordering: X/Y basis rotations sat at mirrored index `n-1-q` while aggregation reads `q`, corrupting sampling energies for 3+ qubits (up to ~0.25 Ha on water CAS(4,4)); now act on `q`
- `UCCDeviceRuntime.energy_and_grad`: the ±π/2 parameter-shift returned ~0 gradient for every UCC parameter (even-harmonic energy surface); replaced with the two-shift rule (exact for harmonics {2,4})
- Shots=0 analytic aggregation: multi-qubit ZZ was factorized into single-qubit ⟨Z⟩ products (exact only for product states; ~0.12 Ha bias on H4); now exact probability-based aggregation
- `QCScanner.set_mm_charges`: bare-SCF `add_mm_charges` returns a new object (not in-place); the dropped return value silently disabled embedding
- `apply_postprocessing` (shots=0): analytic `expectations`/`probabilities` stayed in the driver payload and silently degraded to a constant
- MDI `>COORDS`: engine now receives full-system coordinates and slices the QM subset internally
- SQD sampling-path bit-ordering: LUCJ raw samples are emitted in TyxonQ order `[α0..αN-1|β0..βN-1]` (qubit 0 = MSB), but `bitstring_matrix_to_integers` reads MSB-first while PySCF selected-CI is LSB-first, so feeding them straight into `run_sqd_fermion` reversed the orbital sequence (HF string `'11001100'` froze `[12],[12]` instead of `[3],[3]`, +3.02 Ha on water CAS(4,4)); added `reverse_bitstring_halves()`, applied by callers (`examples/h2o_sqd.py`, `lucj_sampler`) before `run_sqd_fermion` (which internalizes no reversal)

### Known Limitations
- MM back-reaction forces lack post-HF orbital-response terms (~4.3e-5 Ha/Bohr baseline bias); thermostatted MD is fine, strict NVE conservation diagnostics are not

### [1.2.0] - 2026-08-08
### Added
- **LUCJ-SQD workflow** (`tyxonq.applications.chem.algorithms.sqd` / `.lucj`):
  - SQD (Sample-based Quantum Diagonalization) fermionic solver: `run_sqd_fermion`, `solve_sci`, `diagonalize_fermionic_hamiltonian`, plus sampling / subsampling / post-selection and configuration-recovery utilities
  - LUCJ (Local Unitary Cluster Jastrow) ansatz: `build_lucj_circuit`, `initialize_lucj_parameters_from_ccsd`, Givens-schedule & topology helpers
  - PySCF integration: `SQDFCISolver`, `as_pyscf_solver`, `lucj_sampler`; closed-shell workflow correction
  - Example `examples/h2o_sqd.py`, doc `LUCJ_SQD.md`
- **RiverONE QML adapter** (`tyxonq.applications.qml.riverone`): example `examples/riverone_qml.py`, doc `RIVERONE.md`
- **LQCloud Provider**: Added optional `lqcloud==0.4.2` gate-circuit submission, dynamic device discovery, polling, cancellation, result normalization, example, and Chinese usage guide.
- **Guodun Provider**: Added optional `cqlib==1.3.11` integration for official QCIS gate circuits, native gate pulse templates, waveform queries, task submission, result polling, cancellation, and device discovery.

### [1.1.0] - 2026-05-07
### Added
- **Enhanced Qiskit Dialect Support**: Expanded gate conversion with 15+ gate types including x, y, z, s, sdg, t, tdg, ry, cy, cz, swap, iswap, rxx, ryy, rzz, and barrier
- **Improved QCOS Authentication**: Simplified credential management - removed sdk_code requirement, now using access_key + secret_key only (aligned with China Mobile WuYue platform 2026-04 update)

### Changed
- **Qiskit Dialect Conversion**: Comprehensive bidirectional conversion (to_qiskit/from_qiskit) supporting full gate set with proper parameter handling
- **QCOS Driver Options**: Removed whitelist filtering, now forwarding all options to wuyue Runner for better compatibility with new parameters (bit_info, qmachine_type, dry_run, initial_mapping, etc.)
- **QCOS_CHANGES Documentation**: Updated migration guide and installation instructions reflecting platform changes

### Fixed
- **QCOS Credential Flow**: Eliminated unnecessary License.init_license() call that was deprecated by China Mobile
- **Qiskit Gate Mapping**: Fixed cx/cnot equivalence and added proper two-qubit gate parameter handling

### [1.0.0] - 2026-02-15
### Added
- **Revolutionary Pulse-Level Quantum Control**: Complete pulse programming framework with dual-mode architecture (automatic gate→pulse + direct Hamiltonian evolution)
- **Industry-Leading Waveform Library**: 10+ waveform types including DRAG, Hermite, Blackman with physics-validated implementations
- **Three-Level System Support**: Realistic transmon qubit modeling with |2⟩ state leakage simulation
- **TQASM 0.2 + OpenPulse Export**: Full defcal support for cloud QPU deployment
- **China Mobile QCOS Integration**: Direct connectivity to quantum hardware on ecloud via the wuyue SDK without local Docker
- **Enhanced Quantum Chemistry Performance**: Smart caching mechanisms, batch compilation, and hybrid GPU acceleration
- **Quantum Natural Gradient (QNG)**: Advanced optimization with Fubini-Study metric implementation
- **HOMO-LUMO Gap Analysis**: New molecular property computation capabilities
- **GQE Drug Design Transfer Learning**: Research project for quantum-enhanced drug discovery
- **Comprehensive Documentation Upgrade**: Multilingual docs, API references, and technical whitepaper enhancements

### Changed
- **Core Architecture**: Compiler data structure modernization and improved IR handling
- **Device Abstraction**: Enhanced provider resolution and unified device interface
- **Numerics Backend**: Optimized cache mechanisms and performance improvements
- **Runtime Systems**: Refactored UCC/HEA runtimes with grouped compilation and batch processing
- **Error Handling**: Improved error messages and validation across all components

### Performance Improvements
- **UCCSD Execution**: shots=0 performance optimization using intelligent caching
- **Batch Processing**: Significant speedup in grouped measurement compilation

### Fixed
- Critical bugs in UCCSD active space handling
- Statevector operation inconsistencies
- Compiler API bugs in circuit compilation
- Parameter priority resolution in device execution

### [0.3.0] - 2025-08-18
### Added
- comprehensive pulse-level control capabilities for advanced quantum experiments and precise quantum manipulation.

## [0.2.1] - 2025-08-08
### Added
- MCP service integration and multi-tool invocation support in the Homebrew_S2 HTTP API.

### Changed
- Declared official Python support: 3.10+ (tested on 3.10–3.12).
- Docs: Updated localized READMEs (Chinese and Japanese).
- Minor docs typos

## [0.1.1] - 2025-07-21
### Added
- Real quantum hardware (Homebrew_S2) execution path and quantum task management system
- Example `examples/simple_demo_1.py`
### Changed
- Docs: README hardware setup guidance
### Fixed
- Minor docs typos

## [0.1.0] - 2025-01
### Added
- Initial preview release: circuit, compiler, backends, autodiff
