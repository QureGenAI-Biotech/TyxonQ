# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning (https://semver.org/spec/v2.0.0.html).

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