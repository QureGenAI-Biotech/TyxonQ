# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning (https://semver.org/spec/v2.0.0.html).

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
- **China Mobile QCOS Integration**: Direct connectivity to quantum hardware on ecloud via wuyue_plugin without local Docker
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