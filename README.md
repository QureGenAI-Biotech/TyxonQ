<h2><p align="center">TyxonQ</p></h2>
<h3><p align="center">A Modular Full-stack Quantum Software Framework on Real Machine</p></h3>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Real Quantum Hardware](https://img.shields.io/badge/Quantum%20Hardware-Homebrew__S2-brightgreen)](https://www.tyxonq.com/)

For Chinese Introduction, see: [中文README](README_cn.md).
For Japanese Introduction, see: [日本語README](README_jp.md).

TyxonQ 太玄量子 is a next‑generation quantum programming framework with a stable IR, pluggable compiler, unified device abstraction (simulators and hardware), a single numerics backend interface (NumPy/PyTorch/CuPyNumeric), and a device runtime friendly postprocessing layer. It is designed to mirror real devices while remaining simple for engineers and scientists.

### Core goals
- **System‑architect‑friendly, hardware‑realistic programming model**: stable IR + chain pipeline mirroring real device execution; clear contracts for compiler, devices, and postprocessing; closest‑to‑hardware code path.

- **Quantum AIDD (Quantum Computational Chemistry for advanced AI Drug Design)**: PySCF‑like UX, hardware‑realistic execution; familiar molecule/ansatz APIs route to device or numerics without code changes. Mission: prioritize drug design—provide missing microscopic Quantum Chemistry data and robust computational tools for AI drug discovery; roadmap includes drug design–oriented Hamiltonians, method optimization, and AI‑for‑QC.

- **Dual paths**: Hamiltonians, measurement grouping, shot planning, device execution (shots/noise) and exact numerics (statevector/MPS) with shared semantics.

- **Extensible domain layer**: algorithms and chem libs are modular for specialized extensions.

***Try Real Quantum Computer Right Now！***: [Getting a Key](https://www.tyxonq.com/) to register and obtain your API key. 
Directly use the TyxonQ cloud task submission API. For details, see the example: [examples/cloud_api_task.py](cloud_api_task.py)


## Quick start

### Minimal circuit → simulator / hardware
```python
import tyxonq as tq
from tyxonq.postprocessing import metrics
import getpass
tq.set_backend("numpy")

# Configure quantum hardware access
#API_KEY = getpass.getpass("Input your TyxonQ API_KEY:")
#tq.set_token(API_KEY) # Get from https://www.tyxonq.com

# Build once
c = tq.Circuit(2).h(0).cx(0, 1).measure_z(0).measure_z(1)

# Simulator path
sim = (
    c.compile()
     .device(provider="simulator", device="statevector", shots=4096)
     .postprocessing(method=None)
     .run()
)

# Hardware path (example target)
hw = (
    c.compile(output="qasm")
     .device(provider="tyxonq", device="homebrew_s2", shots=4096)
     .run()
)

def counts_of(res):
    payload = res if isinstance(res, dict) else (res[0] if res else {})
    return payload.get("result", {})

ez_sim = metrics.expectation(counts_of(sim), z=[0, 1])
ez_hw  = metrics.expectation(counts_of(hw),  z=[0, 1])
print("E[Z] (sim)", ez_sim)
print("E[Z] (hw) ", ez_hw)
```

### Minimal Quantum Chemistry (PySCF‑style)
```python
# pip install pyscf  # required for UCCSD example
import tyxonq as tq
from tyxonq.applications.chem.algorithms.uccsd import UCCSD
from tyxonq.applications.chem import molecule

tq.set_backend("numpy")

# Preset H2 molecule (RHF defaults handled inside UCCSD)
ucc = UCCSD(molecule.h2)

# Device chain on simulator (counts → energy)
e = ucc.kernel(shots=2048, provider="simulator", device="statevector")
# Device chain on real machine (counts → energy)
#e = ucc.kernel(shots=2048, provider="tyxonq", device="homebrew_s2")
print("UCCSD energy (device path):", e)
```


## Installation
```bash
pip install tyxonq
# or from source
uv build && uv pip install dist/tyxonq-*.whl
```

## 🔑 Quantum Hardware Setup
### Getting API Access
1. **Apply for API Key**: Visit [TyxonQ Quantum AI Portal](https://www.tyxonq.com/) 
to register and obtain your API key
2. **Hardware Access**: Request access to **Homebrew_S2** quantum processor through 
API [TyxonQ QPU API](https://www.tyxonq.com)

### Hardware API Configuration
Set up your API credentials:

```python
import tyxonq as tq
import getpass

# Configure quantum hardware access
API_KEY = getpass.getpass("Input your TyxonQ API_KEY:")
tq.set_token(API_KEY) # Get from https://www.tyxonq.com
# legacy style
# apis.set_token(API_KEY) # Get from https://www.tyxonq.com
```

## 📖 Technical Documentation

### TyxonQ Technical Whitepaper
For developers, researchers, and engineers interested in the deep technical architecture and innovations of TyxonQ, we strongly recommend reading our comprehensive technical whitepaper:

**📋 [TYXONQ_TECHNICAL_WHITEPAPER.md](TYXONQ_TECHNICAL_WHITEPAPER.md)**

This document provides:
- **Novel architectural innovations**: Dual-path execution model, compiler-driven measurement optimization, and stable IR design
- **Quantum AIDD technical details**: AI-driven drug discovery applications with hardware-realistic quantum chemistry stack
- **System design principles**: Cross-vendor portability, counts-first semantics, and single numeric backend abstraction
- **Academic-quality analysis**: Comprehensive comparison with existing frameworks and research directions
- **Implementation details**: Core components, execution flows, and integration patterns

## Architecture
<p align="center">
  <img src="./docs/images/architect.png" alt="TyxonQ Architecture" width="100%">
</p>


### Key features
- **Chain API**: `Circuit.compile().device(...).postprocessing(...).run()`.
- **Compiler passes**: measurement rewrite/grouping, light‑cone simplify, shot scheduling.
- **Devices**: statevector/density‑matrix/MPS simulators and hardware drivers (e.g., `tyxonq:homebrew_s2`).
- **Numerics**: one ArrayBackend for NumPy/PyTorch/CuPyNumeric powering simulators and research kernels.
- **Libraries**: `libs/circuits_library` (templates: VQE/QAOA/trotter/state‑prep), `libs/quantum_library` (numeric kernels), `libs/hamiltonian_encoding` (OpenFermion I/O, encodings), `libs/optimizer` (interop).
- **Real Quantum Hardware Ready**: TyxonQ supports **real quantum machine execution** through our quantum cloud services powered by **QureGenAI**. Currently featuring the **Homebrew_S2** quantum processor, enabling you to run your quantum algorithms on actual quantum hardware, not just simulators.

- **Pulse-Level Control**: Support for both gate-level operations and **pulse-level signals** for advanced quantum control

- **Quantum API Gateway**: RESTful APIs for direct quantum hardware access

- **☁️ Quantum Cloud Services**: Scalable quantum computing as a service



### Quantum AIDD Key features
- **Algorithms**: HEA and UCC family (UCC/UCCSD/k‑UpCCGSD/pUCCD) with consistent energy/gradient/kernel APIs.
- **Runtimes**: device runtime forwards grouped measurements to postprocessing; numeric runtime provides exact statevector/civector (supports PyTorch autograd).
- **Hamiltonians**: unified sparse/MPO/FCI‑function outputs; convenient molecule factories (`applications/chem/molecule.py`).
- **Measurement and shots**: compiler‑driven grouping and shot scheduling enable deterministic, provider‑neutral execution.
- **Properties**: RDM1/2 and basic property operators; dynamics numeric path caches MPO/term matrices to avoid rebuilds.
- **Bridges**: OpenFermion I/O via `libs/hamiltonian_encoding`; tight interop with PySCF for references and integrals.
- **Chem libs**: `applications/chem/chem_libs/` including `circuit_chem_library` (UCC family ansatz), `quantum_chem_library` (CI/civector ops), `hamiltonians_chem_library` (HF/integrals → Hamiltonians).

- **AIDD (AI Drug Design) field Feature**
  - Drug‑design‑oriented Hamiltonians and workflows (ligand–receptor fragments, solvent/embedding, coarse‑grained models) prioritized for AI Drug Design.
  - Method optimization for AIDD tasks: tailored ansatz/measurement grouping, batched parameter‑shift/QNG, adaptive shot allocation.
  - AI‑for‑QC bridges: standardized data schemas and export of Quantum Chemistry field data (energies, RDMs, expectations,ansatz,active space,etc) for QC algorithms development.
  - Expanded properties and excited states (VQD/pVQD) aligned with spectroscopy and binding‑relevant observables.


## Dependencies
- Python >= 3.10 (supports Python 3.10, 3.11, 3.12+)
- PyTorch >= 1.8.0


## 📧 Contact & Support

- **Home**: [www.tyxonq.com](https://www.tyxonq.com)
- **Technical Support**: [code@quregenai.com](mailto:code@quregenai.com)
- **General Inquiries**: [bd@quregenai.com](mailto:bd@quregenai.com)
- **Issue**: [github issue](https://github.com/QureGenAI-Biotech/TyxonQ/issues)

#### 微信公众号 | Official WeChat
<img src="docs/images/wechat_offical_qrcode.jpg" alt="TyxonQ 微信公众号" width="200">

#### 开发者交流群 | Developer Community
<img src="docs/images/developer_group_qrcode.png" alt="TyxonQ 开发者交流群" width="200">

*扫码关注公众号获取最新资讯 | Scan to follow for latest updates*  
*扫码加入开发者群进行技术交流 | Scan to join developer community*

### Development Team
- **QureGenAI**: Quantum hardware infrastructure and services
- **TyxonQ Core Team**: Framework development and optimization
- **Community Contributors**: Open source development and testing

## License
TyxonQ is open source, released under the Apache License, Version 2.0.
