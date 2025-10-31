==================
Pulse Programming
==================

TyxonQ Pulse Programming Guide - 脉冲级量子控制完整文档

.. contents:: 目录
   :depth: 3
   :local:

概述
====

TyxonQ 提供业界最完整的 Pulse 编程支持，采用**双路双模双格式**架构：

核心特性
--------

🔀 **双路执行** 
   - **本地模拟**: ``provider="simulator"`` - 基于物理模型的精确模拟
   - **云端真机**: ``provider="tyxonq"`` - 提交 TQASM 到真实量子硬件

📊 **双模式编程** 
   - **模式 A（链式调用）**: Gate Circuit → Pulse Compiler → Execution
   - **模式 B（直接数值）**: Hamiltonian → pulse_simulation → Evolved State

📦 **双格式输出** 
   - **pulse_ir** (TyxonQ Native): 保留 homebrew_s2 对象，支持 PyTorch autograd
   - **tqasm** (TQASM 0.2): 文本格式，云端兼容，符合国际标准

Pulse 表示的三种形式
===================

1️⃣ Gate-level（抽象层）
-----------------------

高层量子门操作，适合算法设计和教学演示。

.. code-block:: homebrew_s2

   from tyxonq import Circuit
   
   c = Circuit(1)
   c.x(0)  # 高层门操作
   # ops = [("x", 0)]

2️⃣ Pulse-level with References（符号引用，默认）
------------------------------------------------

保留 homebrew_s2 waveform 对象，支持梯度计算和灵活修改。

.. code-block:: homebrew_s2

   from tyxonq.compiler.pulse_compile_engine import PulseCompiler
   
   compiler = PulseCompiler()
   pulse_circuit = compiler.compile(
       c,
       output="pulse_ir",
       inline_pulses=False  # 默认值
   )
   
   # 编译结果：
   # ops = [("pulse", 0, "rx_q0_12345", {params})]
   # metadata["pulse_library"] = {"rx_q0_12345": Drag(...)}

**特点**:

- ✅ 保持 homebrew_s2 waveform 对象
- ✅ 支持 PyTorch autograd（梯度计算）
- ✅ 依赖 ``metadata["pulse_library"]`` 传递
- ✅ 快速编译，灵活修改

**适用场景**:

- 本地模拟优化
- VQE 参数训练
- Pulse 波形调试
- PyTorch/TensorFlow 集成

3️⃣ Pulse-level Inlined（完全展开，序列化友好）
----------------------------------------------

完全自包含，适合云端提交和文件保存。

.. code-block:: homebrew_s2

   pulse_circuit = compiler.compile(
       c,
       output="pulse_ir",
       inline_pulses=True  # 完全内联
   )
   
   # 编译结果：
   # ops = [("pulse_inline", 0, {"type": "drag", "args": [...]}, {params})]

**特点**:

- ✅ 自包含（不依赖 metadata）
- ✅ 可序列化为 JSON
- ✅ 云端兼容（TQASM 导出）
- ❌ 失去 homebrew_s2 对象灵活性
- ❌ 不支持 autograd

**适用场景**:

- 云端提交（TQASM）
- 文件保存与加载
- 跨进程通信
- 调试与可视化

Pulse Lowering 详解
===================

什么是 Lowering？
-----------------

**Lowering = 内联展开（Inlining）**

::

   pulse（符号引用）    →    pulse_inline（完全展开）
        ↓                            ↓
   依赖 metadata 查找          所有信息都在 op 本身

为什么默认不执行 Lowering？
--------------------------

1. **保持灵活性**: homebrew_s2 对象可以修改（如调整 amplitude）
2. **支持 autograd**: PyTorch tensor 梯度链不会断裂
3. **性能优化**: 避免重复序列化/反序列化
4. **兼容现有引擎**: StatevectorEngine 优先设计支持 pulse 模式

何时应该执行 Lowering？
-----------------------

.. list-table:: Lowering 使用场景
   :header-rows: 1
   :widths: 30 20 50

   * - 场景
     - inline_pulses
     - 理由
   * - 本地模拟
     - ``False``
     - 保持对象，支持 autograd
   * - 云端提交
     - ``True``
     - 序列化友好，TQASM 导出
   * - 文件保存（JSON）
     - ``True``
     - JSON 不支持 homebrew_s2 对象
   * - 文件保存（Pickle）
     - ``False``
     - Pickle 保留对象
   * - 参数优化（VQE）
     - ``False``
     - PyTorch requires_grad
   * - 调试可视化
     - ``True``
     - 查看完整数据

序列化指南
==========

JSON 序列化（跨语言，文本）
---------------------------

.. code-block:: homebrew_s2

   from tyxonq.compiler.pulse_compile_engine import save_pulse_circuit, load_pulse_circuit
   
   # 编译（必须 inline_pulses=True）
   pulse_circuit = compiler.compile(c, inline_pulses=True)
   
   # 保存为 JSON
   save_pulse_circuit(pulse_circuit, "pulse.json", format="json")
   
   # 加载
   loaded = load_pulse_circuit("pulse.json", format="json")

**Advantages**:

- ✅ Human-readable
- ✅ Cross-language compatibility
- ✅ Version control friendly (Git diff)
- ✅ Cloud API support

**Disadvantages**:

- ❌ Requires inline_pulses=True
- ❌ Loses homebrew_s2 objects
- ❌ Larger file size

Pickle Serialization (homebrew_s2 Native, Binary)
---------------------------------------------

.. code-block:: homebrew_s2

   # Compile (can use inline_pulses=False)
   pulse_circuit = compiler.compile(c, inline_pulses=False)
   
   # Save as Pickle
   save_pulse_circuit(pulse_circuit, "pulse.pkl", format="pickle")
   
   # Load (fully restore homebrew_s2 objects)
   loaded = load_pulse_circuit("pulse.pkl", format="pickle")

**Advantages**:

- ✅ Preserves homebrew_s2 objects (waveform instances)
- ✅ No need for inline_pulses=True
- ✅ Fast serialization
- ✅ Supports autograd

**Disadvantages**:

- ❌ homebrew_s2-only
- ❌ homebrew_s2 version sensitive
- ❌ Binary, not human-readable

Complete Usage Examples
========================

Example 1: Local VQE Optimization (pulse mode + PyTorch)
---------------------------------------------------------

.. code-block:: homebrew_s2

   import torch
   from tyxonq import Circuit, waveforms
   from tyxonq.compiler.pulse_compile_engine import PulseCompiler
   from tyxonq.numerics.context import set_backend
   
   set_backend("pytorch")
   
   # Parameterized Pulse amplitude
   amp = torch.tensor([1.0], requires_grad=True)
   
   def create_pulse_circuit(amp_val):
       c = Circuit(2)
       c.h(0)
       
       # Add parameterized Pulse calibration
       compiler = PulseCompiler()
       x_pulse = waveforms.Drag(amp=amp_val, duration=160, sigma=40, beta=0.2)
       compiler.add_calibration("x", [1], x_pulse)
       
       c.x(1)  # Use custom Pulse
       c.cx(0, 1)
       
       # Compile (preserve objects, support autograd)
       return compiler.compile(
           c,
           output="pulse_ir",
           inline_pulses=False,  # Keep homebrew_s2 objects
           calibrations=compiler.get_calibrations()
       )
   
   # VQE optimization loop
   optimizer = torch.optim.Adam([amp], lr=0.01)
   
   for step in range(100):
       circuit = create_pulse_circuit(amp)
       energy = circuit.run().expectation("Z0")  # Gradient auto-propagates!
       
       loss = energy
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()

Example 2: Cloud Submission (pulse_inline + TQASM)
---------------------------------------------------

.. code-block:: homebrew_s2

   from tyxonq import Circuit
   from tyxonq.compiler.pulse_compile_engine import PulseCompiler, save_pulse_circuit
   
   c = Circuit(2)
   c.h(0)
   c.cx(0, 1)
   c.measure_z(0)
   c.measure_z(1)
   
   compiler = PulseCompiler()
   
   # Step 1: Compile to pulse_inline (cloud-compatible)
   pulse_circuit = compiler.compile(
       c,
       device_params={
           "qubit_freq": [5.0e9, 5.1e9],
           "anharmonicity": [-330e6, -320e6]
       },
       output="pulse_ir",
       inline_pulses=True,  # Required! Cloud needs fully expanded
       mode="pulse_only"
   )
   
   # Step 2: Export to TQASM 0.2
   tqasm_code = compiler.compile(
       c,
       device_params={
           "qubit_freq": [5.0e9, 5.1e9],
           "anharmonicity": [-330e6, -320e6]
       },
       output="tqasm",  # TQASM format
       mode="pulse_only"
   )
   
   # Step 3: Submit to cloud
   # result = pulse_circuit.device(provider="tyxonq", device="homebrew_s2").run()

Example 3: Mode B - Direct Numerical Simulation
------------------------------------------------

.. code-block:: homebrew_s2

   import numpy as np
   from tyxonq.libs.quantum_library import pulse_simulation
   from tyxonq import waveforms
   from tyxonq.numerics.api import get_backend
   
   backend = get_backend("numpy")
   
   # Initial state: |0⟩
   psi_0 = backend.array([1.0, 0.0], dtype=complex)
   
   # Create DRAG pulse
   x_pulse = waveforms.Drag(
       amp=1.0,
       duration=160,
       sigma=40,
       beta=0.2
   )
   
   # Directly evolve quantum state (Mode B, bypass compiler)
   psi_x = pulse_simulation.evolve_pulse_hamiltonian(
       initial_state=psi_0,
       pulse_waveform=x_pulse,
       qubit=0,
       qubit_freq=5.0e9,
       drive_freq=5.0e9,
       anharmonicity=-330e6,
       T1=80e-6,    # Amplitude damping time
       T2=120e-6,   # Decoherence time
       backend=backend
   )

Decision Flowchart
==================

.. mermaid::

   graph TD
       A[Start] --> B{Cloud execution needed?}
       B -->|Yes| C[inline_pulses=True]
       C --> D[Export TQASM]
       D --> E[Cloud submission]
       
       B -->|No| F{Parameter optimization VQE/QAOA?}
       F -->|Yes| G[inline_pulses=False]
       G --> H[Preserve objects]
       H --> I[PyTorch autograd]
       
       F -->|No| J{File save needed?}
       J -->|JSON| K[inline_pulses=True]
       K --> L[save_pulse_circuit format=json]
       
       J -->|Pickle| M[inline_pulses=False]
       M --> N[save_pulse_circuit format=pickle]

Comparison with Other Frameworks
=================================

.. list-table:: Pulse Programming Feature Comparison
   :header-rows: 1
   :widths: 25 15 15 15 15 15

   * - Feature
     - TyxonQ
     - Qiskit Pulse
     - QuTiP-qip
     - Cirq
     - Pulser
   * - Dual-mode support
     - ✅ Chain+Direct
     - ❌ Chain only
     - ❌ Direct only
     - ❌ Chain only
     - ❌ Direct only
   * - Dual-format output
     - ✅ pulse_ir+TQASM
     - ✅ Qiskit+QASM
     - ❌ homebrew_s2 only
     - ✅ Cirq+JSON
     - ❌ homebrew_s2 only
   * - Dual execution backend
     - ✅ Local+Cloud
     - ✅ Local+IBM cloud
     - ❌ Local only
     - ✅ Local+Google cloud
     - ✅ Local+Pasqal cloud
   * - PyTorch autograd
     - ✅ Native support
     - ❌ Not supported
     - ❌ Not supported
     - ❌ Not supported
     - ❌ Not supported
   * - Serialization
     - ✅ JSON+Pickle
     - ✅ Qiskit Objects
     - ❌ Pickle only
     - ✅ JSON
     - ❌ Limited support
   * - Physical realism
     - ✅ T1/T2/detuning
     - ✅ Partial support
     - ✅ Full support
     - ✅ Partial support
     - ✅ Neutral atom physics

Best Practices Summary
======================

Scenario 1: Algorithm Research (Local)
---------------------------------------

.. code-block:: homebrew_s2

   # Use pulse mode + NumPy
   output="pulse_ir", inline_pulses=False
   provider="simulator", backend="numpy"

Scenario 2: Parameter Optimization (VQE)
-----------------------------------------

.. code-block:: homebrew_s2

   # Use pulse mode + PyTorch autograd
   output="pulse_ir", inline_pulses=False
   provider="simulator", backend="pytorch"
   requires_grad=True  # waveform parameters

Scenario 3: Cloud Submission
-----------------------------

.. code-block:: homebrew_s2

   # Use pulse_inline + TQASM
   output="tqasm", inline_pulses=True
   provider="tyxonq", device="homebrew_s2"

Scenario 4: File Save/Load
---------------------------

.. code-block:: homebrew_s2

   # JSON (cross-language)
   inline_pulses=True
   save_pulse_circuit(format="json")
   
   # Pickle (homebrew_s2 native)
   inline_pulses=False
   save_pulse_circuit(format="pickle")

API Reference
=============

Compiler API
------------

.. autoclass:: tyxonq.compiler.pulse_compile_engine.PulseCompiler
   :members:
   :undoc-members:

Serialization API
-----------------

.. autofunction:: tyxonq.compiler.pulse_compile_engine.save_pulse_circuit
.. autofunction:: tyxonq.compiler.pulse_compile_engine.load_pulse_circuit
.. autofunction:: tyxonq.compiler.pulse_compile_engine.serialize_pulse_circuit_to_json
.. autofunction:: tyxonq.compiler.pulse_compile_engine.deserialize_pulse_circuit_from_json

Physical Simulation API (Mode B)
---------------------------------

.. autofunction:: tyxonq.libs.quantum_library.pulse_simulation.evolve_pulse_hamiltonian
.. autofunction:: tyxonq.libs.quantum_library.pulse_simulation.compile_pulse_to_unitary
.. autofunction:: tyxonq.libs.quantum_library.pulse_simulation.build_pulse_hamiltonian

Noise Modeling API
------------------

ZZ Crosstalk
~~~~~~~~~~~~

.. autofunction:: tyxonq.libs.quantum_library.noise.zz_crosstalk_hamiltonian

Qubit Topology
~~~~~~~~~~~~~~

.. autofunction:: tyxonq.libs.quantum_library.pulse_physics.get_qubit_topology
.. autofunction:: tyxonq.libs.quantum_library.pulse_physics.get_crosstalk_couplings

References
==========

- QuTiP-qip Processor Model (Quantum 6, 630, 2022)
- TQASM 0.2 Specification
- TensorCircuit Pulse Implementation
- Scully & Zubairy, "Quantum Optics" (1997)

Related Documentation
=====================

- :doc:`../../../examples/index` - Pulse programming examples
- :doc:`../../../tutorials/intermediate/pulse_programming_basics` - Pulse programming basics (P0.1-P0.5)
- :doc:`../../../tutorials/advanced/pulse_three_level` - Three-level system simulation (P1.1)
- :doc:`../../../tutorials/advanced/pulse_zz_crosstalk` - ZZ crosstalk noise modeling (P1.2)
- :doc:`../../api/compiler/pulse_compile_engine` - Pulse compiler API
- :doc:`../../api/libs/quantum_library/pulse_simulation` - Pulse physical simulation API
- :doc:`../../../technical_references/whitepaper` - TyxonQ technical whitepaper

.. note::

   This documentation corresponds to TyxonQ v0.2.0+
   
   Last updated: 2025-10-30
   
   Author: TyxonQ Development Team
