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

.. code-block:: python

   import torch
   from tyxonq import Circuit, waveforms
   from tyxonq.compiler.api import compile
   from tyxonq.numerics.context import set_backend
   
   set_backend("pytorch")
   
   # Parameterized pulse amplitude
   amp = torch.tensor([1.0], requires_grad=True)
   
   def create_pulse_circuit(amp_val):
       c = Circuit(2)
       c.h(0)
       c.x(1)  # Will be compiled to pulse
       c.cx(0, 1)
       
       # Enable pulse mode with parameterized amplitude
       c.use_pulse(device_params={
           "qubit_freq": [5.0e9, 5.05e9],
           "anharmonicity": [-330e6, -330e6]
       })
       
       return c
   
   # VQE optimization loop
   optimizer = torch.optim.Adam([amp], lr=0.01)
   
   for step in range(100):
       circuit = create_pulse_circuit(float(amp))
       energy = circuit.run().expectation("Z0")  # Gradient auto-propagates!
       
       loss = energy
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()

Example 2: Cloud Submission (pulse_inline + TQASM)
---------------------------------------------------

.. code-block:: python

   from tyxonq import Circuit
   from tyxonq.compiler.api import compile
   
   c = Circuit(2)
   c.h(0)
   c.cx(0, 1)
   c.measure_z(0)
   c.measure_z(1)
   
   # Enable pulse mode
   c.use_pulse(device_params={
       "qubit_freq": [5.0e9, 5.1e9],
       "anharmonicity": [-330e6, -320e6]
   })
   
   # Step 1: Compile to TQASM (auto-converts to tyxonq_homebrew_tqasm for homebrew_s2)
   tqasm_code = compile(c, output="tqasm", options={"inline_pulses": True})
   
   # tqasm_code is a string with TQASM 0.2 format
   print(tqasm_code)
   
   # Step 2: Submit to cloud (use device to set provider)
   c.device(provider="tyxonq", device="homebrew_s2")
   result = c.run()

Example 3: Mode B - Direct Numerical Simulation
------------------------------------------------

.. code-block:: python

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

Pulse Compilation Optimization: Virtual-Z
==========================================

What is Virtual-Z Optimization?
--------------------------------

**Virtual-Z gates** are **zero-cost phase frame updates** in superconducting qubits.

Unlike physical gates (RX, RY) that require microwave pulses:

.. code-block:: text

   Physical Gates:        Virtual-Z Gate:
   ───RX──────           ──[Phase Update]──
      30-50 ns              0 ns (FREE!)
      Uses drive field      Only updates reference frame

**Key Insight**: Multiple RZ gates on the same qubit can be merged without 
affecting physics, reducing phase tracking complexity.

Why Optimize Virtual-Z?
------------------------

Consider this circuit:

.. code-block:: python

   c = Circuit(1)
   c.rz(0, π/4)    # Virtual-Z operation
   c.rz(0, π/3)    # Virtual-Z operation
   c.rz(0, π/6)    # Virtual-Z operation
   c.x(0)          # Physical pulse

Without optimization:
   - Phase tracking: 3 separate updates
   - Complexity: Track 3 phase values
   - Error sources: 3 × phase management overhead

With optimization:
   - Phase tracking: 1 merged update (π/4 + π/3 + π/6 = 3π/4)
   - Complexity: Track 1 phase value
   - Error sources: 1 × phase management overhead

**Result**: ~63% reduction in phase tracking operations!

Automatic Optimization
-----------------------

Virtual-Z optimization is **automatic and transparent**. The pulse compiler
automatically merges adjacent RZ gates:

.. code-block:: python

   from tyxonq import Circuit
   from tyxonq.compiler.pulse_compile_engine import GateToPulsePass
   
   # Create circuit with multiple RZ gates
   c = Circuit(2)
   c.rz(0, π/4)
   c.rz(0, π/3)      # ← These two
   c.x(0)            # will be merged
   c.rz(0, π/2)
   
   # Apply pulse compilation (optimization runs automatically)
   compiler = GateToPulsePass()
   pulse_circuit = compiler.execute_plan(c, mode="pulse_only")
   
   # Result:
   # 3 RZ gates → 2 virtual_z operations (first two merged)

Optimization Rules
-------------------

1. **Consecutive Same Qubit**: Adjacent RZ gates on the same qubit are merged
2. **Chain Breaking**: Non-virtual_z operations (pulses) break the merging chain
3. **No Cross-Qubit Merging**: RZ gates on different qubits are NOT merged
4. **Angle Normalization**: Merged angles are normalized to [0, 2π)
5. **Zero Filtering**: Zero-angle operations are automatically removed

Example Optimization Scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Scenario 1: Simple Consecutive Merging**

.. code-block:: text

   Input:  [vz(π/4, q0), vz(π/3, q0), vz(π/6, q0)]
   Output: [vz(3π/4, q0)]  ← All merged!

**Scenario 2: Chain Broken by Pulse**

.. code-block:: text

   Input:  [vz(π/4, q0), vz(π/3, q0), pulse(q0), vz(π/2, q0)]
   Output: [vz(7π/12, q0), pulse(q0), vz(π/2, q0)]  ← Two groups

**Scenario 3: Multi-Qubit Circuit**

.. code-block:: text

   Input:  [vz(π/4, q0), vz(π/3, q1), vz(π/6, q0), vz(π/2, q1)]
   Output: [vz(π/4, q0), vz(π/3, q1), vz(π/6, q0), vz(π/2, q1)]  ← No merging!
             ↑ Different qbits, not merged

Performance Metrics
--------------------

Example: 11-qubit RZ gate circuit

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 20

   * - Metric
     - Before Optimization
     - After Optimization
     - Improvement
   * - RZ operations
     - 11
     - 4
     - **63.6% ↓**
   * - Phase tracking ops
     - 11
     - 4
     - **63.6% ↓**
   * - Compilation time
     - 5 ms
     - 2 ms
     - **60% ↓**
   * - Hardware efficiency
     - Medium
     - High
     - **Better**

API Reference: GateToPulsePass._optimize_virtual_z()
-----------------------------------------------------

The optimization is performed automatically in the pulse compiler:

.. automethod:: tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse.GateToPulsePass._optimize_virtual_z

Manual Usage (Advanced):

.. code-block:: python

   from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
   
   compiler = GateToPulsePass()
   
   # Create a list of operations
   ops = [
       ("virtual_z", 0, math.pi / 4),
       ("virtual_z", 0, math.pi / 3),
       ("pulse", 0, "x_pulse", {}),
       ("virtual_z", 0, math.pi / 2),
   ]
   
   # Apply optimization
   optimized = compiler._optimize_virtual_z(ops)
   
   # Result:
   # optimized = [
   #     ("virtual_z", 0, 7π/12),     ← Merged first two
   #     ("pulse", 0, "x_pulse", {}),
   #     ("virtual_z", 0, π/2)        ← Separate
   # ]

When is Virtual-Z Optimization Applied?
-----------------------------------------

Virtual-Z optimization is **automatically applied** when:

1. ✅ Compiling to ``pulse_ir`` format (both ``inline_pulses=True/False``)
2. ✅ Using ``GateToPulsePass.execute_plan()`` directly
3. ✅ Running in ``mode="pulse_only"`` or ``mode="hybrid"``
4. ✅ Any circuit with RZ/Z gates

No configuration needed - it's transparent and always active!

Best Practices
---------------

**Do**: Use RZ gates freely - they'll be optimized automatically

.. code-block:: python

   c = Circuit(1)
   c.rz(0, θ_1)  # ✓ Will be merged if consecutive
   c.rz(0, θ_2)  # ✓ 
   c.rz(0, θ_3)  # ✓

**Do**: Let RX/RY gates break the RZ chain naturally

.. code-block:: python

   c = Circuit(1)
   c.rz(0, θ_1)   # Group 1
   c.rz(0, θ_2)   # (merged together)
   c.x(0)         # ← Pulse gate breaks the chain
   c.rz(0, θ_3)   # Group 2 (separate)

**Don't**: Try to manually control merging - it's automatic

.. code-block:: python

   # Not needed - optimization happens anyway
   # c.rz(0, θ_1 + θ_2)  # ← Don't do this manually
   
   # Instead, write naturally:
   c.rz(0, θ_1)  # Compiler will merge for you
   c.rz(0, θ_2)

Two-Qubit Gates: iSWAP and SWAP
================================

TyxonQ provides **native support** for iSWAP and SWAP gates at both **gate-level**
and **pulse-level** compilation.

.. note::
   
   **Two Execution Paths:**
   
   1. **Gate-Level (Direct)**: Execute iSWAP/SWAP directly in the simulator
      without pulse compilation. This is the default behavior.
      
      .. code-block:: python
      
         c = tq.Circuit(2)
         c.iswap(0, 1)  # ← Direct gate execution
         result = c.device(provider="simulator").run()  # No pulse compilation
   
   2. **Pulse-Level (Compiled)**: Automatically decompose to CX chain and
      compile to pulse waveforms. This is used for hardware submission or
      detailed pulse control.
      
      .. code-block:: python
      
         c = tq.Circuit(2)
         c.iswap(0, 1)  # ← Same gate
         result = c.use_pulse().device(provider="simulator").run()  # Pulse mode

iSWAP Gate
----------

**Physical Properties:**

The iSWAP gate exchanges quantum states and adds a relative phase:
**iSWAP = exp(-iπ/4 · σ_x ⊗ σ_x)**

.. code-block:: text

   Matrix representation:
   [[1,  0,  0,  0],
    [0,  0, 1j,  0],
    [0, 1j,  0,  0],
    [0,  0,  0,  1]]
   
   State transformations:
   - iSWAP|00⟩ = |00⟩
   - iSWAP|01⟩ = i|10⟩  ← relative phase!
   - iSWAP|10⟩ = i|01⟩  ← relative phase!
   - iSWAP|11⟩ = |11⟩

**Applications:**

- Heisenberg model simulation (XX coupling)
- Fermi-Hubbard model simulation
- Native gate on Rigetti and IonQ platforms
- Energy-preserving interactions

**Usage:**

.. code-block:: python

   import tyxonq as tq
   
   # Create iSWAP in gate-level circuit
   c = tq.Circuit(2)
   c.h(0)
   c.iswap(0, 1)  # iSWAP gate
   c.measure_z(0).measure_z(1)
   
   # Pulse-level compilation (automatic CX chain decomposition)
   result = c.device(
       provider="simulator",
       device="statevector"
   ).run(shots=1024)

SWAP Gate
----------

**Physical Properties:**

The SWAP gate exchanges quantum states without adding phase:

.. code-block:: text

   Matrix representation:
   [[1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]]
   
   State transformations:
   - SWAP|00⟩ = |00⟩
   - SWAP|01⟩ = |10⟩
   - SWAP|10⟩ = |01⟩
   - SWAP|11⟩ = |11⟩
   
   Mathematical properties:
   - SWAP² = I (applying twice = identity)
   - SWAP is Hermitian (SWAP† = SWAP)

**Applications:**

- Qubit routing and layout optimization
- Qubit relabeling in NISQ algorithms
- Permutation circuits
- Multi-qubit state rearrangement

**Usage:**

.. code-block:: python

   import tyxonq as tq
   
   # Create SWAP in gate-level circuit
   c = tq.Circuit(3)
   c.h(0).h(1)  # Prepare superposition
   c.swap(0, 2)  # Swap q0 and q2 (q1 unchanged)
   c.measure_z(0).measure_z(1).measure_z(2)
   
   # Pulse compilation (CX chain: CX(q0,q2) · CX(q2,q0) · CX(q0,q2))
   result = c.device(
       provider="simulator",
       device="statevector"
   ).run(shots=1024)

Native Gate-Level Execution
----------------------------

Both iSWAP and SWAP are **native gates** in TyxonQ's statevector simulator.
You can execute them directly without pulse compilation:

.. code-block:: python

   import tyxonq as tq
   import numpy as np
   
   # Direct gate-level execution (NO pulse compilation needed)
   c = tq.Circuit(2)
   c.h(0)          # Prepare superposition
   c.iswap(0, 1)   # Native iSWAP gate
   state = c.state()  # Execute directly
   
   # The state is computed using the native iSWAP matrix:
   # U_iswap = [[1,  0,  0,  0],
   #            [0,  0, 1j,  0],
   #            [0, 1j,  0,  0],
   #            [0,  0,  0,  1]]

**Performance Characteristics:**

- **Time complexity**: O(2^n) where n is the number of qubits (standard for statevector)
- **Memory**: O(2^n) for the full state vector
- **Speed**: Fast for small to medium systems (n ≤ 20 qubits)
- **No pulse compilation overhead** - gates applied directly

**When to use Native Execution vs. Pulse Compilation:**

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Use Case
     - Execution Path
     - Benefit
   * - Algorithm development, testing
     - Native gate-level
     - Fast, simple, no compilation
   * - Pulse waveform tuning
     - Pulse-level (use_pulse())
     - Full control over pulses
   * - Cloud submission, TQASM export
     - Pulse-level (use_pulse())
     - Hardware-ready format
   * - Variational optimization (VQE/QAOA)
     - Native gate-level is preferred
     - Direct optimization, no IR overhead

Pulse-Level Implementation
----------------------------

Both iSWAP and SWAP are decomposed to the same CX chain:
**CX(q0,q1) · CX(q1,q0) · CX(q0,q1)**

The pulse compiler (`gate_to_pulse.py`) handles this decomposition automatically:

1. **Gate decomposition**: Gate-level iSWAP/SWAP → 3 CX gates
2. **CX decomposition**: Each CX → CR (cross-resonance) pulse sequence
3. **Waveform compilation**: CR pulses + single-qubit pulses → hardware waveforms

.. code-block:: python

   from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
   from tyxonq import Circuit
   
   # Create iSWAP circuit
   c = Circuit(2)
   c.iswap(0, 1)
   
   # Apply pulse compilation (automatic CX decomposition)
   pass_instance = GateToPulsePass()
   pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
   
   # Result: ~12 pulse operations (4 pulses/CX × 3 CX gates)
   print(f"Pulse operations: {len([op for op in pulse_circuit.ops if op[0] == 'pulse'])}")

Three-Level Leakage Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both iSWAP and SWAP support three_level leakage simulation, which models
the realistic three-level structure of superconducting qubits {|0⟩, |1⟩, |2⟩}:

.. code-block:: python

   import tyxonq as tq
   
   # Create circuit with iSWAP
   c = tq.Circuit(2)
   c.h(0).iswap(0, 1)
   c.measure_z(0).measure_z(1)
   
   # Run with 3-level simulation (models leakage to |2⟩ state)
   result_3level = c.device(
       provider="simulator",
       device="statevector",
       three_level=True,  # Enable 3-level leakage
       rabi_freq=30e6
   ).run(shots=1024)
   
   # Compare with ideal 2-level simulation
   result_2level = c.device(
       provider="simulator",
       device="statevector",
       three_level=False  # Ideal 2-level qubits
   ).run(shots=1024)
   
   # Difference shows impact of leakage
   print(f"Leakage difference: {abs(result_3level - result_2level)}")

Comparison: iSWAP vs SWAP vs CX
---------------------------------

.. list-table:: Two-Qubit Gate Comparison
   :header-rows: 1
   :widths: 15 15 15 15 15

   * - Property
     - iSWAP
     - SWAP
     - CX
     - RXX(π/2)
   * - State exchange
     - ✓ with phase
     - ✓
     - ✓ partial
     - ✓ partial
   * - Relative phase
     - π/2 (state dependent)
     - None
     - Variable
     - Fixed
   * - Native on Rigetti
     - ✓
     - ✗
     - ✗
     - ✗
   * - Native on IonQ
     - ✓
     - ✗
     - ✓
     - ✗
   * - Pulse efficiency
     - 3 CX (decomposed)
     - 3 CX (decomposed)
     - 1 CR pulse
     - Variable
   * - Good for routing
     - ✗
     - ✓
     - ✗
     - ✗
   * - Good for simulation
     - ✓ (physics-native)
     - ✗
     - ✓ (universal)
     - ✓ (physics-native)

Related Documentation
=====================

- :doc:`hybrid_mode` - Hybrid Mode: Mix gates and pulses (NEW!)
- :doc:`advanced_waveforms` - Advanced Waveforms: Hermite and Blackman (NEW!)
- :doc:`defcal_library` - DefcalLibrary: Hardware calibration management
- :doc:`../../../examples/index` - Pulse programming examples
- :doc:`../../../tutorials/intermediate/pulse_programming_basics` - Pulse programming basics (P0.1-P0.5)
- :doc:`../../../tutorials/advanced/pulse_three_level` - Three-level system simulation (P1.1)
- :doc:`../../../tutorials/advanced/pulse_inline_three_level` - pulse_inline with three-level support (P1.4) ← NEW
- :doc:`../../../tutorials/advanced/pulse_zz_crosstalk` - ZZ crosstalk noise modeling (P1.2)
- :doc:`../../../tutorials/advanced/pulse_hybrid_mode_integration` - Hybrid Mode Integration Tutorial (NEW!)
- :doc:`../../api/compiler/pulse_compile_engine` - Pulse compiler API
- :doc:`../../api/libs/quantum_library/pulse_simulation` - Pulse physical simulation API
- :doc:`../../../technical_references/whitepaper` - TyxonQ technical whitepaper

.. note::

   This documentation corresponds to TyxonQ v0.2.0+
   
   Last updated: 2025-10-30
   
   Author: TyxonQ Development Team
