"""
======================================================================
Quantum Framework Performance Comparison
量子计算框架性能对比：TyxonQ vs Qiskit vs PennyLane
======================================================================

This example provides a comprehensive performance and API comparison between
TyxonQ and other popular quantum computing frameworks (Qiskit, PennyLane).

本示例全面对比TyxonQ与其他主流量子计算框架（Qiskit、PennyLane）的性能和API。

TyxonQ 3 Execution Modes (TyxonQ 3种执行模式):
---------------------------------------------------------------------
1. **Shot-based Sampling** (模拟真机): 
   使用完整的TyxonQ流程 Hamiltonian → Circuit → Compile → Device(shots>0) → Postprocessing
   最接近真实量子硬件的shot-count采样方式

2. **Exact Statevector** (精确解析解):
   使用circuit.state()获取精确状态向量，无采样噪声

3. **Exact + JIT** (JIT加速):
   在精确解析解基础上使用JIT编译加速

Comparison Dimensions (对比维度):
--------------------------------
1. **API Style**: Code elegance and ease of use
2. **Execution Performance**: Circuit simulation speed
3. **Optimization Features**: JIT, autograd
4. **Scalability**: Large circuit performance

Test Scenario (测试场景):
-----------------------
Variational Quantum Eigensolver (VQE) for TFIM:
- System: 6 qubits
- Ansatz: Hardware-Efficient Ansatz (4 layers)
- Observable: Transverse-Field Ising Model Hamiltonian
- Optimizer: Gradient descent (10 steps)

Author: TyxonQ Team
Date: 2025
"""

import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Quantum Framework Performance Comparison")
print("TyxonQ vs Qiskit vs PennyLane")
print("="*70)

# ==============================================================================
# Configuration
# ==============================================================================

N_QUBITS = 6
N_LAYERS = 4
N_STEPS = 10
LEARNING_RATE = 0.1
N_SHOTS = 8192  # For shot-based sampling

print(f"\nTest Configuration:")
print(f"  Qubits: {N_QUBITS}")
print(f"  Ansatz layers: {N_LAYERS}")
print(f"  Optimization steps: {N_STEPS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Shots (Mode 1): {N_SHOTS}")


# ==============================================================================
# TyxonQ Implementation (3 Modes)
# ==============================================================================

def run_tyxonq_comparison():
    """Run TyxonQ with 3 different execution modes."""
    import tyxonq as tq
    import torch
    from tyxonq.libs.quantum_library.kernels.pauli import pauli_string_sum_dense
    
    print("\n" + "="*70)
    print("[1/4] TyxonQ Implementation")
    print("="*70)
    
    tq.set_backend("pytorch")
    
    # TFIM Hamiltonian: H = -Σ Z_i Z_{i+1} - h Σ X_i
    h_field = 1.0
    
    # Build Hamiltonian matrix and Pauli terms
    pauli_terms = []
    weights = []
    # ZZ coupling
    for i in range(N_QUBITS - 1):
        term = [0] * N_QUBITS
        term[i] = 3  # Z
        term[i+1] = 3  # Z
        pauli_terms.append(term)
        weights.append(-1.0)
    # Transverse field
    for i in range(N_QUBITS):
        term = [0] * N_QUBITS
        term[i] = 1  # X
        pauli_terms.append(term)
        weights.append(-h_field)
    
    H_matrix = pauli_string_sum_dense(pauli_terms, weights)
    
    def create_circuit_hea(params):
        """Hardware-Efficient Ansatz."""
        c = tq.Circuit(N_QUBITS)
        # Initial layer
        for i in range(N_QUBITS):
            c.h(i)
        # Variational layers
        for layer in range(N_LAYERS):
            for i in range(N_QUBITS):
                c.ry(i, theta=params[layer, i, 0])
                c.rz(i, theta=params[layer, i, 1])
            for i in range(N_QUBITS - 1):
                c.cx(i, i + 1)
        return c
    
    # ================================================================
    # 初始化共享参数（保证所有模式使用相同的起点）
    # ================================================================
    np.random.seed(42)  # 固定随机种子
    initial_params = np.random.randn(N_LAYERS, N_QUBITS, 2) * 0.1
    
    # ================================================================
    # Mode 1: Shot-based Sampling (模拟真机)
    # ================================================================
    # 使用TyxonQ完整流程：Circuit → Compile → Device(shots>0) → Postprocessing
    # 这是最接近真实量子硬件的方式，包含采样噪声
    # ================================================================
    print("\n[Mode 1] Shot-based Sampling (模拟真机)")
    print("使用完整TyxonQ流程: Hamiltonian → Circuit → Compile → Device → Postprocessing")
    print("-" * 70)
    
    def compute_pauli_expectation_from_counts(counts, pauli_string, n_qubits):
        """从测量counts计算Pauli算符期望值."""
        expectation = 0.0
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            # 计算每个bitstring的parity
            parity = 1.0
            for qubit_idx, pauli_op in enumerate(pauli_string):
                if pauli_op != 0:  # Not identity
                    bit = int(bitstring[qubit_idx])
                    parity *= (-1) ** bit
            expectation += parity * count / total_shots
        
        return expectation
    
    def tfim_energy_shots(params):
        """使用shot-based sampling计算TFIM能量."""
        energy = 0.0
        
        # 对每个Pauli term单独测量
        for term, weight in zip(pauli_terms, weights):
            # 创建电路
            circuit = create_circuit_hea(params)
            
            # 添加basis rotation和测量
            for qubit, pauli in enumerate(term):
                if pauli == 1:  # X basis
                    circuit.h(qubit)
                elif pauli == 2:  # Y basis
                    circuit.sdg(qubit).h(qubit)
                # pauli == 3 (Z) or 0 (I): no rotation needed
            
            # 添加测量
            for i in range(N_QUBITS):
                circuit.measure_z(i)
            
            # 执行完整TyxonQ流程
            result = circuit.compile().device(shots=N_SHOTS).postprocessing().run()
            # 提取counts（result可能是dict或list）
            if isinstance(result, list):
                counts = result[0].get("result", {})
            else:
                counts = result.get("result", {})
            
            # 从counts计算期望值
            expectation = compute_pauli_expectation_from_counts(counts, term, N_QUBITS)
            energy += weight * expectation
        
        return energy
    
    # Parameter Shift Rule for shot-based gradient
    # 这是真实量子硬件上计算梯度的标准方法
    def parameter_shift_gradient(params, shift=np.pi/2):
        """使用Parameter Shift Rule计算梯度.
        
        对于参数化旋转门R(θ), 梯度为:
        ∂⟨H⟩/∂θ = [⟨H⟩(θ+π/2) - ⟨H⟩(θ-π/2)] / 2
        """
        grads = np.zeros_like(params)
        for layer in range(N_LAYERS):
            for qubit in range(N_QUBITS):
                for param_idx in range(2):
                    # θ + π/2
                    params_plus = params.copy()
                    params_plus[layer, qubit, param_idx] += shift
                    energy_plus = tfim_energy_shots(params_plus)
                    
                    # θ - π/2
                    params_minus = params.copy()
                    params_minus[layer, qubit, param_idx] -= shift
                    energy_minus = tfim_energy_shots(params_minus)
                    
                    # Parameter shift gradient
                    grads[layer, qubit, param_idx] = (energy_plus - energy_minus) / 2.0
        
        return grads
    
    params_shots = initial_params.copy()  # 使用共享的初始参数
    lr = LEARNING_RATE
    
    times_shots = []
    energies_shots = []
    
    print("使用Parameter Shift Rule计算梯度（真实硬件标准方法）...")
    t_start = time.time()
    for step in range(N_STEPS):
        t0 = time.time()
        
        # 当前能量
        energy_0 = tfim_energy_shots(params_shots)
        
        # Parameter Shift Rule梯度
        grads = parameter_shift_gradient(params_shots)
        
        # 梯度下降
        params_shots -= lr * grads
        
        t1 = time.time()
        times_shots.append(t1 - t0)
        energies_shots.append(energy_0)
        
        if step % 3 == 0:
            print(f"  Step {step:2d}: Energy = {energy_0:.6f}, Time = {t1-t0:.4f}s")
    
    t_total_shots = time.time() - t_start
    avg_time_shots = np.mean(times_shots[1:])  # Skip first
    
    print(f"\n  Results:")
    print(f"    Total time: {t_total_shots:.3f}s")
    print(f"    Avg step time: {avg_time_shots:.4f}s")
    print(f"    Final energy: {energies_shots[-1]:.6f}")
    print(f"    Note: Includes shot-noise (采样噪声)")
    
    # ================================================================
    # Mode 2: NumPy Backend + Numeric value_and_grad
    # ================================================================
    # 使用NumPy后端 + TyxonQ numerics中的value_and_grad（有限差分梯度）
    # 展示框架自带的梯度计算能力，不依赖PyTorch autograd
    # ================================================================
    print("\n[Mode 2] NumPy Backend + Numeric value_and_grad")
    print("使用NumPy后端 + TyxonQ numerics的value_and_grad（有限差分梯度）")
    print("-" * 70)
    
    # 切换到NumPy后端
    tq.set_backend("numpy")
    from tyxonq.numerics import NumericBackend as nb
    
    # 将Hamiltonian转为numpy
    if isinstance(H_matrix, torch.Tensor):
        H_numpy = H_matrix.detach().cpu().numpy()
    else:
        H_numpy = H_matrix
    
    def tfim_energy_numpy(params_flat):
        """使用NumPy后端计算能量（接收扁平化参数）."""
        # Reshape参数
        params = params_flat.reshape(N_LAYERS, N_QUBITS, 2)
        
        # 创建电路
        circuit = create_circuit_hea(params)
        
        # 获取状态向量（NumPy数组）
        psi = circuit.state()
        
        # 计算期望值 <ψ|H|ψ>
        Hpsi = H_numpy @ psi
        energy = np.real(np.dot(np.conj(psi), Hpsi))
        
        return energy
    
    # 使用NumericBackend的value_and_grad计算梯度
    # 这是TyxonQ框架内置的梯度计算能力（基于有限差分）
    energy_and_grad_fn = nb.value_and_grad(tfim_energy_numpy)
    
    params_numpy = initial_params.copy()  # 使用共享的初始参数
    lr = LEARNING_RATE
    
    times_numpy = []
    energies_numpy = []
    
    print("使用NumericBackend.value_and_grad计算梯度（有限差分方法）...")
    t_start = time.time()
    for step in range(N_STEPS):
        t0 = time.time()
        
        # 扁平化参数
        params_flat = params_numpy.flatten()
        
        # 使用value_and_grad同时计算能量和梯度
        energy, grads_flat = energy_and_grad_fn(params_flat)
        
        # Reshape梯度
        grads = grads_flat.reshape(N_LAYERS, N_QUBITS, 2)
        
        # 梯度下降
        params_numpy -= lr * grads
        
        t1 = time.time()
        times_numpy.append(t1 - t0)
        energies_numpy.append(energy)
        
        if step % 3 == 0:
            print(f"  Step {step:2d}: Energy = {energy:.6f}, Time = {t1-t0:.4f}s")
    
    t_total_numpy = time.time() - t_start
    avg_time_numpy = np.mean(times_numpy[1:])  # Skip first
    
    print(f"\n  Results:")
    print(f"    Total time: {t_total_numpy:.3f}s")
    print(f"    Avg step time: {avg_time_numpy:.4f}s")
    print(f"    Final energy: {energies_numpy[-1]:.6f}")
    print(f"    Speedup vs Mode 1: {avg_time_shots/avg_time_numpy:.2f}x")
    print(f"    关键特性:")
    print(f"      ✓ NumPy后端（CPU计算）")
    print(f"      ✓ NumericBackend.value_and_grad（有限差分）")
    print(f"      ✓ 框架内置梯度能力")
    print(f"      ✓ 精确解析解（无采样噪声）")
    
    # ================================================================
    # Mode 3: PyTorch Backend + Autograd (最优性能)
    # ================================================================
    # 使用PyTorch后端 + PyTorch autograd自动微分
    # TyxonQ推荐的最佳实践：充分利用现代深度学习框架的优化
    # ================================================================
    print("\n[Mode 3] PyTorch Backend + Autograd (最优性能)")
    print("TyxonQ推荐的最佳实践：PyTorch后端 + 自动微分")
    print("-" * 70)
    
    # 切换到PyTorch后端
    tq.set_backend("pytorch")
    
    # 预先转换Hamiltonian为Tensor（避免每次转换）
    if isinstance(H_matrix, torch.Tensor):
        H_tensor = H_matrix.to(torch.complex128)
    else:
        H_tensor = torch.from_numpy(H_matrix).to(torch.complex128)
    
    def tfim_energy_pytorch(circuit, H):
        """使用PyTorch后端计算能量."""
        psi = circuit.state()
        Hpsi = H @ psi
        return torch.real(torch.dot(torch.conj(psi), Hpsi))
    
    params_pytorch = torch.nn.Parameter(torch.from_numpy(initial_params).float())  # 使用相同的初始参数
    optimizer_pytorch = torch.optim.Adam([params_pytorch], lr=LEARNING_RATE)
    
    times_pytorch = []
    energies_pytorch = []
    
    print("使用PyTorch autograd计算梯度（完整VQE优化循环）...")
    t_start = time.time()
    for step in range(N_STEPS):
        t0 = time.time()
        optimizer_pytorch.zero_grad()
        
        circuit = create_circuit_hea(params_pytorch)
        energy = tfim_energy_pytorch(circuit, H_tensor)
        energy.backward()  # PyTorch autograd
        
        optimizer_pytorch.step()
        t1 = time.time()
        
        times_pytorch.append(t1 - t0)
        energies_pytorch.append(energy.item())
        
        if step % 3 == 0:
            print(f"  Step {step:2d}: Energy = {energy.item():.6f}, Time = {t1-t0:.4f}s")
    
    t_total_pytorch = time.time() - t_start
    avg_time_pytorch = np.mean(times_pytorch[1:])
    
    print(f"\n  Results:")
    print(f"    Total time: {t_total_pytorch:.3f}s")
    print(f"    Avg step time: {avg_time_pytorch:.4f}s")
    print(f"    Final energy: {energies_pytorch[-1]:.6f}")
    print(f"    Speedup vs Mode 2: {avg_time_numpy/avg_time_pytorch:.2f}x")
    print(f"    Speedup vs Mode 1: {avg_time_shots/avg_time_pytorch:.2f}x")
    print(f"\n    关键优化点:")
    print(f"      ✓ PyTorch后端（高效tensor运算）")
    print(f"      ✓ PyTorch autograd（自动微分）")
    print(f"      ✓ Adam优化器（自适应学习率）")
    print(f"      ✓ 预构建Hamiltonian")
    
    return {
        'shot_based': {'time': avg_time_shots, 'energy': energies_shots[-1]},
        'numpy_grad': {'time': avg_time_numpy, 'energy': energies_numpy[-1]},
        'pytorch_autograd': {'time': avg_time_pytorch, 'energy': energies_pytorch[-1]}
    }


# ==============================================================================
# Qiskit Implementation
# ==============================================================================

def run_qiskit():
    """Run Qiskit VQE."""
    print("\n" + "="*70)
    print("[2/4] Qiskit Implementation")
    print("="*70)
    
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp, Statevector
        
        print("\n[Qiskit] Imperative API Style")
        print("-" * 70)
        
        # Define TFIM Hamiltonian
        pauli_list = []
        coeffs = []
        for i in range(N_QUBITS - 1):
            pauli_str = 'I' * i + 'ZZ' + 'I' * (N_QUBITS - i - 2)
            pauli_list.append(pauli_str)
            coeffs.append(-1.0)
        for i in range(N_QUBITS):
            pauli_str = 'I' * i + 'X' + 'I' * (N_QUBITS - i - 1)
            pauli_list.append(pauli_str)
            coeffs.append(-1.0)
        
        hamiltonian = SparsePauliOp(pauli_list, np.array(coeffs))
        
        def create_circuit_qiskit(params):
            qc = QuantumCircuit(N_QUBITS)
            for i in range(N_QUBITS):
                qc.h(i)
            for layer in range(N_LAYERS):
                for i in range(N_QUBITS):
                    qc.ry(float(params[layer, i, 0]), i)
                    qc.rz(float(params[layer, i, 1]), i)
                for i in range(N_QUBITS - 1):
                    qc.cx(i, i + 1)
            return qc
        
        def compute_energy_qiskit(params):
            qc = create_circuit_qiskit(params)
            state = Statevector(qc)
            return state.expectation_value(hamiltonian).real
        
        params_qiskit = np.random.randn(N_LAYERS, N_QUBITS, 2) * 0.1
        lr = LEARNING_RATE
        times_qiskit = []
        energies_qiskit = []
        
        t_start = time.time()
        for step in range(N_STEPS):
            t0 = time.time()
            eps = 1e-3
            energy_0 = compute_energy_qiskit(params_qiskit)
            grads = np.zeros_like(params_qiskit)
            
            for layer in range(N_LAYERS):
                for qubit in range(N_QUBITS):
                    for param_idx in range(2):
                        params_plus = params_qiskit.copy()
                        params_plus[layer, qubit, param_idx] += eps
                        energy_plus = compute_energy_qiskit(params_plus)
                        grads[layer, qubit, param_idx] = (energy_plus - energy_0) / eps
            
            params_qiskit -= lr * grads
            t1 = time.time()
            times_qiskit.append(t1 - t0)
            energies_qiskit.append(energy_0)
            
            if step % 3 == 0:
                print(f"  Step {step:2d}: Energy = {energy_0:.6f}, Time = {t1-t0:.4f}s")
        
        t_total_qiskit = time.time() - t_start
        avg_time_qiskit = np.mean(times_qiskit[1:])
        
        print(f"\n  Results:")
        print(f"    Total time: {t_total_qiskit:.3f}s")
        print(f"    Avg step time: {avg_time_qiskit:.4f}s")
        print(f"    Final energy: {energies_qiskit[-1]:.6f}")
        
        return {'time': avg_time_qiskit, 'energy': energies_qiskit[-1]}
        
    except ImportError:
        print("\n  ⚠️  Qiskit not installed. Skipping.")
        return None


# ==============================================================================
# PennyLane Implementation
# ==============================================================================

def run_pennylane():
    """Run PennyLane VQE."""
    print("\n" + "="*70)
    print("[3/4] PennyLane Implementation")
    print("="*70)
    
    try:
        import pennylane as qml
        import torch
        
        print("\n[PennyLane] Functional API Style")
        print("-" * 70)
        
        dev = qml.device('default.qubit', wires=N_QUBITS)
        coeffs = []
        obs = []
        for i in range(N_QUBITS - 1):
            coeffs.append(-1.0)
            obs.append(qml.PauliZ(i) @ qml.PauliZ(i+1))
        for i in range(N_QUBITS):
            coeffs.append(-1.0)
            obs.append(qml.PauliX(i))
        
        hamiltonian = qml.Hamiltonian(coeffs, obs)
        
        @qml.qnode(dev, interface='torch')
        def circuit_pennylane(params):
            for i in range(N_QUBITS):
                qml.Hadamard(wires=i)
            for layer in range(N_LAYERS):
                for i in range(N_QUBITS):
                    qml.RY(params[layer, i, 0], wires=i)
                    qml.RZ(params[layer, i, 1], wires=i)
                for i in range(N_QUBITS - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(hamiltonian)
        
        params_pl = torch.nn.Parameter(torch.randn(N_LAYERS, N_QUBITS, 2) * 0.1)
        optimizer_pl = torch.optim.Adam([params_pl], lr=LEARNING_RATE)
        times_pl = []
        energies_pl = []
        
        t_start = time.time()
        for step in range(N_STEPS):
            t0 = time.time()
            optimizer_pl.zero_grad()
            energy = circuit_pennylane(params_pl)
            energy.backward()
            optimizer_pl.step()
            t1 = time.time()
            
            times_pl.append(t1 - t0)
            energies_pl.append(energy.item())
            
            if step % 3 == 0:
                print(f"  Step {step:2d}: Energy = {energy.item():.6f}, Time = {t1-t0:.4f}s")
        
        t_total_pl = time.time() - t_start
        avg_time_pl = np.mean(times_pl[1:])
        
        print(f"\n  Results:")
        print(f"    Total time: {t_total_pl:.3f}s")
        print(f"    Avg step time: {avg_time_pl:.4f}s")
        print(f"    Final energy: {energies_pl[-1]:.6f}")
        
        return {'time': avg_time_pl, 'energy': energies_pl[-1]}
        
    except ImportError:
        print("\n  ⚠️  PennyLane not installed. Skipping.")
        return None


# ==============================================================================
# Performance Summary
# ==============================================================================

def print_summary(results_tyxonq, results_qiskit, results_pennylane):
    """Print comprehensive performance summary."""
    print("\n" + "="*70)
    print("[4/4] Performance Summary")
    print("="*70)
    
    print("\n" + "="*70)
    print("TyxonQ Performance Modes (TyxonQ性能模式)")
    print("="*70)
    
    print(f"\n  {'Mode':<35} {'Avg Time':<15} {'Speedup':<12} {'Final Energy'}")
    print("  " + "-"*75)
    
    baseline = results_tyxonq['shot_based']['time']
    for mode_name, mode_data in results_tyxonq.items():
        speedup = baseline / mode_data['time']
        mode_display = {
            'shot_based': 'Shot-based (模拟真机)',
            'numpy_grad': 'NumPy + value_and_grad (框架梯度)',
            'pytorch_autograd': 'PyTorch + Autograd (最优)'
        }[mode_name]
        print(f"  {mode_display:<35} {mode_data['time']:.4f}s{'':<8} {speedup:.2f}x{'':<7} {mode_data['energy']:.6f}")
    
    print("\n" + "="*70)
    print("Framework Comparison (框架对比)")
    print("="*70)
    
    print(f"\n  {'Framework':<30} {'API Style':<20} {'Avg Time':<15} {'Relative'}")
    print("  " + "-"*75)
    
    all_results = [
        ('TyxonQ (Shot-based)', 'Compiled', results_tyxonq['shot_based']['time']),
        ('TyxonQ (NumPy+Grad)', 'NumericBackend', results_tyxonq['numpy_grad']['time']),
        ('TyxonQ (PyTorch+Autograd)', 'PyTorch Backend', results_tyxonq['pytorch_autograd']['time'])
    ]
    if results_qiskit:
        all_results.append(('Qiskit', 'Imperative', results_qiskit['time']))
    if results_pennylane:
        all_results.append(('PennyLane', 'Functional', results_pennylane['time']))
    
    fastest_time = min(r[2] for r in all_results)
    
    for framework, api_style, avg_time in all_results:
        relative = avg_time / fastest_time
        print(f"  {framework:<30} {api_style:<20} {avg_time:.4f}s{'':<8} {relative:.2f}x")
    
    print("\n" + "="*70)
    print("Conclusions (结论)")
    print("="*70)
    
    print("\n  TyxonQ 3种模式特点:")
    print("    1️⃣  Shot-based: 最接近真实量子硬件")
    print("        • 使用shots采样，包含统计噪声")
    print("        • Parameter Shift Rule计算梯度（真机标准方法）")
    print("        • 完整编译流程：Circuit → Compile → Device → Postprocessing")
    print("    2️⃣  NumPy + value_and_grad: 框架内置梯度能力")
    print("        • NumPy后端（CPU计算）")
    print("        • NumericBackend.value_and_grad（有限差分梯度）")
    print("        • 不依赖PyTorch，展示框架自带梯度计算")
    print("        • 精确解析解（无采样噪声）")
    print("    3️⃣  PyTorch + Autograd: TyxonQ最佳实践")
    print("        • PyTorch后端（高效tensor运算）")
    print("        • PyTorch autograd（自动微分，最快）")
    print("        • Adam优化器（自适应学习率）")
    
    print("\n  性能对比:")
    fastest_framework = min(all_results, key=lambda x: x[2])
    print(f"    🏆 最快: {fastest_framework[0]} - {fastest_framework[2]:.4f}s per step")
    print(f"    📊 PyTorch加速比: {results_tyxonq['numpy_grad']['time']/results_tyxonq['pytorch_autograd']['time']:.2f}x (vs NumPy)")
    
    if 'TyxonQ' in fastest_framework[0]:
        print("    🎉 TyxonQ achieves best performance!")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("\nRunning comprehensive framework comparison...")
    print("This may take 1-2 minutes.\n")
    
    results_tyxonq = run_tyxonq_comparison()
    results_qiskit = run_qiskit()
    results_pennylane = run_pennylane()
    
    print_summary(results_tyxonq, results_qiskit, results_pennylane)
    
    print("\n" + "="*70)
    print("Framework Performance Comparison Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
