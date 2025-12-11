"""
Numeric Backend Switching Demonstration

This example showcases TyxonQ's flexible numeric backend system, allowing
seamless switching between NumPy, PyTorch, and CuPyNumeric for quantum
circuit simulations. The unified backend interface ensures code portability
across different computational frameworks.

Key Features:
- Unified ArrayBackend protocol
- Seamless backend switching via tq.set_backend()
- Automatic dtype management
- JIT compilation support (framework-specific)
- GPU acceleration (PyTorch/CuPy)

Backends Compared:
1. NumPy: CPU-only, reference implementation
2. PyTorch: CPU/GPU, automatic differentiation, JIT via torch.compile
3. CuPyNumeric (optional): GPU acceleration for NumPy-like API

Performance Characteristics:
- NumPy: Best for small circuits (<12 qubits), simple scripts
- PyTorch: Best for VQAs with gradients, GPU available, JIT benefits
- CuPy: Best for large statevector operations on GPU

Migrated from: examples-ng/aces_for_setting_numeric_backend.py
"""

import time
import numpy as np

import tyxonq as tq


# ==================== Simple VQE Circuit ====================

def build_vqe_circuit(n_qubits, n_layers, params):
    """Build a simple VQE ansatz circuit
    
    Args:
        n_qubits: Number of qubits
        n_layers: Circuit depth
        params: Parameters for rotations
    
    Returns:
        Circuit instance
    """
    K = tq.get_backend()
    params = K.reshape(params, [n_layers, n_qubits])
    
    c = tq.Circuit(n_qubits)
    
    for i in range(n_qubits):
        c.h(i)
    
    for layer in range(n_layers):
        for i in range(n_qubits - 1):
            c.cnot(i, i + 1)
        for i in range(n_qubits):
            c.rz(i, theta=params[layer, i])
    
    return c


def compute_energy(n_qubits, n_layers, params):
    """Compute simple energy expectation
    
    Returns:
        Sum of Z expectations on all qubits
    """
    K = tq.get_backend()
    
    c = build_vqe_circuit(n_qubits, n_layers, params)
    
    # Compute <Z> for each qubit
    expectations = K.stack([
        K.real(c.expectation_ps(z=[i])) 
        for i in range(n_qubits)
    ])
    
    return K.sum(expectations)


# ==================== Backend Demonstrations ====================

def demo_numpy_backend():
    """Demonstrate NumPy backend"""
    print("\n" + "=" * 70)
    print("NumPy Backend Demo")
    print("=" * 70)
    
    tq.set_backend("numpy")
    K = tq.get_backend()
    
    print(f"Backend type: {type(K)}")
    print(f"Default dtype: {K.dtypestr}")
    
    # Create simple circuit
    n, layers = 4, 2
    params = K.ones([layers, n])
    
    t0 = time.time()
    energy = compute_energy(n, layers, params)
    t1 = time.time()
    
    print(f"Energy: {energy:.6f}")
    print(f"Computation time: {(t1-t0)*1000:.2f} ms")
    print(f"Result type: {type(energy)}")
    
    return energy


def demo_pytorch_backend():
    """Demonstrate PyTorch backend with GPU support"""
    print("\n" + "=" * 70)
    print("PyTorch Backend Demo")
    print("=" * 70)
    
    try:
        import torch
        
        tq.set_backend("pytorch")
        K = tq.get_backend()
        
        print(f"Backend type: {type(K)}")
        print(f"Default dtype: {K.dtypestr}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Create circuit with torch tensors
        n, layers = 4, 2
        params = K.ones([layers, n])
        
        t0 = time.time()
        energy = compute_energy(n, layers, params)
        t1 = time.time()
        
        print(f"Energy: {energy:.6f}")
        print(f"Computation time: {(t1-t0)*1000:.2f} ms")
        print(f"Result type: {type(energy)}")
        
        # Demonstrate gradient computation
        params_grad = torch.tensor([[1.0, 2.0, 0.5, 1.5], [0.8, 1.2, 0.3, 1.7]], requires_grad=True)
        energy_grad = compute_energy(n, layers, params_grad)
        energy_grad.backward()
        
        print(f"Gradient norm: {params_grad.grad.norm().item():.6f}")
        
        return energy
        
    except ImportError:
        print("PyTorch not installed, skipping demo")
        return None


def demo_cupynumeric_backend():
    """Demonstrate CuPyNumeric backend (optional)"""
    print("\n" + "=" * 70)
    print("CuPyNumeric Backend Demo (GPU)")
    print("=" * 70)
    
    try:
        tq.set_backend("cupynumeric")
        K = tq.get_backend()
        
        print(f"Backend type: {type(K)}")
        print(f"Default dtype: {K.dtypestr}")
        
        # Create circuit
        n, layers = 4, 2
        params = K.ones([layers, n])
        
        t0 = time.time()
        energy = compute_energy(n, layers, params)
        t1 = time.time()
        
        print(f"Energy: {energy:.6f}")
        print(f"Computation time: {(t1-t0)*1000:.2f} ms")
        
        return energy
        
    except Exception as e:
        print(f"CuPyNumeric not available: {e}")
        print("Falling back to NumPy (this is expected if CuPy not installed)")
        return None


def compare_backends_performance():
    """Compare performance across backends"""
    print("\n" + "=" * 70)
    print("Backend Performance Comparison")
    print("=" * 70)
    
    n, layers = 6, 3
    trials = 10
    
    results = {}
    
    # NumPy benchmark
    tq.set_backend("numpy")
    K = tq.get_backend()
    params = K.ones([layers, n])
    
    times_np = []
    for _ in range(trials):
        t0 = time.time()
        _ = compute_energy(n, layers, params)
        times_np.append(time.time() - t0)
    
    results['numpy'] = {
        'mean_ms': np.mean(times_np[1:]) * 1000,  # Exclude first (warmup)
        'std_ms': np.std(times_np[1:]) * 1000
    }
    
    # PyTorch benchmark  
    try:
        import torch
        tq.set_backend("pytorch")
        K = tq.get_backend()
        params = K.ones([layers, n])
        
        times_pt = []
        for _ in range(trials):
            t0 = time.time()
            _ = compute_energy(n, layers, params)
            times_pt.append(time.time() - t0)
        
        results['pytorch'] = {
            'mean_ms': np.mean(times_pt[1:]) * 1000,
            'std_ms': np.std(times_pt[1:]) * 1000
        }
    except ImportError:
        results['pytorch'] = None
    
    # Print comparison table
    print(f"\nProblem size: {n} qubits, {layers} layers")
    print(f"Trials: {trials} (first trial excluded as warmup)\n")
    print(f"{'Backend':<15} {'Mean Time (ms)':<20} {'Std Dev (ms)':<15}")
    print("-" * 70)
    
    for backend, stats in results.items():
        if stats is not None:
            print(f"{backend:<15} {stats['mean_ms']:<20.4f} {stats['std_ms']:<15.4f}")
        else:
            print(f"{backend:<15} {'Not available':<20} {'-':<15}")
    
    return results


def demonstrate_backend_consistency():
    """Verify results are consistent across backends"""
    print("\n" + "=" * 70)
    print("Backend Consistency Verification")
    print("=" * 70)
    
    n, layers = 4, 2
    
    # Compute with NumPy
    tq.set_backend("numpy")
    K = tq.get_backend()
    params_np = K.ones([layers, n])
    energy_np = float(compute_energy(n, layers, params_np))
    
    print(f"NumPy energy: {energy_np:.10f}")
    
    # Compute with PyTorch
    try:
        import torch
        tq.set_backend("pytorch")
        K = tq.get_backend()
        params_pt = K.ones([layers, n])
        energy_pt = float(compute_energy(n, layers, params_pt))
        
        print(f"PyTorch energy: {energy_pt:.10f}")
        print(f"Difference: {abs(energy_np - energy_pt):.2e}")
        
        if abs(energy_np - energy_pt) < 1e-6:
            print("✓ Results are consistent across backends!")
        else:
            print("⚠ Warning: Results differ (may be due to numerical precision)")
            
    except ImportError:
        print("PyTorch not available for comparison")


# ==================== Main Demo ====================

def main():
    """Run all backend demonstrations"""
    print("=" * 70)
    print("TyxonQ Numeric Backend Switching Demo")
    print("=" * 70)
    
    # Individual backend demos
    demo_numpy_backend()
    demo_pytorch_backend()
    demo_cupynumeric_backend()
    
    # Performance comparison
    compare_backends_performance()
    
    # Consistency check
    demonstrate_backend_consistency()
    
    print("\n" + "=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("1. TyxonQ provides unified backend interface (ArrayBackend protocol)")
    print("2. Switch backends via tq.set_backend('numpy'|'pytorch'|'cupynumeric')")
    print("3. Code remains unchanged when switching backends")
    print("4. PyTorch enables automatic differentiation + GPU acceleration")
    print("5. NumPy is lightweight and suitable for prototyping")
    print("6. CuPyNumeric provides GPU acceleration for NumPy-like code")
    
    print("\nBackend Selection Guide:")
    print("- Prototyping / small circuits → NumPy")
    print("- VQAs with gradients / GPU available → PyTorch")
    print("- Large statevector ops / GPU → CuPyNumeric (if installed)")
    print("- Production deployment → PyTorch (most flexible)")


if __name__ == "__main__":
    main()
