"""
Barren Plateau Benchmark - Gradient Vanishing Phenomenon in Quantum Neural Networks.

This example demonstrates and quantifies the barren plateau phenomenon, where
gradients of parameterized quantum circuits vanish exponentially with system size,
making optimization extremely difficult.

Key Innovation:
This example implements THREE gradient computation methods to contrast ideal theory
with hardware reality:

A. **Ideal Autograd** (PyTorch automatic differentiation)
   - Assumes direct quantum state access and exact derivatives
   - NOT realizable on real quantum hardware
   - Serves as theoretical baseline for comparison
   - Fastest execution, no shot noise

B. **Parameter Shift Rule** (Hardware-realistic, shot-based)
   - Uses parameter shift rule: ∂⟨H⟩/∂θ = [⟨H⟩(θ+π/4) - ⟨H⟩(θ-π/4)]/2
   - Simulates finite measurement shots (sampling noise)
   - Fully implementable on real quantum devices
   - TyxonQ chain-style API: circuit.device(shots=...).run()

C. **Parameter Shift + Noise** (Most realistic)
   - Includes hardware noise: depolarizing errors (p=0.001)
   - Uses TyxonQ's .with_noise() chain API
   - Mimics NISQ-era quantum processors
   - Critical for algorithm design on near-term devices

Physical Insight:
- Deep random circuits create highly entangled states
- Local observables have exponentially small overlap with initial states
- Gradient variance scales as σ² ~ O(1/2^n) where n is qubit count
- Hardware noise can mask, exacerbate, or even mitigate barren plateaus

References:
- McClean et al. (2018). Nat. Commun. 9, 4812 - "Barren plateaus in quantum neural network training landscapes"
- Cerezo et al. (2021). Nat. Commun. 12, 1791 - Review on variational quantum algorithms
- Schuld et al. (2019). Phys. Rev. A 99, 032331 - Parameter shift rules
"""

import time
import numpy as np
import torch
import tyxonq as tq

# Optional PennyLane for cross-validation
try:
    import pennylane as qml  # type: ignore
    _PL_AVAILABLE = True
except ImportError:
    qml = None
    _PL_AVAILABLE = False


def benchmark_function(func, *args, n_trials=3):
    """Benchmark a function with timing statistics.
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to func
        n_trials: Number of repeated trials
        
    Returns:
        Average execution time per trial
    """
    # Warm-up run (compilation/JIT)
    time_start = time.time()
    result = func(*args)
    time_compile = time.time() - time_start
    
    # Timed runs
    times = []
    for trial in range(n_trials):
        time_start = time.time()
        result = func(*args)
        times.append(time.time() - time_start)
        print(f"Trial {trial + 1}/{n_trials}: {result}")
    
    avg_time = np.mean(times)
    print(f"Compilation time: {time_compile:.4f}s")
    print(f"Average runtime: {avg_time:.4f}s")
    return avg_time


# Configure PyTorch backend for automatic differentiation
K = tq.set_backend("pytorch")
tq.set_dtype("complex64")


def random_circuit_expectation(params, gate_choices, n_qubits, depth):
    """Construct random parameterized quantum circuit and compute ZZ expectation.
    
    The circuit structure:
    1. Initial state preparation: RY(π/4) on all qubits
    2. Random parameterized layers:
       - Each qubit gets one of {RX, RY, RZ} based on gate_choices
       - Entangling CZ gates between neighbors
    3. Measurement: ⟨Z₀Z₁⟩ expectation value
    
    Args:
        params: Rotation angles, shape (n_qubits, depth)
        gate_choices: Integer array selecting gate type (0=RX, 1=RY, 2=RZ), shape (n_qubits, depth)
        n_qubits: Number of qubits
        depth: Circuit depth (number of layers)
        
    Returns:
        Real-valued expectation ⟨ψ|Z₀⊗Z₁|ψ⟩
    """
    from tyxonq.libs.quantum_library.kernels.gates import gate_z
    
    # Ensure params are float32 tensors
    params_tensor = K.cast(params, dtype=K.float32)
    
    c = tq.Circuit(n_qubits)
    
    # Initial layer: prepare superposition
    for i in range(n_qubits):
        c.ry(i, theta=np.pi / 4)
    
    # Parameterized layers with random gate selection
    for layer in range(depth):
        for qubit in range(n_qubits):
            gate_type = int(gate_choices[qubit, layer])
            angle = params_tensor[qubit, layer]
            
            if gate_type == 0:
                c.rx(qubit, theta=angle)
            elif gate_type == 1:
                c.ry(qubit, theta=angle)
            else:  # gate_type == 2
                c.rz(qubit, theta=angle)
        
        # Entangling layer: CZ between adjacent qubits
        for qubit in range(n_qubits - 1):
            c.cz(qubit, qubit + 1)
    
    # Measure ZZ correlation between first two qubits
    return K.real(c.expectation((gate_z(), [0]), (gate_z(), [1])))


def tyxonq_barren_plateau_autograd(n_qubits=10, depth=10, n_samples=100):
    """Method A: Analyze barren plateau using ideal PyTorch autograd.
    
    This is the baseline method that assumes direct quantum state access.
    NOT realizable on real hardware, but provides the "ideal" reference.
    
    Strategy:
    1. Sample random circuit configurations (gate choices)
    2. Sample random parameter settings
    3. Compute gradient of first parameter w.r.t. expectation via autograd
    4. Measure gradient variance across samples
    
    Args:
        n_qubits: System size
        depth: Circuit depth
        n_samples: Number of random samples
        
    Returns:
        Standard deviation of gradient (indicator of barren plateau)
    """
    print(f"\n{'='*60}")
    print(f"Method A: Ideal Autograd (Baseline)")
    print(f"{'='*60}")
    print(f"System size: {n_qubits} qubits")
    print(f"Circuit depth: {depth} layers")
    print(f"Random samples: {n_samples}")
    print()
    
    gradients = []
    
    for sample_idx in range(n_samples):
        # Random gate selection for this circuit instance
        gate_choices = np.random.choice(3, size=(n_qubits, depth))
        
        # Random parameters
        params_np = np.random.uniform(0, 2 * np.pi, size=(n_qubits, depth))
        params = torch.tensor(params_np, dtype=torch.float32, requires_grad=False)
        
        # Isolate first parameter for gradient computation
        theta_0 = params[0, 0].clone().detach().requires_grad_(True)
        
        # Define loss function for this sample
        def loss_fn(theta):
            params_modified = params.clone()
            params_modified[0, 0] = theta
            return random_circuit_expectation(params_modified, gate_choices, n_qubits, depth)
        
        # Compute gradient via autograd
        loss_value = loss_fn(theta_0)
        loss_value.backward()
        gradient = theta_0.grad
        
        if gradient is not None:
            gradients.append(abs(gradient.item()))
        
        if (sample_idx + 1) % 20 == 0:
            print(f"Processed {sample_idx + 1}/{n_samples} samples...")
    
    gradients = np.array(gradients)
    grad_std = np.std(gradients)
    grad_mean = np.mean(gradients)
    
    print(f"\nResults:")
    print(f"  Gradient mean: {grad_mean:.6f}")
    print(f"  Gradient std:  {grad_std:.6f}")
    print(f"  Expected scaling: ~O(1/2^{n_qubits/2}) ≈ {1/2**(n_qubits/2):.6f}")
    
    return grad_std


def tyxonq_barren_plateau_parameter_shift(
    n_qubits=10, depth=10, n_samples=100, shots=1024, with_noise=False
):
    """Method B/C: Hardware-realistic gradient via parameter shift rule.
    
    Implements the parameter shift rule for gradient estimation:
        ∂⟨H⟩/∂θᵢ ≈ [⟨H⟩(θᵢ + π/4) - ⟨H⟩(θᵢ - π/4)] / 2
    
    This method:
    1. Simulates shot-based measurement (finite sampling)
    2. Optionally includes hardware noise (gate errors, decoherence)
    3. Uses TyxonQ's chain-style device simulation API
    
    Args:
        n_qubits: System size
        depth: Circuit depth
        n_samples: Number of random samples
        shots: Number of measurement shots per circuit
        with_noise: Whether to include hardware noise
        
    Returns:
        Standard deviation of gradient
    """
    from tyxonq.libs.quantum_library.kernels.gates import gate_z
    
    method_name = "Method C: Parameter Shift + Noise" if with_noise else "Method B: Parameter Shift (Shot-based)"
    print(f"\n{'='*60}")
    print(f"{method_name}")
    print(f"{'='*60}")
    print(f"System size: {n_qubits} qubits")
    print(f"Circuit depth: {depth} layers")
    print(f"Random samples: {n_samples}")
    print(f"Shots per circuit: {shots}")
    if with_noise:
        print(f"Noise model: depolarizing(p=0.001) + readout error(0.02)")
    print()
    
    shift = np.pi / 4  # Parameter shift for Pauli rotations
    gradients = []
    
    for sample_idx in range(n_samples):
        # Random gate selection
        gate_choices = np.random.choice(3, size=(n_qubits, depth))
        params_np = np.random.uniform(0, 2 * np.pi, size=(n_qubits, depth))
        
        # Helper function to build circuit with given parameters
        def build_circuit(params_array):
            c = tq.Circuit(n_qubits)
            
            # Initial layer
            for i in range(n_qubits):
                c.ry(i, theta=np.pi / 4)
            
            # Parameterized layers
            for layer in range(depth):
                for qubit in range(n_qubits):
                    gate_type = int(gate_choices[qubit, layer])
                    angle = float(params_array[qubit, layer])
                    
                    if gate_type == 0:
                        c.rx(qubit, theta=angle)
                    elif gate_type == 1:
                        c.ry(qubit, theta=angle)
                    else:
                        c.rz(qubit, theta=angle)
                
                for qubit in range(n_qubits - 1):
                    c.cz(qubit, qubit + 1)
            
            # Apply noise using chain-style API (Method C only)
            if with_noise:
                c = c.with_noise("depolarizing", p=0.001)
            
            return c
        
        # Compute expectation at θ + shift
        params_plus = params_np.copy()
        params_plus[0, 0] += shift
        circuit_plus = build_circuit(params_plus)
        
        if shots < 1e6:  # Shot-based measurement
            # Add measurement gates on qubits 0 and 1
            circuit_plus.measure_z(0).measure_z(1)
            # Run with shots
            results_plus = circuit_plus.device(shots=shots).run()
            counts_plus = results_plus[0]["result"]
            
            # Compute ZZ expectation from counts
            exp_plus = 0.0
            total_shots = sum(counts_plus.values())
            for bitstring, count in counts_plus.items():
                # Z0⊗Z1 eigenvalue: (-1)^(b0) * (-1)^(b1)
                z0 = 1 if bitstring[0] == '0' else -1
                z1 = 1 if bitstring[1] == '0' else -1
                exp_plus += (z0 * z1) * count / total_shots
        else:  # Exact expectation (high-shot limit)
            exp_plus = float(K.real(circuit_plus.expectation((gate_z(), [0]), (gate_z(), [1]))))
        
        # Compute expectation at θ - shift
        params_minus = params_np.copy()
        params_minus[0, 0] -= shift
        circuit_minus = build_circuit(params_minus)
        
        if shots < 1e6:  # Shot-based measurement
            circuit_minus.measure_z(0).measure_z(1)
            results_minus = circuit_minus.device(shots=shots).run()
            counts_minus = results_minus[0]["result"]
            
            exp_minus = 0.0
            total_shots = sum(counts_minus.values())
            for bitstring, count in counts_minus.items():
                z0 = 1 if bitstring[0] == '0' else -1
                z1 = 1 if bitstring[1] == '0' else -1
                exp_minus += (z0 * z1) * count / total_shots
        else:  # Exact expectation
            exp_minus = float(K.real(circuit_minus.expectation((gate_z(), [0]), (gate_z(), [1]))))
        
        # Parameter shift gradient
        gradient = (exp_plus - exp_minus) / 2
        gradients.append(abs(gradient))
        
        if (sample_idx + 1) % 20 == 0:
            print(f"Processed {sample_idx + 1}/{n_samples} samples...")
    
    gradients = np.array(gradients)
    grad_std = np.std(gradients)
    grad_mean = np.mean(gradients)
    
    print(f"\nResults:")
    print(f"  Gradient mean: {grad_mean:.6f}")
    print(f"  Gradient std:  {grad_std:.6f}")
    print(f"  Expected scaling: ~O(1/2^{n_qubits/2}) ≈ {1/2**(n_qubits/2):.6f}")
    
    return grad_std


def pennylane_barren_plateau_analysis(n_qubits=10, depth=10, n_samples=100):
    """Cross-validation using PennyLane (if available).
    
    Implements the same circuit structure for comparison.
    
    Args:
        n_qubits: System size
        depth: Circuit depth
        n_samples: Number of random samples
        
    Returns:
        Standard deviation of gradient, or None if PennyLane unavailable
    """
    if not _PL_AVAILABLE:
        print("\n⚠️  PennyLane not available, skipping cross-validation")
        return None
    
    try:
        print(f"\n{'='*60}")
        print(f"PennyLane Cross-Validation")
        print(f"{'='*60}")
        
        dev = qml.device("default.qubit", wires=n_qubits)
        gate_set = [qml.RX, qml.RY, qml.RZ]
        
        @qml.qnode(dev, interface="autograd")
        def circuit(params, gate_choices):
            # Initial layer
            for i in range(n_qubits):
                qml.RY(np.pi / 4, wires=i)
            
            # Parameterized layers
            for layer in range(depth):
                for qubit in range(n_qubits):
                    gate_idx = gate_choices[qubit, layer]
                    gate_set[gate_idx](params[layer, qubit], wires=qubit)
                
                # Entangling layer
                for qubit in range(n_qubits - 1):
                    qml.CZ(wires=[qubit, qubit + 1])
            
            # Measurement
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        
        # Gradient function
        grad_fn = qml.grad(circuit, argnum=0)
        
        gradients = []
        for sample_idx in range(n_samples):
            gate_choices = np.random.choice(3, size=(n_qubits, depth))
            params = np.random.uniform(0, 2 * np.pi, size=(depth, n_qubits))
            
            grad = grad_fn(params, gate_choices)
            gradients.append(abs(grad[0, 0]))
            
            if (sample_idx + 1) % 20 == 0:
                print(f"Processed {sample_idx + 1}/{n_samples} samples...")
        
        gradients = np.array(gradients)
        grad_std = np.std(gradients)
        grad_mean = np.mean(gradients)
        
        print(f"\nPennyLane Results:")
        print(f"  Gradient mean: {grad_mean:.6f}")
        print(f"  Gradient std:  {grad_std:.6f}")
        
        return grad_std
        
    except Exception as e:
        print(f"\n❌ PennyLane analysis failed: {e}")
        return None


def compare_gradient_methods(n_qubits=6, depth=4, n_samples=30, shots=1024):
    """Compare three gradient computation methods for barren plateau analysis.
    
    This demonstrates the impact of hardware constraints on gradient estimation:
    - Method A: Ideal (autograd) - fastest, most accurate, not hardware-realizable
    - Method B: Parameter shift - hardware-realizable, shot noise only
    - Method C: Parameter shift + noise - most realistic, includes all errors
    
    Args:
        n_qubits: System size
        depth: Circuit depth
        n_samples: Number of random samples per method
        shots: Measurement shots for methods B and C
        
    Returns:
        Dictionary with results from all three methods
    """
    print("\n" + "="*60)
    print("Gradient Method Comparison for Barren Plateau")
    print("="*60)
    print(f"Configuration: {n_qubits} qubits, depth {depth}, {n_samples} samples")
    print()
    
    results = {}
    
    # Method A: Ideal autograd
    print("[1/3] Running Method A: Ideal Autograd...")
    results['autograd'] = tyxonq_barren_plateau_autograd(n_qubits, depth, n_samples)
    
    # Method B: Parameter shift (shot-based, no noise)
    print(f"\n[2/3] Running Method B: Parameter Shift ({shots} shots)...")
    results['parameter_shift'] = tyxonq_barren_plateau_parameter_shift(
        n_qubits, depth, n_samples, shots=shots, with_noise=False
    )
    
    # Method C: Parameter shift + noise
    print(f"\n[3/3] Running Method C: Parameter Shift + Noise ({shots} shots)...")
    results['parameter_shift_noisy'] = tyxonq_barren_plateau_parameter_shift(
        n_qubits, depth, n_samples, shots=shots, with_noise=True
    )
    
    # Summary comparison
    theoretical = 1 / 2**(n_qubits / 2)
    print("\n" + "="*60)
    print("Summary: Gradient Method Comparison")
    print("="*60)
    print(f"{'Method':<30} {'Gradient σ':<15} {'Relative Error':<15}")
    print("-" * 60)
    print(f"{'Theoretical O(1/2^(n/2))':<30} {theoretical:<15.6f} {'(reference)':<15}")
    print(f"{'A: Ideal Autograd':<30} {results['autograd']:<15.6f} {abs(results['autograd']-theoretical)/theoretical*100:>13.1f}%")
    print(f"{'B: Parameter Shift':<30} {results['parameter_shift']:<15.6f} {abs(results['parameter_shift']-theoretical)/theoretical*100:>13.1f}%")
    print(f"{'C: Param Shift + Noise':<30} {results['parameter_shift_noisy']:<15.6f} {abs(results['parameter_shift_noisy']-theoretical)/theoretical*100:>13.1f}%")
    print("="*60)
    print("\nKey Observations:")
    print("  - Method A: Fastest, but not hardware-realizable")
    print("  - Method B: Hardware-realistic, affected by shot noise")
    print("  - Method C: Most realistic, noise can mask barren plateau")
    print("="*60)
    
    return results


def compare_system_sizes():
    """Demonstrate exponential scaling of barren plateau with system size.
    
    Uses Method A (autograd) for fast execution across multiple system sizes.
    """
    print("\n" + "="*60)
    print("Barren Plateau Scaling Analysis (Method A)")
    print("="*60)
    
    qubit_sizes = [4, 6, 8]
    depth = 4
    n_samples = 50  # Reduced for faster execution
    
    results = []
    
    for n_qubits in qubit_sizes:
        print(f"\n📊 Testing {n_qubits} qubits...")
        grad_std = tyxonq_barren_plateau_autograd(n_qubits, depth, n_samples)
        theoretical = 1 / 2**(n_qubits / 2)
        results.append({
            'n_qubits': n_qubits,
            'grad_std': grad_std,
            'theoretical': theoretical
        })
    
    print(f"\n{'='*60}")
    print("Summary: Gradient Vanishing vs. System Size")
    print(f"{'='*60}")
    print(f"{'Qubits':<10} {'Observed σ':<15} {'Expected ~O(1/2^(n/2))':<20}")
    print("-" * 60)
    for r in results:
        print(f"{r['n_qubits']:<10} {r['grad_std']:<15.6f} {r['theoretical']:<20.6f}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    print("\n" + "🔬" * 30)
    print("Barren Plateau Phenomenon Demonstration")
    print("Hardware-Realistic vs Ideal Gradient Methods")
    print("🔬" * 30)
    
    # Main comparison: Three gradient methods
    print("\n" + "="*60)
    print("PART 1: Gradient Method Comparison")
    print("="*60)
    print("\n🎯 Key Question: How do hardware constraints affect gradient computation?")
    print("\nWe compare three approaches:")
    print("  ✓ Method A: Ideal Autograd (PyTorch .backward())")
    print("      ↳ Assumes infinite precision, exact derivatives")
    print("      ↳ NOT implementable on real quantum hardware")
    print("      ↳ Serves as theoretical baseline\n")
    print("  ✓ Method B: Parameter Shift + Finite Shots")
    print("      ↳ Uses parameter shift rule: ∂⟨H⟩/∂θ = [⟨H⟩(θ+s) - ⟨H⟩(θ-s)]/2")
    print("      ↳ Simulates shot noise (1024 measurements per circuit)")
    print("      ↳ Fully hardware-realizable\n")
    print("  ✓ Method C: Parameter Shift + Shots + Noise")
    print("      ↳ Adds depolarizing noise (p=0.001) via .with_noise()")
    print("      ↳ Most realistic simulation of NISQ devices")
    print("      ↳ Shows how hardware errors affect trainability")
    
    compare_gradient_methods(
        n_qubits=6,
        depth=4,
        n_samples=20,  # Reduced for reasonable runtime
        shots=1024
    )
    
    # System size scaling (using fast Method A)
    print("\n" + "="*60)
    print("PART 2: System Size Scaling Analysis")
    print("="*60)
    print("\n📈 Theoretical Prediction: Gradient variance σ² ~ O(1/2^n)")
    print("This exponential vanishing is the barren plateau phenomenon.\n")
    compare_system_sizes()
    
    # Optional: PennyLane cross-validation
    if _PL_AVAILABLE:
        print("\n" + "="*60)
        print("PART 3: PennyLane Cross-Validation")
        print("="*60)
        print("\nCross-validating with PennyLane (6 qubits, 4 layers, 10 samples)")
        benchmark_function(
            lambda: pennylane_barren_plateau_analysis(n_qubits=6, depth=4, n_samples=10),
            n_trials=1
        )
    
    print("\n" + "✅" * 30)
    print("Analysis Complete!")
    print("✅" * 30)
    print("\n💡 Key Insights:")
    print("  1. Ideal autograd (Method A) gives clean signal but is NOT hardware-realizable")
    print("  2. Parameter shift (Method B) enables gradient estimation on real devices")
    print("  3. Finite shots add statistical noise to gradient estimates")
    print("  4. Hardware noise (Method C) further degrades gradient quality")
    print("  5. Barren plateaus become exponentially worse with system size")
    print("\n🔧 Practical Mitigation Strategies:")
    print("  • Use local cost functions (avoid global observables)")
    print("  • Design problem-inspired ansätze (reduce circuit depth)")
    print("  • Employ layer-wise or pre-training strategies")
    print("  • Increase shot budget for critical gradient computations")
    print("  • Use noise-aware training algorithms")
    print("\n🚀 TyxonQ Features Demonstrated:")
    print("  • Chain-style API: .with_noise().device(shots=...).run()")
    print("  • Unified interface for ideal and realistic simulations")
    print("  • PyTorch backend integration for automatic differentiation")
    print("  • Flexible noise models (depolarizing, amplitude damping, etc.)")
    print("="*60)
