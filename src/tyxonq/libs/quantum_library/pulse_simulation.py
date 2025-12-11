"""
Pulse-level quantum simulation engine for TyxonQ.

This module provides physics-based pulse simulation capabilities,
enabling time-dependent Hamiltonian evolution and pulse-to-unitary
compilation. Physical models are adapted from:

- QuTiP-qip: Quantum 6, 630 (2022) - "Pulse-level noisy quantum circuits"
- OpenPulse specification (OpenQASM 3.0)
- Scully & Zubairy: "Quantum Optics" (1997)

Implementation is completely independent with no QuTiP dependency,
supporting TyxonQ's unified ArrayBackend system (NumPy/PyTorch/CuPy).

Key Features
------------
- Dual-mode execution: direct state evolution or unitary compilation
- PyTorch autograd support for pulse parameter optimization
- Physical noise models (T1/T2 relaxation, frequency detuning)
- Backend-agnostic (works with any ArrayBackend implementation)

Examples
--------
# Direct state evolution (Mode B)
>>> psi0 = tq.statevector.zero_state(1)
>>> pulse = tq.waveforms.CosineDrag(duration=160, amp=1.0, phase=0, alpha=0.5)
>>> psi_final = evolve_pulse_hamiltonian(psi0, pulse, qubit=0, qubit_freq=5e9)

# Compile to unitary (Mode A: for chain API)
>>> U = compile_pulse_to_unitary(pulse, qubit_freq=5e9, drive_freq=5e9)
>>> psi_final = U @ psi0
"""

from typing import Any, Optional, Dict, Tuple
import numpy as np
from scipy.integrate import solve_ivp


# Physical constants
SAMPLING_RATE = 2e9  # 2 GHz sampling rate (standard for superconducting qubits)
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)


def sample_waveform(waveform: Any, t: float, backend: Any = None) -> Any:
    """
    Sample waveform envelope at time t.
    
    Converts waveform definition to time-domain amplitude.
    
    Parameters
    ----------
    waveform : waveform object
        Must have attributes: duration, amp, and waveform-specific params
    t : float
        Time in seconds
    backend : ArrayBackend, optional
        Numeric backend for array operations
        
    Returns
    -------
    complex or tensor
        Complex amplitude at time t (Ω(t) in Hamiltonian)
        Returns Python complex for numpy backend, tensor for PyTorch backend
        
    Notes
    -----
    Waveform formulas reference QuTiP-qip documentation:
    - CosineDrag: Ω(t) = (A/2)[cos(2πt/T - π) + 1] + iα·dΩ/dt
    - Gaussian: Ω(t) = A·exp(-(t-T/2)²/(2σ²))
    - Flattop: Ω(t) = (A/2)[erf((w+T-t)/σ) - erf((w-t)/σ)]
    """
    if backend is None:
        from ...numerics.api import get_backend
        backend = get_backend()
    
    # Detect if we're using PyTorch backend
    use_torch = backend.name == 'pytorch'
    
    waveform_type = waveform.__class__.__name__.lower()
    
    # Convert time to sample index
    t_sample = t * SAMPLING_RATE
    duration_samples = waveform.duration
    
    # Outside pulse duration → zero amplitude
    if t_sample < 0 or t_sample >= duration_samples:
        if use_torch:
            import torch
            return torch.tensor(0.0 + 0.0j, dtype=torch.complex128)
        return 0.0 + 0.0j
    
    # Normalized time: τ ∈ [0, 1]
    tau = t_sample / duration_samples
    
    # Waveform-specific sampling (formulas from QuTiP-qip paper)
    if waveform_type == "cosinedrag":
        # Cosine DRAG: suppresses leakage to |2⟩ state
        # Formula: g(τ) = (A/2)[cos(2πτ - π) + 1]
        #          Ω(τ) = g(τ) + iα·g'(τ)
        amp = waveform.amp
        phase = waveform.phase
        alpha = waveform.alpha
        
        g_tau = (amp / 2) * (np.cos(2 * np.pi * tau - np.pi) + 1)
        
        # Derivative for DRAG correction
        dg_dtau = -(amp * np.pi) * np.sin(2 * np.pi * tau - np.pi)
        
        # Apply phase rotation
        envelope = g_tau * np.exp(1j * phase) + 1j * alpha * dg_dtau
        if use_torch:
            import torch
            return torch.tensor(complex(envelope), dtype=torch.complex128)
        return complex(envelope)
    
    elif waveform_type == "drag":
        # DRAG pulse: Derivative Removal by Adiabatic Gate
        # Formula: Ω(τ) = [A·exp(-(τ-0.5)²/(2σ²)) + iβ·dΩ/dτ] · exp(iφ)
        amp = waveform.amp
        sigma = waveform.sigma / duration_samples
        beta = waveform.beta
        phase = getattr(waveform, 'phase', 0.0)
        
        # Smart dispatch: use PyTorch if parameters are tensors
        if use_torch:
            import torch
            # Convert to tensors if not already
            amp_t = amp if torch.is_tensor(amp) else torch.tensor(float(amp), dtype=torch.float64)
            beta_t = beta if torch.is_tensor(beta) else torch.tensor(float(beta), dtype=torch.float64)
            phase_t = torch.tensor(float(phase), dtype=torch.float64)
            
            # Gaussian envelope (use torch math)
            exponent = -((tau - 0.5)**2) / (2 * sigma**2)
            g_tau = amp_t * torch.exp(torch.tensor(exponent, dtype=torch.float64))
            dg_dtau = g_tau * (-(tau - 0.5) / sigma**2)
            
            # DRAG correction with phase
            envelope = (g_tau + 1j * beta_t * dg_dtau) * torch.exp(1j * phase_t)
            return envelope.to(torch.complex128)
        else:
            # Use numpy
            g_tau = amp * np.exp(-((tau - 0.5)**2) / (2 * sigma**2))
            dg_dtau = g_tau * (-(tau - 0.5) / sigma**2)
            envelope = (g_tau + 1j * beta * dg_dtau) * np.exp(1j * phase)
            return complex(envelope)
    
    elif waveform_type == "gaussian":
        # Gaussian pulse: smooth envelope
        # Formula: Ω(τ) = A·exp(-(τ-0.5)²/(2σ²))·exp(iφ)
        amp = waveform.amp
        sigma = waveform.sigma / duration_samples
        phase = getattr(waveform, 'phase', 0.0)
        angle = getattr(waveform, "angle", 0.0)  # Legacy support
        
        if use_torch:
            import torch
            amp_t = amp if torch.is_tensor(amp) else torch.tensor(float(amp), dtype=torch.float64)
            phase_t = torch.tensor(float(phase + angle), dtype=torch.float64)
            
            # Gaussian envelope (use torch math)
            exponent = -((tau - 0.5)**2) / (2 * sigma**2)
            envelope = amp_t * torch.exp(torch.tensor(exponent, dtype=torch.float64))
            envelope = envelope * torch.exp(1j * phase_t)
            return envelope.to(torch.complex128)
        else:
            # Use numpy
            envelope = amp * np.exp(-((tau - 0.5)**2) / (2 * sigma**2))
            envelope *= np.exp(1j * (phase + angle))
            return complex(envelope)
    
    elif waveform_type == "flattop":
        # Flat-top pulse: Gaussian rise/fall + constant plateau
        # Formula: Ω(t) = (A/2)[erf((w+T-t)/σ) - erf((w-t)/σ)]
        from scipy.special import erf
        
        amp = waveform.amp
        width_samples = waveform.width
        
        # Convert to time domain
        t_abs = t_sample
        T = duration_samples
        w = width_samples
        sigma = width_samples / 4  # Typical choice
        
        envelope = (amp / 2) * (
            erf((w + T - t_abs) / sigma) - erf((w - t_abs) / sigma)
        )
        if use_torch:
            import torch
            return torch.tensor(complex(envelope), dtype=torch.complex128)
        return complex(envelope)
    
    elif waveform_type == "sine":
        # Sine waveform: periodic oscillation
        # Formula: Ω(t) = A·sin(2πf·t + φ)·exp(iθ)
        amp = waveform.amp
        freq = waveform.frequency if hasattr(waveform, 'frequency') else getattr(waveform, 'freq', 0.1)
        phase = getattr(waveform, 'phase', 0.0)
        angle = getattr(waveform, 'angle', 0.0)
        
        t_abs = t_sample / SAMPLING_RATE
        envelope = amp * np.sin(2 * np.pi * freq * t_abs + phase)
        envelope *= np.exp(1j * angle)
        if use_torch:
            import torch
            return torch.tensor(complex(envelope), dtype=torch.complex128)
        return complex(envelope)
    
    elif waveform_type == "constant":
        # Constant amplitude (DC pulse)
        if use_torch:
            import torch
            return torch.tensor(complex(waveform.amp), dtype=torch.complex128)
        return complex(waveform.amp)
    
    elif waveform_type == "hermite":
        # Hermite polynomial envelope: minimal spectral leakage
        # Uses probabilist's Hermite polynomials for smooth envelope
        # Formula (order 2): H₂(x) = x² - 1
        #         (order 3): H₃(x) = x³ - 3x
        # Where x = (τ - 0.5) / σ (normalized centered coordinate)
        
        amp = waveform.amp
        order = int(waveform.order) if hasattr(waveform, 'order') else 2
        phase = getattr(waveform, 'phase', 0.0)
        
        # Normalized coordinate (centered at τ=0.5)
        x = (tau - 0.5) * 4  # Stretch to [-2, 2] for better coverage
        
        # Gaussian envelope (always present)
        sigma_envelope = 0.35  # Controls envelope width
        gaussian = np.exp(-(x**2) / (2 * sigma_envelope**2))
        
        # Hermite modulation
        if order == 2:
            # H₂(x) = x² - 1, normalized
            hermite_mod = 1.0 + (x**2 - 1) / 4.0
        elif order == 3:
            # H₃(x) = x³ - 3x, normalized
            hermite_mod = 1.0 + (x**3 - 3*x) / 6.0
        else:
            # Default: use Gaussian only
            hermite_mod = 1.0
        
        # Combine envelope and modulation
        envelope = amp * gaussian * np.maximum(hermite_mod, 0.01)  # Avoid zero
        envelope *= np.exp(1j * phase)
        
        if use_torch:
            import torch
            return torch.tensor(complex(envelope), dtype=torch.complex128)
        return complex(envelope)
    
    elif waveform_type == "blackman_square" or waveform_type == "blackmansquare":
        # Blackman window with flat-top: extremely low spectral sidelobes
        # Blackman window: w(t) = 0.42 - 0.5·cos(2πt/T) + 0.08·cos(4πt/T)
        # Applied as: ramp-up [0, w/2] → plateau [w/2, T-w/2] → ramp-down [T-w/2, T]
        
        amp = waveform.amp
        width_samples = waveform.width  # Plateau width
        duration_samples_total = waveform.duration
        phase = getattr(waveform, 'phase', 0.0)
        
        # Ramp duration (from plateau width)
        ramp_duration = (duration_samples_total - width_samples) / 2
        
        t_abs = t_sample  # Current sample
        
        # Determine region
        if t_abs < ramp_duration:
            # Ramp-up region [0, ramp_duration]
            t_ramp = t_abs / ramp_duration  # Normalized to [0, 1]
            # Blackman window
            window = 0.42 - 0.5 * np.cos(2 * np.pi * t_ramp) + 0.08 * np.cos(4 * np.pi * t_ramp)
            envelope = amp * window
        elif t_abs < ramp_duration + width_samples:
            # Plateau region [ramp_duration, ramp_duration + width_samples]
            envelope = amp  # Full amplitude
        else:
            # Ramp-down region [ramp_duration + width_samples, duration]
            t_ramp = (duration_samples_total - t_abs) / ramp_duration  # Normalized to [0, 1]
            # Blackman window
            window = 0.42 - 0.5 * np.cos(2 * np.pi * t_ramp) + 0.08 * np.cos(4 * np.pi * t_ramp)
            envelope = amp * window
        
        envelope *= np.exp(1j * phase)
        
        if use_torch:
            import torch
            return torch.tensor(complex(envelope), dtype=torch.complex128)
        return complex(envelope)
    
    else:
        raise ValueError(f"Unsupported waveform type: {waveform_type}")


def build_pulse_hamiltonian(
    pulse_waveform: Any,
    qubit_freq: float,
    drive_freq: float,
    anharmonicity: float = -300e6,
    rabi_freq: float = 50e6,  # NEW: Rabi frequency parameter
    backend: Any = None
) -> Tuple[Any, Any]:
    """
    Build time-dependent Hamiltonian for pulse evolution.
    
    Physical model (rotating wave approximation):
    
        H(t) = H_drift + H_drive(t)
        H_drift = (ω_q / 2) σ_z + (α / 2) σ_z²  (qubit + anharmonicity)
        H_drive(t) = Ω(t) [cos(ω_d·t + φ) σ_x + sin(ω_d·t + φ) σ_y]
    
    In rotating frame at drive frequency ω_d:
        H_RWA(t) = (Δ / 2) σ_z + Ω(t)/2 [σ_+ exp(-iφ) + σ_- exp(iφ)]
    
    where Δ = ω_q - ω_d (detuning)
    
    Parameters
    ----------
    pulse_waveform : waveform object
        Pulse envelope definition
    qubit_freq : float
        Qubit transition frequency (Hz), e.g., 5e9 for 5 GHz
    drive_freq : float
        Drive frequency (Hz), typically near qubit_freq
    anharmonicity : float, optional
        Qubit anharmonicity (Hz), default -300 MHz for transmon
    rabi_freq : float, optional
        Peak Rabi frequency (Hz), default 50 MHz
        This scales the normalized pulse amplitude to physical drive strength.
    backend : ArrayBackend, optional
        
    Returns
    -------
    H_drift : ndarray
        Time-independent drift Hamiltonian (2×2)
    H_drive_func : callable
        Function H_drive(t) returning drive Hamiltonian at time t
        
    Notes
    -----
    Reference: QuTiP-qip Processor model (Quantum 6, 630, 2022)
    Physical basis: Scully & Zubairy, "Quantum Optics" (1997)
    """
    if backend is None:
        from ...numerics.api import get_backend
        backend = get_backend()
    
    # Pauli matrices (2×2 qubit subspace)
    sigma_z = backend.array([[1, 0], [0, -1]], dtype=backend.complex128)
    sigma_x = backend.array([[0, 1], [1, 0]], dtype=backend.complex128)
    sigma_y = backend.array([[0, -1j], [1j, 0]], dtype=backend.complex128)
    
    # Drift Hamiltonian (in rotating frame)
    # H_drift = (Δ/2) σ_z where Δ = ω_q - ω_d
    detuning = qubit_freq - drive_freq
    H_drift = (detuning / 2) * sigma_z
    
    # Rabi frequency scaling (convert to angular frequency)
    rabi_angular = 2 * np.pi * rabi_freq
    
    # Drive Hamiltonian function (time-dependent)
    def H_drive(t):
        # Sample waveform envelope (normalized)
        omega_t = sample_waveform(pulse_waveform, t, backend)
        
        # Scale by physical Rabi frequency
        # omega_t is normalized (max ≈ pulse.amp), scale to target Rabi freq
        omega_drive = omega_t * rabi_angular
        
        # In RWA: H_drive = Ω(t)/2 [σ_+ exp(-iφ) + σ_- exp(iφ)]
        # Simplified (φ=0): H_drive = Ω(t)/2 (σ_x + iσ_y) + c.c.
        #                            = Ω(t) σ_x  (if Ω real)
        
        # For complex Ω(t), split into real/imaginary parts
        omega_real = backend.real(omega_drive)
        omega_imag = backend.imag(omega_drive)
        
        H_d = omega_real * sigma_x + omega_imag * sigma_y
        return H_d
    
    return H_drift, H_drive


def evolve_pulse_hamiltonian(
    initial_state: Any,
    pulse_waveform: Any,
    qubit: int = 0,
    qubit_freq: float = 5.0e9,
    drive_freq: Optional[float] = None,
    anharmonicity: float = -300e6,
    rabi_freq: float = 50e6,  # NEW: Rabi frequency parameter
    T1: Optional[float] = None,
    T2: Optional[float] = None,
    backend: Any = None
) -> Any:
    """
    Evolve quantum state under pulse-driven Hamiltonian (Direct Mode B).
    
    Solves time-dependent Schrödinger equation:
        iℏ ∂|ψ⟩/∂t = H(t)|ψ⟩
    
    Parameters
    ----------
    initial_state : ndarray
        Initial state vector (2^n,) for n-qubit system
    pulse_waveform : waveform object
        Pulse definition (CosineDrag, Gaussian, etc.)
    qubit : int, optional
        Target qubit index (default 0)
    qubit_freq : float, optional
        Qubit frequency in Hz (default 5 GHz)
    drive_freq : float, optional
        Drive frequency in Hz (default = qubit_freq, on-resonance)
    anharmonicity : float, optional
        Qubit anharmonicity in Hz (default -300 MHz)
    rabi_freq : float, optional
        Peak Rabi frequency in Hz (default 50 MHz)
    T1 : float, optional
        Amplitude damping time in seconds (if None, no relaxation)
    T2 : float, optional
        Dephasing time in seconds (if None, no dephasing)
    backend : ArrayBackend, optional
        
    Returns
    -------
    final_state : ndarray
        Evolved state vector (same shape as initial_state)
        
    Examples
    --------
    >>> psi0 = zero_state(1)  # |0⟩
    >>> pulse = CosineDrag(duration=160, amp=1.0, phase=0, alpha=0.5)
    >>> psi_x = evolve_pulse_hamiltonian(psi0, pulse)
    >>> # psi_x should be close to |1⟩ (X gate effect)
    
    Notes
    -----
    - Uses scipy.integrate.solve_ivp with RK45 method
    - Noise models (T1/T2) applied post-evolution (Kraus operators)
    - For multi-qubit systems, applies local Hamiltonian on target qubit
    """
    if backend is None:
        from ...numerics.api import get_backend
        backend = get_backend()
    
    if drive_freq is None:
        drive_freq = qubit_freq  # On-resonance by default
    
    # Build Hamiltonian
    H_drift, H_drive = build_pulse_hamiltonian(
        pulse_waveform, qubit_freq, drive_freq, anharmonicity, rabi_freq, backend
    )
    
    # Total Hamiltonian function
    def H_total_func(t):
        return H_drift + H_drive(t)
    
    # Time evolution duration
    duration_sec = pulse_waveform.duration / SAMPLING_RATE
    
    # Extract number of qubits
    n_qubits = int(np.log2(len(initial_state)))
    
    if n_qubits == 1:
        # Single qubit: direct evolution
        def dydt(t, y):
            # Schrödinger equation: iℏ dψ/dt = H(t)ψ
            # Normalized: dψ/dt = -i H(t) ψ  (ℏ=1 units)
            H_t = H_total_func(t)
            return -1j * (H_t @ y)
        
        # Solve ODE
        result = solve_ivp(
            dydt,
            (0, duration_sec),
            initial_state,
            method='RK45',
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )
        
        final_state = result.y[:, -1]
    
    else:
        # Multi-qubit: apply local Hamiltonian on target qubit
        # 构造作用在 qubit 上的 Hamiltonian，其他比特为 identity
        # H_total = I ⊗ ... ⊗ H_local ⊗ ... ⊗ I
        
        def dydt(t, y):
            # 获取单比特 Hamiltonian
            H_local = H_total_func(t)
            
            # 扩展到多比特系统（使用 Kronecker 积）
            # H_full = I_0 ⊗ I_1 ⊗ ... ⊗ H_qubit ⊗ ... ⊗ I_n
            H_full = _expand_single_qubit_operator(
                H_local, qubit, n_qubits, backend
            )
            
            # Schrödinger equation: dψ/dt = -i H(t) ψ
            return -1j * (H_full @ y)
        
        # Solve ODE
        result = solve_ivp(
            dydt,
            (0, duration_sec),
            initial_state,
            method='RK45',
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )
        
        final_state = result.y[:, -1]
    
    # Apply decoherence if T1/T2 specified
    if T1 is not None or T2 is not None:
        final_state = _apply_decoherence(
            final_state, duration_sec, T1, T2, backend
        )
    
    return backend.asarray(final_state)


def compile_pulse_to_unitary(
    pulse_waveform: Any,
    qubit_freq: float = 5.0e9,
    drive_freq: Optional[float] = None,
    anharmonicity: float = -300e6,
    rabi_freq: float = 50e6,  # NEW: Rabi frequency parameter
    backend: Any = None
) -> Any:
    """
    Compile pulse waveform to unitary matrix (for Chain Mode A).
    
    Computes U = T exp[-i ∫ H(t) dt] via time evolution of identity
    basis states.
    
    **Autograd Support**: When using PyTorch backend, this function
    preserves gradient information for automatic differentiation.
    
    Parameters
    ----------
    pulse_waveform : waveform object
        Pulse definition
    qubit_freq : float, optional
        Qubit frequency in Hz
    drive_freq : float, optional
        Drive frequency in Hz (default = qubit_freq)
    anharmonicity : float, optional
        Anharmonicity in Hz
    rabi_freq : float, optional
        Peak Rabi frequency in Hz (default 50 MHz)
    backend : ArrayBackend, optional
        
    Returns
    -------
    U : ndarray
        Unitary matrix (2×2 for single qubit)
        
    Examples
    --------
    >>> pulse = CosineDrag(duration=160, amp=1.0, phase=0, alpha=0.5)
    >>> U = compile_pulse_to_unitary(pulse)
    >>> # U should approximate X gate
    >>> np.allclose(U, [[0, 1], [1, 0]], atol=1e-2)
    True
    
    Notes
    -----
    Used in chain API: compile() → Pulse → unitary → apply to circuit
    
    **For PyTorch backend**: Uses differentiable Euler integration instead
    of scipy.solve_ivp to preserve autograd chain.
    """
    if backend is None:
        from ...numerics.api import get_backend
        backend = get_backend()
    
    # Check if using PyTorch backend for autograd support
    use_pytorch = backend.name == 'pytorch'
    
    if use_pytorch:
        # Use differentiable PyTorch integration
        return _compile_pulse_pytorch(
            pulse_waveform, qubit_freq, drive_freq, anharmonicity, rabi_freq, backend
        )
    else:
        # Use standard scipy integration (more accurate)
        return _compile_pulse_scipy(
            pulse_waveform, qubit_freq, drive_freq, anharmonicity, rabi_freq, backend
        )


def _compile_pulse_scipy(
    pulse_waveform: Any,
    qubit_freq: float,
    drive_freq: float,
    anharmonicity: float,
    rabi_freq: float,
    backend: Any
) -> Any:
    """Compile pulse using scipy integration (original implementation)."""
    # Evolve basis states |0⟩ and |1⟩
    psi0 = backend.array([1, 0], dtype=backend.complex128)
    psi1 = backend.array([0, 1], dtype=backend.complex128)
    
    col0 = evolve_pulse_hamiltonian(
        psi0, pulse_waveform, 
        qubit_freq=qubit_freq, 
        drive_freq=drive_freq,
        anharmonicity=anharmonicity,
        rabi_freq=rabi_freq,
        backend=backend
    )
    
    col1 = evolve_pulse_hamiltonian(
        psi1, pulse_waveform,
        qubit_freq=qubit_freq,
        drive_freq=drive_freq,
        anharmonicity=anharmonicity,
        rabi_freq=rabi_freq,
        backend=backend
    )
    
    # Stack columns to form unitary
    U = backend.stack([col0, col1], axis=1)
    
    return U


def _compile_pulse_pytorch(
    pulse_waveform: Any,
    qubit_freq: float,
    drive_freq: float,
    anharmonicity: float,
    rabi_freq: float,
    backend: Any
) -> Any:
    """Compile pulse using PyTorch differentiable integration.
    
    This version preserves autograd chain by using Euler method instead
    of scipy.solve_ivp.
    """
    import torch
    
    # Physical constants
    duration_sec = pulse_waveform.duration / SAMPLING_RATE
    detuning = qubit_freq - drive_freq
    rabi_angular = 2 * np.pi * rabi_freq
    
    # Pauli matrices
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    
    # Drift Hamiltonian
    H_drift = (detuning / 2) * sigma_z
    
    # Time evolution parameters
    num_steps = min(pulse_waveform.duration, 200)  # Limit for speed
    dt = duration_sec / num_steps
    
    # Evolution function
    def evolve_state(psi_init):
        psi = psi_init.clone()
        for step in range(num_steps):
            t = step * dt
            
            # Sample waveform (must be PyTorch compatible!)
            omega_t = sample_waveform(pulse_waveform, t, backend)
            omega_drive = omega_t * rabi_angular
            
            # Drive Hamiltonian
            H_drive = torch.real(omega_drive) * sigma_x + torch.imag(omega_drive) * sigma_y
            
            # Total Hamiltonian
            H_total = H_drift + H_drive
            
            # Euler step: ψ(t+dt) = ψ(t) - i*dt*H*ψ(t)
            psi = psi - 1j * dt * (H_total @ psi)
            
            # Normalize
            psi = psi / torch.sqrt(torch.sum(torch.abs(psi)**2))
        
        return psi
    
    # Evolve basis states
    psi0 = torch.tensor([1, 0], dtype=torch.complex128)
    psi1 = torch.tensor([0, 1], dtype=torch.complex128)
    
    col0 = evolve_state(psi0)
    col1 = evolve_state(psi1)
    
    # Stack to form unitary
    U = torch.stack([col0, col1], dim=1)
    
    return U


def _expand_single_qubit_operator(
    operator: Any,
    target_qubit: int,
    n_qubits: int,
    backend: Any
) -> Any:
    """
    Expand single-qubit operator to multi-qubit system via Kronecker product.
    
    Constructs: I_0 ⊗ I_1 ⊗ ... ⊗ operator_target ⊗ ... ⊗ I_n
    
    Parameters
    ----------
    operator : ndarray
        Single-qubit operator (2×2)
    target_qubit : int
        Index of target qubit (0-indexed)
    n_qubits : int
        Total number of qubits
    backend : ArrayBackend
        Numeric backend
        
    Returns
    -------
    expanded_op : ndarray
        Expanded operator (2^n × 2^n)
        
    Examples
    --------
    >>> X = np.array([[0, 1], [1, 0]])
    >>> # Apply X to qubit 1 in 2-qubit system: I ⊗ X
    >>> IX = _expand_single_qubit_operator(X, 1, 2, backend)
    >>> IX.shape
    (4, 4)
    """
    # Identity matrix
    I = backend.eye(2, dtype=backend.complex128)
    
    # Build Kronecker product
    result = None
    for i in range(n_qubits):
        if i == target_qubit:
            current = operator
        else:
            current = I
        
        if result is None:
            result = current
        else:
            result = backend.kron(result, current)
    
    return result


def _apply_decoherence(
    state: Any,
    duration: float,
    T1: Optional[float],
    T2: Optional[float],
    backend: Any
) -> Any:
    """
    Apply T1/T2 decoherence to state (post-evolution approximation).
    
    Uses Kraus operators for amplitude damping and dephasing.
    Note: This is an approximation for weak coupling regime.
    
    Parameters
    ----------
    state : ndarray
        Pure state vector (2,)
    duration : float
        Evolution time in seconds
    T1 : float, optional
        Amplitude damping time
    T2 : float, optional
        Dephasing time
    backend : ArrayBackend
        
    Returns
    -------
    state : ndarray
        State after decoherence (may be mixed, represented as vector)
        
    Notes
    -----
    For proper mixed-state handling, use DensityMatrixEngine.
    This function provides approximate decoherence for statevector mode.
    """
    # Convert to density matrix
    rho = backend.outer(state, backend.conj(state))
    
    # Amplitude damping (T1)
    if T1 is not None:
        gamma = 1 - backend.exp(-duration / T1)
        
        # Kraus operators (QuTiP-qip noise model)
        K0 = backend.array([
            [1, 0],
            [0, backend.sqrt(1 - gamma)]
        ], dtype=backend.complex128)
        
        K1 = backend.array([
            [0, backend.sqrt(gamma)],
            [0, 0]
        ], dtype=backend.complex128)
        
        # ρ → K0 ρ K0† + K1 ρ K1†
        rho = K0 @ rho @ backend.conj(K0.T) + K1 @ rho @ backend.conj(K1.T)
    
    # Phase damping (T2)
    if T2 is not None:
        # Pure dephasing: T_φ from T_φ^-1 = T_2^-1 - (2T_1)^-1
        if T1 is not None:
            T_phi_inv = 1/T2 - 1/(2*T1)
            if T_phi_inv > 0:
                T_phi = 1 / T_phi_inv
                lam = 1 - backend.exp(-duration / T_phi)
            else:
                lam = 0  # No additional dephasing
        else:
            lam = 1 - backend.exp(-duration / T2)
        
        # Kraus operators for dephasing
        K0_ph = backend.array([
            [1, 0],
            [0, backend.sqrt(1 - lam)]
        ], dtype=backend.complex128)
        
        K1_ph = backend.array([
            [0, 0],
            [0, backend.sqrt(lam)]
        ], dtype=backend.complex128)
        
        rho = K0_ph @ rho @ backend.conj(K0_ph.T) + K1_ph @ rho @ backend.conj(K1_ph.T)
    
    # Extract state (diagonal elements for population)
    # Note: This loses coherence information for mixed states
    # Proper handling requires DensityMatrixEngine
    
    # Get phase from complex number (compatible with all backends)
    import numpy as np
    phase_1 = np.arctan2(backend.imag(rho[1, 1]), backend.real(rho[1, 1]))
    
    state_approx = backend.array([
        backend.sqrt(backend.real(rho[0, 0])),
        backend.sqrt(backend.real(rho[1, 1])) * backend.exp(1j * phase_1)
    ], dtype=backend.complex128)
    
    return state_approx


__all__ = [
    "sample_waveform",
    "build_pulse_hamiltonian",
    "evolve_pulse_hamiltonian",
    "compile_pulse_to_unitary",
    "SAMPLING_RATE",
]
