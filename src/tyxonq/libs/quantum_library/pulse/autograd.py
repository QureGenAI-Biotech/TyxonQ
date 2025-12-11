"""
PyTorch autograd support for pulse parameter optimization.

This module enables automatic differentiation through pulse simulation,
allowing gradient-based optimization of pulse parameters such as:
- Amplitude (amp)
- DRAG beta coefficient
- Pulse duration
- Drive frequency detuning

Key Features:
    - Preserves autograd chain through pulse evolution
    - Supports arbitrary pulse waveforms (Gaussian, DRAG, etc.)
    - Compatible with three-level system modeling
    - Integrates with existing StatevectorEngine

Examples:
    >>> import torch
    >>> from tyxonq import waveforms
    >>> from tyxonq.libs.quantum_library.pulse import DifferentiablePulseSimulation
    >>> 
    >>> # Define differentiable pulse parameters
    >>> amp = torch.tensor([1.0], requires_grad=True)
    >>> beta = torch.tensor([0.15], requires_grad=True)
    >>> 
    >>> # Optimize pulse to match target gate
    >>> sim = DifferentiablePulseSimulation()
    >>> optimal_params = sim.optimize_to_target(
    ...     initial_params={'amp': 1.0, 'beta': 0.15},
    ...     target_gate='X',
    ...     max_iter=100
    ... )
    >>> print(f"Optimal amp: {optimal_params['amp']:.4f}")
    >>> print(f"Optimal beta: {optimal_params['beta']:.4f}")
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
import warnings


class DifferentiablePulseSimulation:
    """Wrapper for pulse simulation with PyTorch autograd support.
    
    This class bridges TyxonQ's pulse simulation functions with PyTorch's
    automatic differentiation system, enabling end-to-end optimization of
    pulse parameters.
    
    Attributes:
        backend: PyTorch numerical backend
        
    Examples:
        >>> # Single parameter optimization
        >>> import torch
        >>> sim = DifferentiablePulseSimulation()
        >>> amp = torch.tensor([1.0], requires_grad=True)
        >>> 
        >>> # Simulate and get fidelity
        >>> fidelity = sim.compute_fidelity(
        ...     pulse_params={'amp': amp, 'duration': 160, 'sigma': 40, 'beta': 0.15},
        ...     target_unitary='X'
        ... )
        >>> 
        >>> # Compute gradient
        >>> loss = 1.0 - fidelity
        >>> loss.backward()
        >>> print(f"∂loss/∂amp = {amp.grad.item():.6f}")
    """
    
    def __init__(self, backend: str = "pytorch"):
        """Initialize differentiable pulse simulator.
        
        Args:
            backend: Numerical backend (must be "pytorch" for autograd)
            
        Raises:
            ValueError: If backend is not "pytorch"
            ImportError: If PyTorch is not available
        """
        if backend != "pytorch":
            raise ValueError(
                f"Autograd requires PyTorch backend, got '{backend}'. "
                "Please use backend='pytorch'."
            )
        
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for autograd support. "
                "Install with: pip install torch"
            )
        
        # Set TyxonQ backend to PyTorch
        try:
            import tyxonq as tq
            tq.set_backend("pytorch")
            from ....numerics.api import get_backend
            self.backend = get_backend()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PyTorch backend: {e}")
    
    def compute_fidelity(
        self,
        pulse_params: Dict[str, Union[float, Any]],
        target_unitary: Union[str, Any],
        qubit_freq: float = 5.0e9,
        drive_freq: Optional[float] = None,
        anharmonicity: float = -300e6,
        three_level: bool = False,
        rabi_freq: float = 50e6
    ) -> Any:
        """Compute gate fidelity for given pulse parameters.
        
        This function is fully differentiable when pulse_params contain
        torch tensors with requires_grad=True.
        
        Args:
            pulse_params: Pulse parameters dict, can contain torch tensors.
                         Required keys depend on pulse type:
                         - DRAG: 'amp', 'duration', 'sigma', 'beta'
                         - Gaussian: 'amp', 'duration', 'sigma'
            target_unitary: Target gate, either:
                           - String: 'X', 'Y', 'H', 'RX(pi/2)', etc.
                           - Tensor: 2×2 or 3×3 unitary matrix
            qubit_freq: Qubit frequency (Hz)
            drive_freq: Drive frequency (Hz), defaults to qubit_freq
            anharmonicity: Anharmonicity (Hz)
            three_level: Enable three-level simulation
            rabi_freq: Rabi frequency (Hz) for three-level sim
            
        Returns:
            fidelity: Gate fidelity ∈ [0, 1], differentiable w.r.t. pulse_params
            
        Examples:
            >>> import torch
            >>> sim = DifferentiablePulseSimulation()
            >>> 
            >>> # Optimize DRAG beta
            >>> beta = torch.tensor([0.1], requires_grad=True)
            >>> fid = sim.compute_fidelity(
            ...     pulse_params={'amp': 1.0, 'duration': 160, 'sigma': 40, 'beta': beta},
            ...     target_unitary='X'
            ... )
            >>> fid.backward()
            >>> print(f"Gradient: {beta.grad}")
        """
        # Build pulse waveform
        pulse = self._build_pulse(pulse_params)
        
        # Get target unitary
        U_target = self._get_target_unitary(target_unitary, three_level)
        
        # Simulate pulse evolution (now uses upgraded core libraries!)
        if three_level:
            from ..three_level_system import compile_three_level_unitary
            
            U_achieved = compile_three_level_unitary(
                pulse,
                qubit_freq=qubit_freq,
                drive_freq=drive_freq if drive_freq is not None else qubit_freq,
                anharmonicity=anharmonicity,
                rabi_freq=rabi_freq,
                backend=self.backend  # Will auto-select PyTorch version!
            )
            d = 3
        else:
            from ..pulse_simulation import compile_pulse_to_unitary
            
            U_achieved = compile_pulse_to_unitary(
                pulse,
                qubit_freq=qubit_freq,
                drive_freq=drive_freq if drive_freq is not None else qubit_freq,
                anharmonicity=anharmonicity,
                rabi_freq=rabi_freq,
                backend=self.backend  # Will auto-select PyTorch version!
            )
            d = 2
        
        # Compute fidelity: f = |tr(U†_target · U_achieved)| / d
        # This is fully differentiable when using PyTorch backend
        trace = self.torch.trace(
            self.torch.conj(U_target.T) @ U_achieved
        )
        fidelity = self.torch.abs(trace) / d
        
        return fidelity
    
    def optimize_to_target(
        self,
        initial_params: Dict[str, float],
        target_unitary: Union[str, Any],
        param_names: Optional[list[str]] = None,
        lr: float = 0.01,
        max_iter: int = 100,
        target_fidelity: float = 0.999,
        verbose: bool = True,
        **kwargs: Any
    ) -> Dict[str, float]:
        """Optimize pulse parameters to match target unitary.
        
        Uses PyTorch's Adam optimizer for gradient-based optimization.
        
        Args:
            initial_params: Initial parameter values
            target_unitary: Target gate (string or matrix)
            param_names: List of parameters to optimize. If None,
                        optimizes all parameters in initial_params.
            lr: Learning rate for Adam optimizer
            max_iter: Maximum iterations
            target_fidelity: Convergence threshold
            verbose: Print optimization progress
            **kwargs: Additional arguments passed to compute_fidelity
            
        Returns:
            Optimized parameter dictionary
            
        Examples:
            >>> sim = DifferentiablePulseSimulation()
            >>> 
            >>> # Optimize DRAG pulse for X gate
            >>> result = sim.optimize_to_target(
            ...     initial_params={'amp': 1.0, 'duration': 160, 'sigma': 40, 'beta': 0.1},
            ...     target_unitary='X',
            ...     param_names=['amp', 'beta'],  # Only optimize these
            ...     lr=0.01,
            ...     max_iter=200,
            ...     target_fidelity=0.9999
            ... )
            >>> print(f"Optimal parameters: {result}")
            >>> # Output: {'amp': 0.998, 'duration': 160, 'sigma': 40, 'beta': 0.152}
        """
        import torch
        
        # Determine which parameters to optimize
        if param_names is None:
            param_names = list(initial_params.keys())
        
        # Convert to torch tensors with gradients
        params_torch = {}
        params_fixed = {}
        
        for key, val in initial_params.items():
            if key in param_names:
                params_torch[key] = torch.tensor(
                    [float(val)],
                    requires_grad=True,
                    dtype=torch.float32
                )
            else:
                params_fixed[key] = val
        
        # Optimizer
        optimizer = torch.optim.Adam(list(params_torch.values()), lr=lr)
        
        # Optimization loop
        best_fidelity = 0.0
        best_params = None
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # Merge optimized and fixed parameters
            current_params = {}
            for key in params_torch:
                current_params[key] = params_torch[key].squeeze()
            for key in params_fixed:
                current_params[key] = params_fixed[key]
            
            # Compute fidelity
            fidelity = self.compute_fidelity(
                pulse_params=current_params,
                target_unitary=target_unitary,
                **kwargs
            )
            
            # Track best
            fid_val = fidelity.item()
            if fid_val > best_fidelity:
                best_fidelity = fid_val
                best_params = {k: v.item() if isinstance(v, torch.Tensor) else v 
                              for k, v in current_params.items()}
            
            # Loss: infidelity
            loss = 1.0 - fidelity
            
            # Backward
            loss.backward()
            
            # Gradient step
            optimizer.step()
            
            # Logging
            if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
                print(f"Iter {iteration:4d}: Fidelity = {fid_val:.6f}, Loss = {loss.item():.6e}")
            
            # Check convergence
            if fid_val >= target_fidelity:
                if verbose:
                    print(f"✅ Converged at iteration {iteration} with fidelity {fid_val:.6f}")
                break
        else:
            if verbose:
                print(f"⚠️  Max iterations reached. Best fidelity: {best_fidelity:.6f}")
        
        return best_params
    
    def _build_pulse(self, params: Dict[str, Any]) -> Any:
        """Build pulse waveform from parameter dict.
        
        IMPORTANT: Preserve tensor parameters to maintain autograd chain!
        
        Args:
            params: Parameter dict with keys like 'amp', 'duration', 'sigma', 'beta', 'phase'
            
        Returns:
            Pulse waveform object
        """
        from .... import waveforms
        
        # Determine pulse type based on available parameters
        if 'beta' in params:
            # DRAG pulse
            # DON'T extract values - keep tensors for autograd!
            return waveforms.Drag(
                amp=params.get('amp', 1.0),  # Keep as tensor if it's a tensor
                duration=int(self._extract_value(params.get('duration', 160))),
                sigma=self._extract_value(params.get('sigma', 40)),
                beta=params.get('beta', 0.15),  # Keep as tensor if it's a tensor
                phase=self._extract_value(params.get('phase', 0.0))
            )
        else:
            # Gaussian pulse
            return waveforms.Gaussian(
                amp=params.get('amp', 1.0),  # Keep as tensor if it's a tensor
                duration=int(self._extract_value(params.get('duration', 160))),
                sigma=self._extract_value(params.get('sigma', 40)),
                phase=self._extract_value(params.get('phase', 0.0))
            )
    
    def _extract_value(self, param: Union[float, Any]) -> float:
        """Extract numeric value from parameter (may be torch tensor)."""
        if self.torch.is_tensor(param):
            if param.numel() == 1:
                return param.item()
            else:
                return param
        return float(param)
    
    def _get_target_unitary(self, target: Union[str, Any], three_level: bool = False) -> Any:
        """Get target unitary matrix.
        
        Args:
            target: Gate name (string) or matrix
            three_level: Whether to use 3×3 matrices
            
        Returns:
            Target unitary as torch tensor
        """
        import torch
        
        if isinstance(target, str):
            # Build standard gate
            target = target.upper()
            
            if three_level:
                # 3×3 gates (leakage-free extensions)
                if target == 'X':
                    U = np.array([
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]
                    ], dtype=np.complex128)
                elif target == 'Y':
                    U = np.array([
                        [0, -1j, 0],
                        [1j, 0, 0],
                        [0, 0, 1]
                    ], dtype=np.complex128)
                elif target == 'H':
                    U = np.array([
                        [1, 1, 0],
                        [1, -1, 0],
                        [0, 0, np.sqrt(2)]
                    ], dtype=np.complex128) / np.sqrt(2)
                else:
                    raise ValueError(f"Unsupported 3-level gate: {target}")
            else:
                # 2×2 gates
                if target == 'X':
                    U = np.array([[0, 1], [1, 0]], dtype=np.complex128)
                elif target == 'Y':
                    U = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
                elif target == 'Z':
                    U = np.array([[1, 0], [0, -1]], dtype=np.complex128)
                elif target == 'H':
                    U = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
                elif target == 'I':
                    U = np.eye(2, dtype=np.complex128)
                else:
                    raise ValueError(f"Unsupported gate: {target}")
            
            return torch.tensor(U, dtype=torch.complex128)
        else:
            # Already a matrix, convert to torch if needed
            if not torch.is_tensor(target):
                return torch.tensor(target, dtype=torch.complex128)
            return target


def optimize_pulse_parameters(
    pulse_type: str,
    target_gate: str,
    initial_guess: Optional[Dict[str, float]] = None,
    optimize_params: Optional[list[str]] = None,
    lr: float = 0.01,
    max_iter: int = 100,
    target_fidelity: float = 0.999,
    verbose: bool = True,
    **sim_kwargs: Any
) -> Dict[str, float]:
    """Convenience function for pulse parameter optimization.
    
    Args:
        pulse_type: 'drag' or 'gaussian'
        target_gate: Target gate name ('X', 'Y', 'H', etc.)
        initial_guess: Initial parameter values
        optimize_params: List of parameters to optimize
        lr: Learning rate
        max_iter: Maximum iterations
        target_fidelity: Convergence threshold
        verbose: Print progress
        **sim_kwargs: Additional kwargs for simulation (qubit_freq, etc.)
        
    Returns:
        Optimized parameter dictionary
        
    Examples:
        >>> # Optimize DRAG pulse for X gate
        >>> result = optimize_pulse_parameters(
        ...     pulse_type='drag',
        ...     target_gate='X',
        ...     optimize_params=['amp', 'beta'],
        ...     max_iter=200
        ... )
        >>> print(f"Optimal amp: {result['amp']:.4f}")
        >>> print(f"Optimal beta: {result['beta']:.4f}")
    """
    # Default initial guesses
    if initial_guess is None:
        if pulse_type.lower() == 'drag':
            initial_guess = {
                'amp': 1.0,
                'duration': 160,
                'sigma': 40,
                'beta': 0.15
            }
        else:  # gaussian
            initial_guess = {
                'amp': 1.0,
                'duration': 160,
                'sigma': 40
            }
    
    # Default optimization parameters
    if optimize_params is None:
        if pulse_type.lower() == 'drag':
            optimize_params = ['amp', 'beta']
        else:
            optimize_params = ['amp']
    
    # Run optimization
    sim = DifferentiablePulseSimulation()
    
    result = sim.optimize_to_target(
        initial_params=initial_guess,
        target_unitary=target_gate,
        param_names=optimize_params,
        lr=lr,
        max_iter=max_iter,
        target_fidelity=target_fidelity,
        verbose=verbose,
        **sim_kwargs
    )
    
    return result
