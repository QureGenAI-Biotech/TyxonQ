from __future__ import annotations

from typing import Any, Tuple

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


class PyTorchBackend:
    """Array backend backed by PyTorch tensors."""

    name = "pytorch"
    available = _TORCH_AVAILABLE

    # dtype constants (only defined if torch is importable)
    if _TORCH_AVAILABLE:  # pragma: no cover - simple attribute wiring
        complex64 = torch.complex64
        complex128 = torch.complex128
        float32 = torch.float32
        float64 = torch.float64
        int8 = torch.int8
        int32 = torch.int32
        int64 = torch.int64
        bool = torch.bool
        int = torch.int64
    
    # Default dtype strings (can be overridden by set_dtype)
    dtypestr = "complex128"  # complex dtype for quantum states
    rdtypestr = "float64"     # real dtype for parameters/measurements
    
    def set_dtype(self, dtype_str: str) -> Tuple[Any, Any]:
        """Set default dtype for this backend.
        
        Args:
            dtype_str: One of "complex64", "complex128"
            
        Returns:
            Tuple of (complex_dtype, real_dtype)
        """
        if dtype_str == "complex64":
            self.dtypestr = "complex64"
            self.rdtypestr = "float32"
            return (self.complex64, self.float32)
        elif dtype_str == "complex128":
            self.dtypestr = "complex128"
            self.rdtypestr = "float64"
            return (self.complex128, self.float64)
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}. Use 'complex64' or 'complex128'.")

    def _to_torch_dtype(self, dtype: Any | None):  # pragma: no cover - small mapper
        if torch is None:
            raise RuntimeError("torch not available")
        if dtype is None:
            return None
        if isinstance(dtype, torch.dtype):
            return dtype
        try:
            import numpy as _np

            mapping = {
                _np.float32: torch.float32,
                _np.float64: torch.float64,
                _np.complex64: torch.complex64,
                _np.complex128: torch.complex128,
                float: torch.float32,
                complex: torch.complex64,
            }
            return mapping.get(dtype, None)
        except Exception:
            return None

    def array(self, data: Any, dtype: Any | None = None) -> Any:
        """Create tensor from data, preserving autograd if input has gradients.
        
        CRITICAL: This method MUST preserve requires_grad for autograd support.
        If input is already a tensor with requires_grad=True, the output must also
        have requires_grad=True to maintain gradient flow.
        """
        td = self._to_torch_dtype(dtype)
        if torch.is_tensor(data):
            # Preserve requires_grad when converting dtype
            result = data.to(td) if td is not None else data
            # CRITICAL: Ensure requires_grad is preserved
            if hasattr(data, 'requires_grad') and data.requires_grad:
                if not result.requires_grad:
                    result = result.requires_grad_(True)
            return result
        # For non-tensor data, convert to tensor
        # IMPORTANT: Use torch.as_tensor which shares data when possible
        return torch.as_tensor(data, dtype=td)

    def asarray(self, data: Any, dtype: Any | None = None) -> Any:
        """Convert to tensor with optional dtype, preserving autograd chain.
        
        CRITICAL: This method MUST preserve requires_grad for autograd support.
        This is called extensively in statevector operations and gate functions.
        If gradient chain is broken here, VQE optimization will fail!
        
        Args:
            data: Input data (array, tensor, list, etc.)
            dtype: Target dtype (optional). If None, infers from data.
                   Can be PyTorch dtype or NumPy dtype (auto-converted).
        
        Returns:
            PyTorch tensor with specified or inferred dtype
        """
        # If already a tensor
        if torch.is_tensor(data):
            if dtype is not None:
                td = self._to_torch_dtype(dtype)
                result = data.to(td) if td is not None else data
                # Preserve requires_grad
                if hasattr(data, 'requires_grad') and data.requires_grad:
                    if not result.requires_grad:
                        result = result.requires_grad_(True)
                return result
            return data
        
        # For non-tensor data, convert to tensor
        if dtype is not None:
            td = self._to_torch_dtype(dtype)
            return torch.as_tensor(data, dtype=td)
        return torch.as_tensor(data)

    def to_numpy(self, data: Any):  # type: ignore[override]
        return data.detach().cpu().numpy() if hasattr(data, "detach") else data

    def matmul(self, a: Any, b: Any) -> Any:
        return a @ b

    def dot(self, a: Any, b: Any) -> Any:
        """Dot product with automatic conversion to PyTorch tensors.
        
        Args:
            a: Matrix (array or tensor)
            b: Vector (array or tensor)
            
        Returns:
            a @ b as PyTorch tensor
        """
        # Convert to tensors if needed
        if not torch.is_tensor(a):
            a = torch.as_tensor(a)
        if not torch.is_tensor(b):
            b = torch.as_tensor(b)
        
        # Ensure same dtype and device
        if a.dtype != b.dtype:
            a = a.to(b.dtype)
        if a.device != b.device:
            a = a.to(b.device)
        
        return torch.matmul(a, b)

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        return torch.einsum(subscripts, *operands)

    # Array ops
    def reshape(self, a: Any, shape: Any) -> Any:
        return torch.reshape(a, shape)

    def moveaxis(self, a: Any, source: int, destination: int) -> Any:
        return torch.movedim(a, source, destination)

    def sum(self, a: Any, axis: int | None = None) -> Any:
        return torch.sum(a, dim=axis) if axis is not None else torch.sum(a)

    def mean(self, a: Any, axis: int | None = None) -> Any:
        return torch.mean(a, dim=axis) if axis is not None else torch.mean(a)

    def abs(self, a: Any) -> Any:
        return torch.abs(a)

    def real(self, a: Any) -> Any:
        return torch.real(a)

    def conj(self, a: Any) -> Any:
        return torch.conj(a)

    def diag(self, a: Any) -> Any:
        return torch.diag(a)

    def zeros(self, shape: tuple[int, ...], dtype: Any | None = None) -> Any:
        td = self._to_torch_dtype(dtype)
        return torch.zeros(shape, dtype=td)

    def ones(self, shape: tuple[int, ...], dtype: Any | None = None) -> Any:
        td = self._to_torch_dtype(dtype)
        return torch.ones(shape, dtype=td)

    def zeros_like(self, a: Any) -> Any:
        return torch.zeros_like(a)

    def ones_like(self, a: Any) -> Any:
        return torch.ones_like(a)

    def eye(self, n: int, dtype: Any | None = None) -> Any:
        td = self._to_torch_dtype(dtype)
        return torch.eye(n, dtype=td)

    def kron(self, a: Any, b: Any) -> Any:
        return torch.kron(a, b)

    # Elementary math
    def exp(self, a: Any) -> Any:
        return torch.exp(a)

    def sin(self, a: Any) -> Any:
        return torch.sin(a)

    def cos(self, a: Any) -> Any:
        return torch.cos(a)

    def sqrt(self, a: Any) -> Any:
        return torch.sqrt(a)

    def square(self, a: Any) -> Any:
        return torch.square(a)
    
    def copy(self, a: Any) -> Any:
        """Create a copy of the tensor."""
        return a.clone() if hasattr(a, 'clone') else torch.as_tensor(a).clone()
    
    def allclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if two tensors are element-wise equal within tolerance."""
        return torch.allclose(a, b, rtol=rtol, atol=atol)
    
    def isclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> Any:
        """Element-wise comparison with tolerance."""
        return torch.isclose(a, b, rtol=rtol, atol=atol)

    def log(self, a: Any) -> Any:
        return torch.log(a)

    def log2(self, a: Any) -> Any:
        return torch.log2(a)

    # Additional array ops
    def stack(self, arrays: Any, axis: int = 0) -> Any:
        """Stack tensors along new dimension."""
        return torch.stack(arrays, dim=axis)

    def concatenate(self, arrays: Any, axis: int = 0) -> Any:
        """Concatenate tensors along existing dimension."""
        return torch.cat(arrays, dim=axis)

    def arange(self, start: Any, stop: Any | None = None, step: Any = 1) -> Any:
        """Return evenly spaced values within interval."""
        if stop is None:
            return torch.arange(start)
        return torch.arange(start, stop, step)

    def linspace(self, start: Any, stop: Any, num: int = 50) -> Any:
        """Return evenly spaced numbers over interval."""
        return torch.linspace(start, stop, num)

    def transpose(self, a: Any, axes: Any | None = None) -> Any:
        """Transpose tensor dimensions."""
        if axes is None:
            return torch.t(a) if a.ndim == 2 else a.permute(*reversed(range(a.ndim)))
        return a.permute(*axes)

    def norm(self, a: Any, ord: Any | None = None, axis: Any | None = None) -> Any:
        """Matrix or vector norm."""
        if axis is None:
            return torch.linalg.norm(a, ord=ord)
        return torch.linalg.norm(a, ord=ord, dim=axis)

    def imag(self, a: Any) -> Any:
        """Imaginary part of complex tensor."""
        return torch.imag(a)

    def cast(self, a: Any, dtype: Any) -> Any:
        """Cast tensor to specified dtype."""
        td = self._to_torch_dtype(dtype)
        if td is None:
            return a
        return a.to(td) if hasattr(a, 'to') else torch.as_tensor(a, dtype=td)

    def sign(self, a: Any) -> Any:
        """Element-wise sign function."""
        return torch.sign(a)

    def outer(self, a: Any, b: Any) -> Any:
        """Outer product of vectors."""
        return torch.outer(a.flatten(), b.flatten())

    # Linear algebra
    def svd(self, a: Any, full_matrices: bool = False) -> Tuple[Any, Any, Any]:
        # torch.linalg.svd returns U, S, Vh similar to numpy when full_matrices=False
        U, S, Vh = torch.linalg.svd(a, full_matrices=full_matrices)
        return U, S, Vh

    def eigh(self, a: Any) -> Tuple[Any, Any]:
        """Eigenvalues and eigenvectors of Hermitian matrix."""
        return torch.linalg.eigh(a)

    def eig(self, a: Any) -> Tuple[Any, Any]:
        """Eigenvalues and eigenvectors of general matrix."""
        return torch.linalg.eig(a)

    def solve(self, a: Any, b: Any, assume_a: str = 'gen') -> Any:
        """Solve linear system ax = b."""
        return torch.linalg.solve(a, b)

    def inv(self, a: Any) -> Any:
        """Matrix inverse."""
        return torch.linalg.inv(a)

    def expm(self, a: Any) -> Any:
        """Matrix exponential."""
        return torch.linalg.matrix_exp(a)

    def tensordot(self, a: Any, b: Any, axes: Any = 2) -> Any:
        """Tensor dot product."""
        if isinstance(axes, int):
            dims_a = list(range(-axes, 0))
            dims_b = list(range(0, axes))
        else:
            dims_a, dims_b = axes
        return torch.tensordot(a, b, dims=(dims_a, dims_b))

    def rng(self, seed: int | None = None) -> Any:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        return g

    def normal(self, rng: Any, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        return torch.normal(mean=0.0, std=1.0, size=shape, generator=rng, dtype=dtype)

    # Discrete ops / sampling helpers
    def choice(self, rng: Any, a: int, *, size: int, p: Any | None = None) -> Any:
        # PyTorch doesn't expose choice with prob directly; implement via multinomial
        if p is None:
            probs = torch.full((a,), 1.0 / a, dtype=torch.float64)
        else:
            probs = torch.as_tensor(p, dtype=torch.float64)
        idx = torch.multinomial(probs, num_samples=size, replacement=True, generator=rng)
        return idx.cpu().numpy() if hasattr(idx, 'cpu') else idx

    def bincount(self, x: Any, minlength: int = 0) -> Any:
        t = torch.as_tensor(x, dtype=torch.int64)
        return torch.bincount(t, minlength=minlength).cpu().numpy()

    def nonzero(self, x: Any) -> Any:
        t = torch.as_tensor(x)
        nz = torch.nonzero(t, as_tuple=False).squeeze(-1)
        return (nz.cpu().numpy(),)

    def requires_grad(self, x: Any, flag: bool = True) -> Any:
        if hasattr(x, "requires_grad"):
            x.requires_grad_(flag)
        return x

    def detach(self, x: Any) -> Any:
        return x.detach() if hasattr(x, "detach") else x

    # Optional: expose vmap if available
    def vmap(self, fn):  # pragma: no cover - thin wrapper
        try:
            from torch.func import vmap as torch_vmap  # type: ignore

            return torch_vmap(fn)
        except Exception:  # pragma: no cover
            def _fallback(*args: Any, **kwargs: Any):
                return fn(*args, **kwargs)

            return _fallback

    # --- K-like helpers ---
    def jit(self, fn):  # PyTorch eager; return original or torch.compile if available
        try:
            compiled = torch.compile(fn)  # type: ignore[attr-defined]
            return compiled
        except Exception:
            return fn

    def jacfwd(self, fn, argnums: int = 0):
        """Forward-mode Jacobian computation.
        
        Uses torch.func.jacfwd when available (PyTorch >= 2.0).
        
        Args:
            fn: Function to compute Jacobian of
            argnums: Argument index to differentiate with respect to
            
        Returns:
            Function that computes Jacobian
        """
        try:
            from torch.func import jacfwd as torch_jacfwd
            return torch_jacfwd(fn, argnums=argnums)
        except ImportError:
            # Fallback: not available in older PyTorch
            import warnings
            warnings.warn(
                "torch.func.jacfwd requires PyTorch >= 2.0. "
                "Please upgrade PyTorch or use finite difference approximation.",
                RuntimeWarning
            )
            # Return a wrapper that raises an error when called
            def _unavailable(*args: Any, **kwargs: Any):
                raise NotImplementedError(
                    "jacfwd requires PyTorch >= 2.0. "
                    "Please upgrade PyTorch or implement finite difference manually."
                )
            return _unavailable
    
    def hessian(self, fn, argnums: int = 0):
        """Hessian computation.
        
        Uses torch.func.hessian when available (PyTorch >= 2.0).
        
        Args:
            fn: Function to compute Hessian of (must return scalar)
            argnums: Argument index to differentiate with respect to
            
        Returns:
            Function that computes Hessian matrix
        """
        try:
            from torch.func import hessian as torch_hessian
            return torch_hessian(fn, argnums=argnums)
        except ImportError:
            # Fallback: not available in older PyTorch
            import warnings
            warnings.warn(
                "torch.func.hessian requires PyTorch >= 2.0. "
                "Please upgrade PyTorch or use finite difference approximation.",
                RuntimeWarning
            )
            # Return a wrapper that raises an error when called
            def _unavailable(*args: Any, **kwargs: Any):
                raise NotImplementedError(
                    "hessian requires PyTorch >= 2.0. "
                    "Please upgrade PyTorch or implement finite difference manually."
                )
            return _unavailable

    def value_and_grad(self, fn, argnums: int | tuple[int, ...] = 0):
        """Return a function that computes both value and gradient.
        
        This implementation uses PyTorch autograd when possible, falling back
        to finite difference if gradient computation fails.
        
        Args:
            fn: Function to differentiate
            argnums: Argument indices to differentiate with respect to
            
        Returns:
            Wrapped function that returns (value, gradients)
        """

        def wrapped(*args: Any, **kwargs: Any):
            arg_idx = (argnums,) if isinstance(argnums, int) else tuple(argnums)
            
            try:
                # Autograd path: ensure all selected args have gradients enabled
                args_list = list(args)
                grad_vars = []  # Track variables that need gradients
                
                for i in arg_idx:
                    xi = args_list[i]
                    # Convert to tensor and enable gradient tracking
                    if isinstance(xi, torch.Tensor):
                        # If already a tensor, ensure it requires grad
                        ti = xi.detach().clone().requires_grad_(True)
                    else:
                        # Convert numpy/list to tensor
                        ti = torch.tensor(xi, dtype=torch.float64, requires_grad=True)
                    args_list[i] = ti
                    grad_vars.append(ti)
                
                # Evaluate function
                y = fn(*args_list, **kwargs)
                
                # Ensure output is a tensor
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y, dtype=torch.float64)
                
                # Compute gradients
                # Use create_graph=False to avoid retaining computation graph
                grads = torch.autograd.grad(
                    y, 
                    grad_vars, 
                    allow_unused=True, 
                    retain_graph=False, 
                    create_graph=False
                )
                
                # Check if any gradient is None (computational graph broken)
                if any(g is None for g in grads):
                    raise RuntimeError("Gradient is None - computation graph may be broken")
                
                # Convert outputs to numpy
                y_out = y.detach().cpu().numpy()
                if y_out.size == 1:
                    y_out = y_out.item()
                    
                grads_out = [g.detach().cpu().numpy() for g in grads]
                return y_out, (grads_out[0] if len(grads_out) == 1 else tuple(grads_out))
                
            except Exception as e:
                # Fallback to finite difference
                import numpy as _np
                import warnings
                
                warnings.warn(
                    f"PyTorch autograd failed ({type(e).__name__}: {e}), "
                    f"falling back to finite difference approximation",
                    RuntimeWarning
                )
                
                # Use original args for finite difference
                # Convert tensors to numpy to avoid autograd issues
                orig_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        orig_args.append(arg.detach().cpu().numpy())
                    else:
                        orig_args.append(arg)
                
                # Evaluate base value
                y0 = fn(*orig_args, **kwargs)
                try:
                    y0_scalar = float(y0)
                except Exception:
                    y0_scalar = _np.asarray(y0, dtype=_np.float64).item()
                
                eps = 1e-7
                grad_results = []
                
                for i in arg_idx:
                    xi = _np.asarray(orig_args[i], dtype=_np.float64)
                    g = _np.zeros_like(xi, dtype=_np.float64)
                    flat = g.reshape(-1)
                    xi_base = xi.reshape(-1)
                    
                    for k in range(flat.size):
                        x_plus = xi.copy().reshape(-1)
                        x_minus = xi.copy().reshape(-1)
                        x_plus[k] = xi_base[k] + eps
                        x_minus[k] = xi_base[k] - eps
                        
                        a_plus = list(orig_args)
                        a_plus[i] = x_plus.reshape(xi.shape)
                        a_minus = list(orig_args)
                        a_minus[i] = x_minus.reshape(xi.shape)
                        
                        y_plus = float(fn(*a_plus, **kwargs))
                        y_minus = float(fn(*a_minus, **kwargs))
                        flat[k] = (y_plus - y_minus) / (2.0 * eps)
                    
                    grad_results.append(g)
                
                return y0_scalar, (grad_results[0] if len(grad_results) == 1 else tuple(grad_results))

        return wrapped


