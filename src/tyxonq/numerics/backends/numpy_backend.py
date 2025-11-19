from __future__ import annotations

from typing import Any, Tuple

import numpy as np


class NumpyBackend:
    name = "numpy"

    # dtype constants
    import numpy as _np  # local alias to avoid polluting module
    complex64 = _np.complex64
    complex128 = _np.complex128
    float32 = _np.float32
    float64 = _np.float64
    int8 = _np.int8
    int32 = _np.int32
    int64 = _np.int64
    bool = _np.bool_
    int = _np.int64
    
    # Default dtype strings (can be overridden by set_dtype)
    dtypestr = "complex128"  # complex dtype for quantum states
    rdtypestr = "float64"     # real dtype for parameters/measurements
    
    def set_dtype(self, dtype_str: str) -> tuple[Any, Any]:
        """Set default dtype for this backend.
        
        Args:
            dtype_str: One of "complex64", "complex128"
            
        Returns:
            Tuple of (complex_dtype, real_dtype)
        """
        import numpy as _np
        if dtype_str == "complex64":
            self.dtypestr = "complex64"
            self.rdtypestr = "float32"
            return (_np.complex64, _np.float32)
        elif dtype_str == "complex128":
            self.dtypestr = "complex128"
            self.rdtypestr = "float64"
            return (_np.complex128, _np.float64)
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}. Use 'complex64' or 'complex128'.")

    def array(self, data: Any, dtype: Any | None = None) -> Any:
        return np.array(data, dtype=dtype)

    def asarray(self, data: Any, dtype: Any | None = None) -> Any:
        """Convert data to array with optional dtype specification.
        
        Args:
            data: Input data (list, array, tensor, etc.)
            dtype: Target dtype (optional). If None, infers from data.
        
        Returns:
            NumPy array with specified or inferred dtype
        """
        if dtype is not None:
            return np.asarray(data, dtype=dtype)
        return np.asarray(data)

    def to_numpy(self, data: Any) -> np.ndarray:  # type: ignore[override]
        return np.asarray(data)

    def matmul(self, a: Any, b: Any) -> Any:
        return np.matmul(a, b)

    def dot(self, a: Any, b: Any) -> Any:
        """Dot product (delegates to matmul for consistency).
        
        Args:
            a: Matrix (dense ndarray)
            b: Vector (ndarray)
            
        Returns:
            a @ b
        """
        return np.matmul(a, b)

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        return np.einsum(subscripts, *operands)

    # Array ops
    def reshape(self, a: Any, shape: Any) -> Any:
        return np.reshape(a, shape)

    def moveaxis(self, a: Any, source: int, destination: int) -> Any:
        return np.moveaxis(a, source, destination)

    def sum(self, a: Any, axis: int | None = None) -> Any:
        return np.sum(a, axis=axis)

    def mean(self, a: Any, axis: int | None = None) -> Any:
        return np.mean(a, axis=axis)

    def abs(self, a: Any) -> Any:
        return np.abs(a)

    def real(self, a: Any) -> Any:
        return np.real(a)

    def conj(self, a: Any) -> Any:
        return np.conj(a)

    def diag(self, a: Any) -> Any:
        return np.diag(a)

    def zeros(self, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        return np.ones(shape, dtype=dtype)

    def zeros_like(self, a: Any) -> Any:
        return np.zeros_like(a)

    def ones_like(self, a: Any) -> Any:
        return np.ones_like(a)

    def eye(self, n: int, dtype: Any | None = None) -> Any:
        return np.eye(n, dtype=dtype)

    def kron(self, a: Any, b: Any) -> Any:
        return np.kron(a, b)

    def square(self, a: Any) -> Any:
        return np.square(a)
    
    def copy(self, a: Any) -> Any:
        """Create a copy of the array."""
        return np.copy(a)
    
    def allclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if two arrays are element-wise equal within tolerance."""
        return np.allclose(a, b, rtol=rtol, atol=atol)
    
    def isclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> Any:
        """Element-wise comparison with tolerance."""
        return np.isclose(a, b, rtol=rtol, atol=atol)

    # Elementary math
    def exp(self, a: Any) -> Any:
        return np.exp(a)

    def sin(self, a: Any) -> Any:
        return np.sin(a)

    def cos(self, a: Any) -> Any:
        return np.cos(a)

    def sqrt(self, a: Any) -> Any:
        return np.sqrt(a)

    def log(self, a: Any) -> Any:
        return np.log(a)

    def log2(self, a: Any) -> Any:
        return np.log2(a)

    # Additional array ops
    def stack(self, arrays: Any, axis: int = 0) -> Any:
        """Stack arrays along new axis."""
        return np.stack(arrays, axis=axis)

    def concatenate(self, arrays: Any, axis: int = 0) -> Any:
        """Concatenate arrays along existing axis."""
        return np.concatenate(arrays, axis=axis)

    def arange(self, start: Any, stop: Any | None = None, step: Any = 1) -> Any:
        """Return evenly spaced values within interval."""
        if stop is None:
            return np.arange(start)
        return np.arange(start, stop, step)

    def linspace(self, start: Any, stop: Any, num: int = 50) -> Any:
        """Return evenly spaced numbers over interval."""
        return np.linspace(start, stop, num)

    def transpose(self, a: Any, axes: Any | None = None) -> Any:
        """Transpose array dimensions."""
        return np.transpose(a, axes=axes)

    def norm(self, a: Any, ord: Any | None = None, axis: Any | None = None) -> Any:
        """Matrix or vector norm."""
        return np.linalg.norm(a, ord=ord, axis=axis)

    def imag(self, a: Any) -> Any:
        """Imaginary part of complex array."""
        return np.imag(a)

    def cast(self, a: Any, dtype: Any) -> Any:
        """Cast array to specified dtype."""
        return np.asarray(a, dtype=dtype)

    def sign(self, a: Any) -> Any:
        """Element-wise sign function."""
        return np.sign(a)

    def outer(self, a: Any, b: Any) -> Any:
        """Outer product of vectors."""
        return np.outer(a, b)

    # Linear algebra
    def svd(self, a: Any, full_matrices: bool = False) -> Tuple[Any, Any, Any]:
        return np.linalg.svd(a, full_matrices=full_matrices)

    def eigh(self, a: Any) -> Tuple[Any, Any]:
        """Eigenvalues and eigenvectors of Hermitian matrix."""
        return np.linalg.eigh(a)

    def eig(self, a: Any) -> Tuple[Any, Any]:
        """Eigenvalues and eigenvectors of general matrix."""
        return np.linalg.eig(a)

    def solve(self, a: Any, b: Any, assume_a: str = 'gen') -> Any:
        """Solve linear system ax = b."""
        if assume_a == 'sym' or assume_a == 'her':
            # For symmetric/Hermitian, could use specialized solver
            return np.linalg.solve(a, b)
        return np.linalg.solve(a, b)

    def inv(self, a: Any) -> Any:
        """Matrix inverse."""
        return np.linalg.inv(a)

    def expm(self, a: Any) -> Any:
        """Matrix exponential."""
        import scipy.linalg
        return scipy.linalg.expm(a)

    def tensordot(self, a: Any, b: Any, axes: Any = 2) -> Any:
        """Tensor dot product."""
        return np.tensordot(a, b, axes=axes)

    def rng(self, seed: int | None = None) -> Any:
        return np.random.default_rng(seed)

    def normal(self, rng: Any, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        out = rng.normal(size=shape)
        return out.astype(dtype) if dtype is not None else out

    # Discrete ops / sampling helpers
    def choice(self, rng: Any, a: int, *, size: int, p: Any | None = None) -> Any:
        return rng.choice(a, size=size, p=p)

    def bincount(self, x: Any, minlength: int = 0) -> Any:
        return np.bincount(x, minlength=minlength)

    def nonzero(self, x: Any) -> Any:
        return np.nonzero(x)

    def requires_grad(self, x: Any, flag: bool = True) -> Any:
        return x

    def detach(self, x: Any) -> Any:
        return np.asarray(x)

    # --- K-like helpers (no-op/finite-diff implementations) ---
    def jit(self, fn):  # numpy has no JIT; return original
        return fn

    def jacfwd(self, fn, argnums: int = 0):
        """Forward-mode Jacobian via finite difference.
        
        Args:
            fn: Function to compute Jacobian of
            argnums: Argument index to differentiate with respect to
            
        Returns:
            Function that computes Jacobian matrix
        """
        eps = 1e-7
        
        def wrapped(*args: Any, **kwargs: Any):
            import numpy as _np
            
            args_list = list(args)
            x = _np.asarray(args_list[argnums], dtype=float)
            
            # Evaluate function at base point to get output shape
            y0 = fn(*args_list, **kwargs)
            y0 = _np.asarray(y0)
            
            # Jacobian shape: (output_size, input_size)
            x_flat = x.reshape(-1)
            y_flat = y0.reshape(-1)
            jac = _np.zeros((y_flat.size, x_flat.size))
            
            # Compute each column via finite difference
            for i in range(x_flat.size):
                x_plus = x_flat.copy()
                x_minus = x_flat.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                
                args_plus = list(args_list)
                args_minus = list(args_list)
                args_plus[argnums] = x_plus.reshape(x.shape)
                args_minus[argnums] = x_minus.reshape(x.shape)
                
                y_plus = _np.asarray(fn(*args_plus, **kwargs)).reshape(-1)
                y_minus = _np.asarray(fn(*args_minus, **kwargs)).reshape(-1)
                
                jac[:, i] = (y_plus - y_minus) / (2 * eps)
            
            # Reshape to (output_shape, input_shape)
            return jac.reshape(y0.shape + x.shape)
        
        return wrapped
    
    def hessian(self, fn, argnums: int = 0):
        """Hessian via finite difference.
        
        Args:
            fn: Function to compute Hessian of (must return scalar)
            argnums: Argument index to differentiate with respect to
            
        Returns:
            Function that computes Hessian matrix
        """
        eps = 1e-5
        
        def wrapped(*args: Any, **kwargs: Any):
            import numpy as _np
            
            args_list = list(args)
            x = _np.asarray(args_list[argnums], dtype=float)
            x_flat = x.reshape(-1)
            n = x_flat.size
            
            hess = _np.zeros((n, n))
            
            # Compute Hessian using central difference
            # H[i,j] = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*eps^2)
            for i in range(n):
                for j in range(i, n):  # Symmetric, only compute upper triangle
                    # Four evaluations for mixed partial
                    if i == j:
                        # Diagonal: use 3-point stencil
                        # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
                        x0 = x_flat.copy()
                        x_plus = x_flat.copy(); x_plus[i] += eps
                        x_minus = x_flat.copy(); x_minus[i] -= eps
                        
                        args0 = list(args_list)
                        args_plus = list(args_list)
                        args_minus = list(args_list)
                        
                        args0[argnums] = x0.reshape(x.shape)
                        args_plus[argnums] = x_plus.reshape(x.shape)
                        args_minus[argnums] = x_minus.reshape(x.shape)
                        
                        f0 = float(fn(*args0, **kwargs))
                        f_plus = float(fn(*args_plus, **kwargs))
                        f_minus = float(fn(*args_minus, **kwargs))
                        
                        hess[i, i] = (f_plus - 2*f0 + f_minus) / (eps**2)
                    else:
                        # Off-diagonal: mixed partial
                        x_pp = x_flat.copy(); x_pp[i] += eps; x_pp[j] += eps
                        x_pm = x_flat.copy(); x_pm[i] += eps; x_pm[j] -= eps
                        x_mp = x_flat.copy(); x_mp[i] -= eps; x_mp[j] += eps
                        x_mm = x_flat.copy(); x_mm[i] -= eps; x_mm[j] -= eps
                        
                        args_pp = list(args_list); args_pp[argnums] = x_pp.reshape(x.shape)
                        args_pm = list(args_list); args_pm[argnums] = x_pm.reshape(x.shape)
                        args_mp = list(args_list); args_mp[argnums] = x_mp.reshape(x.shape)
                        args_mm = list(args_list); args_mm[argnums] = x_mm.reshape(x.shape)
                        
                        f_pp = float(fn(*args_pp, **kwargs))
                        f_pm = float(fn(*args_pm, **kwargs))
                        f_mp = float(fn(*args_mp, **kwargs))
                        f_mm = float(fn(*args_mm, **kwargs))
                        
                        hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
                        hess[j, i] = hess[i, j]  # Symmetric
            
            # Reshape to input shape
            return hess.reshape(x.shape + x.shape)
        
        return wrapped

    def value_and_grad(self, fn, argnums: int | tuple[int, ...] = 0):
        """Compute value and gradient using optimized finite difference.
        
        Uses scipy.optimize.approx_fprime for efficient vectorized gradient computation
        when available, falls back to manual finite difference otherwise.
        
        Args:
            fn: Function to differentiate
            argnums: Argument index or indices to differentiate with respect to
            
        Returns:
            Wrapped function that returns (value, gradient(s))
        """
        eps = 1e-6

        def wrapped(*args: Any, **kwargs: Any) -> Tuple[Any, Any]:
            import numpy as _np
            from scipy.optimize import approx_fprime

            def _to_tuple(idx) -> tuple[int, ...]:
                return (idx,) if isinstance(idx, int) else tuple(idx)

            arg_idx = _to_tuple(argnums)
            args_list = list(args)
            
            # Evaluate function at base point
            val = fn(*args_list, **kwargs)
            grads: list[Any] = []
            
            for ai in arg_idx:
                x = _np.asarray(args_list[ai], dtype=float)
                original_shape = x.shape
                x_flat = x.reshape(-1)
                
                # Define objective function for this argument
                def f_for_grad(x_test):
                    args_test = list(args_list)
                    args_test[ai] = x_test.reshape(original_shape)
                    return float(fn(*args_test, **kwargs))
                
                # Use scipy's optimized finite difference for vectorized computation
                # approx_fprime 使用前向差分，比手动逐元素快
                try:
                    grad_flat = approx_fprime(x_flat, f_for_grad, epsilon=eps)
                except Exception:
                    # 降级：手动有限差分（带向量化优化）
                    grad_flat = _np.zeros_like(x_flat)
                    for i in range(x_flat.size):
                        x_plus = x_flat.copy()
                        x_minus = x_flat.copy()
                        x_plus[i] += eps
                        x_minus[i] -= eps
                        
                        args_plus = list(args_list)
                        args_minus = list(args_list)
                        args_plus[ai] = x_plus.reshape(original_shape)
                        args_minus[ai] = x_minus.reshape(original_shape)
                        
                        f_plus = float(fn(*args_plus, **kwargs))
                        f_minus = float(fn(*args_minus, **kwargs))
                        grad_flat[i] = (f_plus - f_minus) / (2 * eps)
                
                # Reshape gradient back to original shape
                grad = grad_flat.reshape(original_shape)
                grads.append(grad)
            
            return val, (grads[0] if len(grads) == 1 else tuple(grads))

        return wrapped


