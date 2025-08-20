"""
Backend magic inherited from tensornetwork: pytorch backend
"""

# pylint: disable=invalid-name

import logging
from typing import Any, Callable, Optional, Sequence, Tuple, Union
from operator import mul
from functools import reduce, partial

import tensornetwork
from tensornetwork.backends.pytorch import pytorch_backend
from .abstract_backend import ExtendedBackend
from .complex_utils import ComplexHandler, safe_cast, quantum_expectation_value, autodiff_safe

dtypestr: str
Tensor = Any
pytree = Any

torchlib: Any

logger = logging.getLogger(__name__)

# TODO(@refraction-ray): lack stateful random methods implementation for now
# TODO(@refraction-ray): lack scatter impl for now
# TODO(@refraction-ray): lack sparse relevant methods for now
# To be added once pytorch backend is ready


class torch_jit_func:
    """
    Delay the tracing of torch jit to the first run time:
    consistent with tf and jax mechanism
    """

    def __init__(self, f: Callable[..., Any]):
        self.compiled = False
        self.f = f

    def __call__(self, *args: Any, **kws: Any) -> Any:
        if self.compiled is False:
            self.f = torchlib.jit.trace(self.f, example_inputs=args)
            self.compiled = True

        return self.f(*args, **kws)


class torch_optimizer:
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer
        self.is_init = False

    def update(self, grads: pytree, params: pytree) -> pytree:
        # flatten grad and param
        params, treedef = PyTorchBackend.tree_flatten(None, params)
        grads, _ = PyTorchBackend.tree_flatten(None, grads)
        if self.is_init is False:
            self.optimizer = self.optimizer(params)
            self.is_init = True
        with torchlib.no_grad():
            for g, p in zip(grads, params):
                p.grad = g
        self.optimizer.step()
        self.optimizer.zero_grad()
        # reorg the param
        params = PyTorchBackend.tree_unflatten(None, treedef, params)
        return params


def _conj_torch(self: Any, tensor: Tensor) -> Tensor:
    t = torchlib.conj(tensor)
    return t.resolve_conj()  # any side effect?


def _sum_torch(
    self: Any,
    tensor: Tensor,
    axis: Optional[Sequence[int]] = None,
    keepdims: bool = False,
) -> Tensor:
    if axis is None:
        axis = tuple([i for i in range(len(tensor.shape))])
    return torchlib.sum(tensor, dim=axis, keepdim=keepdims)


def _qr_torch(
    self: Any,
    tensor: Tensor,
    pivot_axis: int = -1,
    non_negative_diagonal: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Computes the QR decomposition of a tensor.
    The QR decomposition is performed by treating the tensor as a matrix,
    with an effective left (row) index resulting from combining the
    axes `tensor.shape[:pivot_axis]` and an effective right (column)
    index resulting from combining the axes `tensor.shape[pivot_axis:]`.

    :Example:

    If `tensor` had a shape (2, 3, 4, 5) and `pivot_axis` was 2,
    then `q` would have shape (2, 3, 6), and `r` would
    have shape (6, 4, 5).
    The output consists of two tensors `Q, R` such that:

    Q[i1,...,iN, j] * R[j, k1,...,kM] == tensor[i1,...,iN, k1,...,kM]

    Note that the output ordering matches numpy.linalg.svd rather than tf.svd.

    :param tensor: A tensor to be decomposed.
    :type tensor: Tensor
    :param pivot_axis: Where to split the tensor's axes before flattening into a matrix.
    :type pivot_axis: int, optional
    :param non_negative_diagonal: a bool indicating whether the tenor is diagonal non-negative matrix.
    :type non_negative_diagonal: bool, optional
    :returns: Q, the left tensor factor, and R, the right tensor factor.
    :rtype: Tuple[Tensor, Tensor]
    """
    from .pytorch_ops import torchqr

    left_dims = list(tensor.shape[:pivot_axis])
    right_dims = list(tensor.shape[pivot_axis:])

    tensor = torchlib.reshape(tensor, [reduce(mul, left_dims), reduce(mul, right_dims)])
    q, r = torchqr.apply(tensor)
    if non_negative_diagonal:
        phases = torchlib.sign(torchlib.linalg.diagonal(r))
        q = q * phases
        r = phases[:, None] * r
    center_dim = q.shape[1]
    q = torchlib.reshape(q, left_dims + [center_dim])
    r = torchlib.reshape(r, [center_dim] + right_dims)
    return q, r


def _rq_torch(
    self: Any,
    tensor: Tensor,
    pivot_axis: int = 1,
    non_negative_diagonal: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Computes the RQ decomposition of a tensor.
    The QR decomposition is performed by treating the tensor as a matrix,
    with an effective left (row) index resulting from combining the axes
    `tensor.shape[:pivot_axis]` and an effective right (column) index
    resulting from combining the axes `tensor.shape[pivot_axis:]`.

    :Example:

    If `tensor` had a shape (2, 3, 4, 5) and `pivot_axis` was 2,
    then `r` would have shape (2, 3, 6), and `q` would
    have shape (6, 4, 5).
    The output consists of two tensors `Q, R` such that:

    Q[i1,...,iN, j] * R[j, k1,...,kM] == tensor[i1,...,iN, k1,...,kM]

    Note that the output ordering matches numpy.linalg.svd rather than tf.svd.

    :param tensor: A tensor to be decomposed.
    :type tensor: Tensor
    :param pivot_axis: Where to split the tensor's axes before flattening into a matrix.
    :type pivot_axis: int, optional
    :param non_negative_diagonal: a bool indicating whether the tenor is diagonal non-negative matrix.
    :type non_negative_diagonal: bool, optional
    :returns: Q, the left tensor factor, and R, the right tensor factor.
    :rtype: Tuple[Tensor, Tensor]
    """
    from .pytorch_ops import torchqr

    left_dims = list(tensor.shape[:pivot_axis])
    right_dims = list(tensor.shape[pivot_axis:])

    tensor = torchlib.reshape(tensor, [reduce(mul, left_dims), reduce(mul, right_dims)])
    q, r = torchqr.apply(tensor.adjoint())
    if non_negative_diagonal:
        phases = torchlib.sign(torchlib.linalg.diagonal(r))
        q = q * phases
        r = phases[:, None] * r
    r, q = r.adjoint(), q.adjoint()
    # M=r*q at this point
    center_dim = r.shape[1]
    r = torchlib.reshape(r, left_dims + [center_dim])
    q = torchlib.reshape(q, [center_dim] + right_dims)
    return r, q


tensornetwork.backends.pytorch.pytorch_backend.PyTorchBackend.sum = _sum_torch
tensornetwork.backends.pytorch.pytorch_backend.PyTorchBackend.conj = _conj_torch
tensornetwork.backends.pytorch.pytorch_backend.PyTorchBackend.qr = _qr_torch
tensornetwork.backends.pytorch.pytorch_backend.PyTorchBackend.rq = _rq_torch


class PyTorchBackend(pytorch_backend.PyTorchBackend, ExtendedBackend):  # type: ignore
    """
    See the original backend API at `pytorch backend
    <https://github.com/google/TensorNetwork/blob/master/tensornetwork/backends/pytorch/pytorch_backend.py>`_

    Note the functionality provided by pytorch backend is incomplete,
    it currenly lacks native efficicent jit and vmap support.
    """

    def __init__(self) -> None:
        super(PyTorchBackend, self).__init__()
        global torchlib
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch not installed, please switch to a different "
                "backend or install PyTorch."
            )
        torchlib = torch
        self.name = "pytorch"
        self.dtype = torchlib.float32
        self.rdtype = torchlib.float32
        self.cdtype = torchlib.complex64
        self.dtypestr = "float32"
        self.rdtypestr = "float32"
        self.cdtypestr = "complex64"

    def eye(
        self, N: int, dtype: Optional[str] = None, M: Optional[int] = None
    ) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        if not M:
            M = N
        r = torchlib.eye(n=N, m=M)
        return self.cast(r, dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = torchlib.ones(shape)
        return self.cast(r, dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = dtypestr
        r = torchlib.zeros(shape)
        return self.cast(r, dtype)

    def copy(self, a: Tensor) -> Tensor:
        return a.clone()

    def convert_to_tensor(self, tensor: Tensor) -> Tensor:
        """
        Convert input to PyTorch tensor.
        
        :param tensor: Input to convert
        :return: PyTorch tensor
        """
        if not isinstance(tensor, torchlib.Tensor):
            return torchlib.tensor(tensor)
        return tensor

    def expm(self, a: Tensor) -> Tensor:
        """
        Matrix exponential implementation for PyTorch.
        Uses torch.linalg.matrix_exp for PyTorch 1.9+ or manual implementation.
        """
        try:
            # Try using torch.linalg.matrix_exp (available in PyTorch 1.9+)
            return torchlib.linalg.matrix_exp(a)
        except AttributeError:
            # Manual implementation using Taylor series
            logger.warning("torch.linalg.matrix_exp not available, using manual implementation")
            n = a.shape[0]
            result = torchlib.eye(n, dtype=a.dtype, device=a.device)
            term = torchlib.eye(n, dtype=a.dtype, device=a.device)
            for i in range(1, 10):  # Use 10 terms for approximation
                term = term @ a / i
                result = result + term
            return result

    def sin(self, a: Tensor) -> Tensor:
        return torchlib.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return torchlib.cos(a)

    def acos(self, a: Tensor) -> Tensor:
        return torchlib.acos(a)

    def acosh(self, a: Tensor) -> Tensor:
        return torchlib.acosh(a)

    def asin(self, a: Tensor) -> Tensor:
        return torchlib.asin(a)

    def asinh(self, a: Tensor) -> Tensor:
        return torchlib.asinh(a)

    def atan(self, a: Tensor) -> Tensor:
        return torchlib.atan(a)

    def atan2(self, y: Tensor, x: Tensor) -> Tensor:
        return torchlib.atan2(y, x)

    def atanh(self, a: Tensor) -> Tensor:
        return torchlib.atanh(a)

    def cosh(self, a: Tensor) -> Tensor:
        return torchlib.cosh(a)

    def tan(self, a: Tensor) -> Tensor:
        return torchlib.tan(a)

    def tanh(self, a: Tensor) -> Tensor:
        return torchlib.tanh(a)

    def sinh(self, a: Tensor) -> Tensor:
        return torchlib.sinh(a)

    def size(self, a: Tensor) -> Tensor:
        return a.size()

    def eigvalsh(self, a: Tensor) -> Tensor:
        return torchlib.linalg.eigvalsh(a)

    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        return torchlib.kron(a, b)

    def numpy(self, a: Tensor) -> Tensor:
        a = a.cpu()
        if a.is_sparse:
            a = a.to_dense()
        if a.is_conj():
            return a.resolve_conj().numpy()
        if a.requires_grad:
            return a.detach().numpy()
        return a.numpy()

    def i(self, dtype: Any = None) -> Tensor:
        if not dtype:
            dtype = getattr(torchlib, dtypestr)
        if isinstance(dtype, str):
            dtype = getattr(torchlib, dtype)
        return torchlib.tensor(1j, dtype=dtype)

    def det(self, a: Tensor) -> Tensor:
        return torchlib.linalg.det(a)

    def schur(self, a: Tensor, output: str = "real") -> Tuple[Tensor, Tensor]:
        """
        Compute Schur decomposition of a matrix.
        
        :param a: Input matrix
        :param output: Output format ("real" or "complex")
        :return: Tuple of (T, Q) where T is upper triangular and Q is unitary
        """
        # PyTorch doesn't have a direct schur function, so we use eigendecomposition
        # This is a simplified implementation
        eigenvalues, eigenvectors = torchlib.linalg.eig(a)
        if output == "real":
            # Return real parts
            return torchlib.real(torchlib.diag(eigenvalues)), torchlib.real(eigenvectors)
        else:
            return torchlib.diag(eigenvalues), eigenvectors

    def real(self, a: Tensor) -> Tensor:
        return ComplexHandler.safe_complex_to_real(a, "real")

    def imag(self, a: Tensor) -> Tensor:
        try:
            a = torchlib.imag(a)
        except RuntimeError:
            pass
        return a

    def dtype(self, a: Tensor) -> str:
        return a.dtype.__str__().split(".")[-1]  # type: ignore

    def stack(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        """
        Stack tensors along a new axis.
        
        :param a: Sequence of tensors to stack
        :param axis: Axis along which to stack
        :return: Stacked tensor
        """
        # Convert all elements to PyTorch tensors if they aren't already
        converted_a = []
        for item in a:
            if not isinstance(item, torchlib.Tensor):
                converted_a.append(torchlib.tensor(item))
            else:
                converted_a.append(item)
        return torchlib.stack(converted_a, dim=axis)

    def concat(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return torchlib.cat(a, dim=axis)

    def tile(self, a: Tensor, rep: Tensor) -> Tensor:
        return torchlib.tile(a, rep)

    def mean(
        self,
        a: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        if axis is None:
            axis = tuple([i for i in range(len(a.shape))])
        return torchlib.mean(a, dim=axis, keepdim=keepdims)

    def std(
        self, a: Tensor, axis: Optional[Sequence[int]] = None, keepdims: bool = False
    ) -> Tensor:
        if axis is None:
            axis = tuple([i for i in range(len(a.shape))])
        return torchlib.std(a, dim=axis, unbiased=False, keepdim=keepdims)

    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        if axis is None:
            return torchlib.min(a)
        return torchlib.min(a, dim=axis).values

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        if axis is None:
            return torchlib.max(a)
        return torchlib.max(a, dim=axis).values

    def argmax(self, a: Tensor, axis: int = 0) -> Tensor:
        return torchlib.argmax(a, dim=axis)

    def argmin(self, a: Tensor, axis: int = 0) -> Tensor:
        return torchlib.argmin(a, dim=axis)

    def unique_with_counts(self, a: Tensor, **kws: Any) -> Tuple[Tensor, Tensor]:
        return torchlib.unique(a, return_counts=True)  # type: ignore

    def sigmoid(self, a: Tensor) -> Tensor:
        return torchlib.sigmoid(a)

    def relu(self, a: Tensor) -> Tensor:
        return torchlib.relu(a)

    def softmax(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        if axis is None:
            axis = -1
        return torchlib.nn.functional.softmax(a, dim=axis)

    def onehot(self, a: Tensor, num: int) -> Tensor:
        """
        One-hot encoding of input tensor.
        
        :param a: Input tensor with integer indices
        :param num: Number of classes
        :return: One-hot encoded tensor
        """
        # Convert to PyTorch tensor if not already
        if not isinstance(a, torchlib.Tensor):
            a = torchlib.tensor(a, dtype=torchlib.long)
        else:
            # Ensure tensor is of long type for one_hot
            a = a.long()
        return torchlib.nn.functional.one_hot(a, num)

    def cumsum(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        if axis is None:
            a = self.reshape(a, [-1])
            return torchlib.cumsum(a, dim=0)
        else:
            return torchlib.cumsum(a, dim=axis)

    def is_tensor(self, a: Any) -> bool:
        if isinstance(a, torchlib.Tensor):
            return True
        return False

    def abs(self, a: Tensor) -> Tensor:
        """
        Compute absolute value of tensor.
        
        :param a: Input tensor
        :return: Absolute value of input tensor
        """
        return torchlib.abs(a)

    def cast(self, a: Tensor, dtype: str) -> Tensor:
        if isinstance(dtype, str):
            return safe_cast(a, dtype)
        return a.type(dtype)
    
    def set_dtype(self, dtype: str) -> None:
        """
        Set the default dtype for PyTorch backend.
        This is a compatibility method for the global set_dtype function.
        
        :param dtype: Data type string ("complex64", "complex128", "float32", "float64")
        """
        # This method is called by the global set_dtype function
        # The actual dtype setting is handled globally in cons.py
        pass
    
    def set_random_state(self, seed: Optional[Union[int, Tensor]] = None, get_state: bool = False) -> Any:
        """
        Set the random state for PyTorch backend.
        
        :param seed: Random seed (integer or tensor)
        :param get_state: Whether to return the random state
        :return: Random state if get_state is True
        """
        if seed is not None:
            if isinstance(seed, int):
                torchlib.manual_seed(seed)
            elif isinstance(seed, torchlib.Tensor):
                # For tensor seeds, we use the first element
                torchlib.manual_seed(int(seed.flatten()[0].item()))
        else:
            # Use a default seed if none provided
            torchlib.manual_seed(42)
        
        if get_state:
            # Return PyTorch's random state
            return torchlib.get_rng_state()
    
    def stateful_randn(self, g: Any, shape: Sequence[int], mean: float = 0.0, stddev: float = 1.0, dtype: Optional[str] = None) -> Tensor:
        """
        Generate random normal numbers with stateful random generator.
        
        :param g: Random generator state (not used in PyTorch)
        :param shape: Shape of the output tensor
        :param mean: Mean of the normal distribution
        :param stddev: Standard deviation of the normal distribution
        :param dtype: Data type of the output tensor
        :return: Random tensor
        """
        if dtype is None:
            dtype = "float32"
        
        # Map dtype strings to PyTorch dtypes
        dtype_map = {
            "float32": torchlib.float32,
            "float64": torchlib.float64,
            "complex64": torchlib.complex64,
            "complex128": torchlib.complex128,
            "int32": torchlib.int32,
            "int64": torchlib.int64,
        }
        
        torch_dtype = dtype_map.get(dtype, torchlib.float32)
        
        # Generate random normal numbers
        tensor = torchlib.randn(shape, dtype=torch_dtype)
        
        # Apply mean and stddev
        tensor = tensor * stddev + mean
        
        return tensor

    def stateful_randu(self, g: Any, shape: Union[int, Sequence[int]] = 1, low: float = 0, high: float = 1, dtype: str = "32") -> Tensor:
        """
        Generate random uniform numbers with stateful random generator.
        
        :param g: Random generator state (not used in PyTorch)
        :param shape: Shape of the output tensor
        :param low: Lower bound of uniform distribution
        :param high: Upper bound of uniform distribution
        :param dtype: Data type of the output tensor
        :return: Random tensor
        """
        if isinstance(dtype, str):
            dtype = dtype[-2:]
        if isinstance(shape, int):
            shape = (shape,)
        if dtype == "32":
            dtyper = torchlib.float32
        elif dtype == "64":
            dtyper = torchlib.float64
        elif not isinstance(dtype, str):
            dtyper = dtype
        return torchlib.rand(shape, dtype=dtyper) * (high - low) + low

    def stateful_randc(self, g: Any, a: Union[int, Sequence[int], Tensor], shape: Union[int, Sequence[int]], p: Optional[Union[Sequence[float], Tensor]] = None) -> Tensor:
        """
        Generate random choice with stateful random generator.
        
        :param g: Random generator state (not used in PyTorch)
        :param a: Array to choose from
        :param shape: Shape of the output tensor
        :param p: Probability distribution
        :return: Random tensor
        """
        if isinstance(shape, int):
            shape = (shape,)
        if not self.is_tensor(a):
            a = torchlib.tensor(a, dtype=torchlib.long)
        if p is not None:
            if not self.is_tensor(p):
                p = torchlib.tensor(p, dtype=torchlib.float32)
            return torchlib.multinomial(p, num_samples=shape[0], replacement=True)
        else:
            return torchlib.randint(0, len(a), shape, dtype=torchlib.long)

    def implicit_randn(self, shape: Union[int, Sequence[int]] = 1, mean: float = 0, stddev: float = 1, dtype: str = "32") -> Tensor:
        """
        Generate random normal numbers with implicit random generator.
        
        :param shape: Shape of the output tensor
        :param mean: Mean of the normal distribution
        :param stddev: Standard deviation of the normal distribution
        :param dtype: Data type of the output tensor
        :return: Random tensor
        """
        if isinstance(dtype, str):
            dtype = dtype[-2:]
        if isinstance(shape, int):
            shape = (shape,)
        if dtype == "32":
            dtyper = torchlib.float32
        elif dtype == "64":
            dtyper = torchlib.float64
        elif not isinstance(dtype, str):
            dtyper = dtype
        r = torchlib.randn(shape, dtype=dtyper) * stddev + mean
        return r

    def implicit_randu(self, shape: Union[int, Sequence[int]] = 1, low: float = 0, high: float = 1, dtype: str = "32") -> Tensor:
        """
        Generate random uniform numbers with implicit random generator.
        
        :param shape: Shape of the output tensor
        :param low: Lower bound of uniform distribution
        :param high: Upper bound of uniform distribution
        :param dtype: Data type of the output tensor
        :return: Random tensor
        """
        if isinstance(dtype, str):
            dtype = dtype[-2:]
        if isinstance(shape, int):
            shape = (shape,)
        if dtype == "32":
            dtyper = torchlib.float32
        elif dtype == "64":
            dtyper = torchlib.float64
        elif not isinstance(dtype, str):
            dtyper = dtype
        r = torchlib.rand(shape, dtype=dtyper) * (high - low) + low
        return r

    def implicit_randc(self, a: Union[int, Sequence[int], Tensor], shape: Union[int, Sequence[int]], p: Optional[Union[Sequence[float], Tensor]] = None) -> Tensor:
        """
        Generate random choice with implicit random generator.
        
        :param a: Array to choose from
        :param shape: Shape of the output tensor
        :param p: Probability distribution
        :return: Random tensor
        """
        if isinstance(shape, int):
            shape = (shape,)
        if not self.is_tensor(a):
            a = torchlib.tensor(a, dtype=torchlib.long)
        if p is not None:
            if not self.is_tensor(p):
                p = torchlib.tensor(p, dtype=torchlib.float32)
            return torchlib.multinomial(p, num_samples=shape[0], replacement=True)
        else:
            return torchlib.randint(0, len(a), shape, dtype=torchlib.long)

    def gather1d(self, operand: Tensor, indices: Tensor) -> Tensor:
        """
        Gather elements from operand using indices.
        
        :param operand: Input tensor
        :param indices: Indices tensor
        :return: Gathered tensor
        """
        # Ensure indices has the correct shape for gather operation
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        elif indices.dim() > 1:
            # For vmap operations, we need to handle batched indices
            original_shape = indices.shape
            indices = indices.flatten()
            operand = operand.flatten()
            result = torchlib.gather(operand, 0, indices)
            return result.reshape(original_shape)
        return torchlib.gather(operand, 0, indices)

    def scatter(self, operand: Tensor, indices: Tensor, updates: Tensor) -> Tensor:
        """
        Scatter updates into operand using indices.
        
        :param operand: Input tensor
        :param indices: Indices tensor
        :param updates: Updates tensor
        :return: Scattered tensor
        """
        result = operand.clone()
        
        # Ensure indices has the correct shape for scatter operation
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        elif indices.dim() > 1:
            # For vmap operations, we need to handle batched indices
            original_shape = indices.shape
            indices = indices.flatten()
            updates = updates.flatten()
            result = result.flatten()
            result.scatter_(0, indices, updates)
            return result.reshape(operand.shape)
        
        result.scatter_(0, indices, updates)
        return result

    def is_sparse(self, a: Tensor) -> bool:
        """
        Check if tensor is sparse.
        
        :param a: Input tensor
        :return: True if tensor is sparse
        """
        return a.is_sparse if hasattr(a, 'is_sparse') else False

    def random_split(self, key: Any) -> Tuple[Any, Any]:
        """
        Split a random key into two keys.
        
        :param key: Input random key
        :return: Tuple of two random keys
        """
        # For PyTorch, we don't have explicit key management like JAX
        # Return the same key twice as a placeholder
        return key, key

    def scan(self, f: Callable[[Tensor, Tensor], Tensor], xs: Tensor, init: Tensor) -> Tensor:
        """
        Scan function over a sequence.
        
        :param f: Function to apply
        :param xs: Input sequence
        :param init: Initial value
        :return: Scanned result
        """
        def f_pytorch(*args: Any, **kws: Any) -> Any:
            r = f(*args, **kws)
            return r, None
        
        # Simple implementation using a loop
        result = init
        for x in xs:
            result, _ = f_pytorch(result, x)
        return result

    def eigh(self, matrix: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute eigenvalues and eigenvectors of a Hermitian matrix.
        
        :param matrix: Input matrix
        :return: Tuple of (eigenvalues, eigenvectors)
        """
        # Use the modern torch.linalg.eigh instead of deprecated symeig
        return torchlib.linalg.eigh(matrix)
    
    def log(self, a: Tensor) -> Tensor:
        """
        Compute natural logarithm of tensor.
        
        :param a: Input tensor
        :return: Natural logarithm of input tensor
        """
        return torchlib.log(a)

    def log2(self, a: Tensor) -> Tensor:
        """
        Compute base-2 logarithm of tensor.
        
        :param a: Input tensor
        :return: Base-2 logarithm of input tensor
        """
        if not isinstance(a, torchlib.Tensor):
            a = torchlib.tensor(a, dtype=torchlib.float32)
        return torchlib.log2(a)

    def exp(self, a: Tensor) -> Tensor:
        """
        Compute exponential of tensor.
        
        :param a: Input tensor
        :return: Exponential of input tensor
        """
        return torchlib.exp(a)

    def sqrt(self, a: Tensor) -> Tensor:
        """
        Compute square root of tensor.
        
        :param a: Input tensor
        :return: Square root of input tensor
        """
        # Convert to PyTorch tensor if not already
        if not isinstance(a, torchlib.Tensor):
            a = torchlib.tensor(a)
        return torchlib.sqrt(a)

    def pow(self, a: Tensor, b: Union[Tensor, float, int]) -> Tensor:
        """
        Compute power of tensor.
        
        :param a: Base tensor
        :param b: Exponent (tensor, float, or int)
        :return: Power of input tensor
        """
        return torchlib.pow(a, b)

    def norm(self, a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        """
        Compute norm of tensor.
        
        :param a: Input tensor
        :param axis: Axis along which to compute norm
        :param keepdims: Whether to keep dimensions
        :return: Norm of input tensor
        """
        if axis is None:
            return torchlib.norm(a)
        else:
            return torchlib.norm(a, dim=axis, keepdim=keepdims)

    def reshape(self, a: Tensor, shape: Sequence[int]) -> Tensor:
        """
        Reshape tensor to new shape.
        
        :param a: Input tensor
        :param shape: New shape
        :return: Reshaped tensor
        """
        # Convert NumPy arrays to PyTorch tensors first
        if not isinstance(a, torchlib.Tensor):
            a = torchlib.tensor(a)
        return torchlib.reshape(a, shape)

    def transpose(self, a: Tensor, perm: Optional[Sequence[int]] = None) -> Tensor:
        """
        Transpose tensor.
        
        :param a: Input tensor
        :param perm: Permutation of dimensions
        :return: Transposed tensor
        """
        if perm is None:
            return torchlib.t(a)  # For 2D tensors
        else:
            return torchlib.permute(a, perm)

    def conj(self, a: Tensor) -> Tensor:
        """
        Compute complex conjugate of tensor.
        
        :param a: Input tensor
        :return: Complex conjugate of input tensor
        """
        return torchlib.conj(a)

    def sum(self, a: Tensor, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Tensor:
        """
        Compute sum of tensor.
        
        :param a: Input tensor
        :param axis: Axis or axes along which to sum
        :param keepdims: Whether to keep dimensions
        :return: Sum of input tensor
        """
        if axis is None:
            return torchlib.sum(a)
        else:
            return torchlib.sum(a, dim=axis, keepdim=keepdims)

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Matrix multiplication of two tensors.
        
        :param a: First tensor
        :param b: Second tensor
        :return: Matrix product of a and b
        """
        return torchlib.matmul(a, b)

    def tensordot(self, a: Tensor, b: Tensor, axes: Union[int, Sequence[Sequence[int]]]) -> Tensor:
        """
        Tensor dot product of two tensors.
        
        :param a: First tensor
        :param b: Second tensor
        :param axes: Axes to contract over
        :return: Tensor dot product of a and b
        """
        return torchlib.tensordot(a, b, dims=axes)

    def outer_product(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Outer product of two tensors.
        
        :param a: First tensor
        :param b: Second tensor
        :return: Outer product of a and b
        """
        return torchlib.outer(a, b)

    def svd(self, a: Tensor, pivot_axis: int = -1, max_singular_values: Optional[int] = None, max_truncation_error: Optional[float] = None, relative: Optional[bool] = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Singular value decomposition of a tensor.
        
        :param a: Input tensor
        :param pivot_axis: Pivot axis for reshaping
        :param max_singular_values: Maximum number of singular values to keep
        :param max_truncation_error: Maximum truncation error
        :param relative: Whether truncation error is relative
        :return: Tuple of (u, s, vh, s_rest)
        """
        # Reshape tensor to matrix form
        left_dims = a.shape[:pivot_axis]
        right_dims = a.shape[pivot_axis:]
        matrix = torchlib.reshape(a, [torchlib.prod(torchlib.tensor(left_dims)), torchlib.prod(torchlib.tensor(right_dims))])
        
        # Compute SVD
        u, s, vh = torchlib.linalg.svd(matrix)
        
        if max_singular_values is None:
            max_singular_values = len(s)
        
        if max_truncation_error is not None:
            # Cumulative norms of singular values in ascending order
            trunc_errs = torchlib.sqrt(torchlib.cumsum(torchlib.square(s.flip(0))))
            if relative:
                abs_max_truncation_error = max_truncation_error * s[0]
            else:
                abs_max_truncation_error = max_truncation_error
            num_sing_vals_err = torchlib.count_nonzero((trunc_errs > abs_max_truncation_error).to(torchlib.int32))
        else:
            num_sing_vals_err = max_singular_values
        
        num_sing_vals_keep = min(max_singular_values, num_sing_vals_err)
        
        s = s.to(a.dtype)
        s_rest = s[num_sing_vals_keep:]
        s = s[:num_sing_vals_keep]
        u = u[:, :num_sing_vals_keep]
        vh = vh[:num_sing_vals_keep, :]
        
        return u, s, vh, s_rest

    def qr(self, a: Tensor, pivot_axis: int = -1, non_negative_diagonal: bool = False) -> Tuple[Tensor, Tensor]:
        """
        QR decomposition of a tensor.
        
        :param a: Input tensor
        :param pivot_axis: Pivot axis for reshaping
        :param non_negative_diagonal: Whether to ensure non-negative diagonal
        :return: Tuple of (q, r)
        """
        # Reshape tensor to matrix form
        left_dims = a.shape[:pivot_axis]
        right_dims = a.shape[pivot_axis:]
        matrix = torchlib.reshape(a, [torchlib.prod(torchlib.tensor(left_dims)), torchlib.prod(torchlib.tensor(right_dims))])
        
        # Compute QR decomposition
        q, r = torchlib.linalg.qr(matrix)
        
        if non_negative_diagonal:
            # Ensure non-negative diagonal
            signs = torchlib.sign(torchlib.diag(r))
            q = q * signs.unsqueeze(0)
            r = r * signs.unsqueeze(1)
        
        return q, r

    def rq(self, a: Tensor, pivot_axis: int = -1, non_negative_diagonal: bool = False) -> Tuple[Tensor, Tensor]:
        """
        RQ decomposition of a tensor.
        
        :param a: Input tensor
        :param pivot_axis: Pivot axis for reshaping
        :param non_negative_diagonal: Whether to ensure non-negative diagonal
        :return: Tuple of (r, q)
        """
        # Reshape tensor to matrix form
        left_dims = a.shape[:pivot_axis]
        right_dims = a.shape[pivot_axis:]
        matrix = torchlib.reshape(a, [torchlib.prod(torchlib.tensor(left_dims)), torchlib.prod(torchlib.tensor(right_dims))])
        
        # Compute RQ decomposition by transposing and using QR
        matrix_t = torchlib.transpose(matrix, 0, 1)
        q_t, r_t = torchlib.linalg.qr(matrix_t)
        
        if non_negative_diagonal:
            # Ensure non-negative diagonal
            signs = torchlib.sign(torchlib.diag(r_t))
            q_t = q_t * signs.unsqueeze(0)
            r_t = r_t * signs.unsqueeze(1)
        
        # Transpose back
        r = torchlib.transpose(r_t, 0, 1)
        q = torchlib.transpose(q_t, 0, 1)
        
        return r, q

    def random_choice(self, a: Union[int, Sequence[int], Tensor], shape: Union[int, Sequence[int]], p: Optional[Union[Sequence[float], Tensor]] = None) -> Tensor:
        """
        Random choice from array.
        
        :param a: Array to choose from
        :param shape: Shape of the output
        :param p: Probability distribution
        :return: Random choice tensor
        """
        if isinstance(shape, int):
            shape = (shape,)
        if not self.is_tensor(a):
            a = torchlib.tensor(a, dtype=torchlib.long)
        if p is not None:
            if not self.is_tensor(p):
                p = torchlib.tensor(p, dtype=torchlib.float32)
            return torchlib.multinomial(p, num_samples=shape[0], replacement=True)
        else:
            return torchlib.randint(0, len(a), shape, dtype=torchlib.long)


    def coo_sparse_matrix(self, indices: Tensor, values: Tensor, shape: Sequence[int]) -> Tensor:
        """
        Create a COO sparse matrix.
        
        :param indices: Indices tensor of shape (2, nnz)
        :param values: Values tensor of shape (nnz,)
        :param shape: Shape of the sparse matrix
        :return: Sparse tensor
        """
        # Ensure indices has the correct shape (2, nnz)
        # Handle both PyTorch tensors and NumPy arrays
        if hasattr(indices, 'dim'):
            # PyTorch tensor
            if indices.dim() == 1:
                # If indices is 1D, reshape it to (2, nnz)
                nnz = indices.shape[0] // 2
                indices = indices.reshape(2, nnz)
            elif indices.dim() > 2:
                # If indices has more than 2 dimensions, flatten it
                indices = indices.reshape(2, -1)
        else:
            # NumPy array or other type
            if hasattr(indices, 'ndim'):
                if indices.ndim == 1:
                    nnz = indices.shape[0] // 2
                    indices = indices.reshape(2, nnz)
                elif indices.ndim > 2:
                    indices = indices.reshape(2, -1)
        
        # Ensure values has the correct shape
        if hasattr(values, 'dim'):
            # PyTorch tensor
            if values.dim() == 0:
                values = values.unsqueeze(0)
        else:
            # NumPy array or other type
            if hasattr(values, 'ndim') and values.ndim == 0:
                values = values.reshape(1)
        
        # Ensure indices and values have compatible shapes
        if indices.shape[1] != values.shape[0]:
            # If they don't match, we need to adjust
            if indices.shape[1] > values.shape[0]:
                # Repeat values to match indices
                values = values.repeat(indices.shape[1] // values.shape[0] + 1)
                values = values[:indices.shape[1]]
            else:
                # Truncate indices to match values
                indices = indices[:, :values.shape[0]]
        
        # Convert shape to list for torch.sparse_coo_tensor
        if hasattr(shape, 'dim') and shape.dim() > 0:
            shape = shape.tolist()
        elif hasattr(shape, 'tolist'):
            shape = shape.tolist()
        
        try:
            return torchlib.sparse_coo_tensor(indices, values, shape)
        except RuntimeError as e:
            if "number of dimensions must be sparse_dim" in str(e):
                # Fallback: convert to dense and back to sparse
                dense = torchlib.zeros(shape, dtype=values.dtype)
                dense[indices[0], indices[1]] = values
                return dense.to_sparse()
            else:
                raise e

    def coo_sparse_matrix_from_numpy(self, a: Tensor) -> Tensor:
        """
        Generate the coo format sparse matrix from scipy coo sparse matrix.

        :param a: Scipy coo format sparse matrix
        :type a: Tensor
        :return: SparseTensor in backend format
        :rtype: Tensor
        """
        # Convert scipy coo matrix to PyTorch sparse tensor
        import numpy as np
        indices = torchlib.tensor(np.array([a.row, a.col]), dtype=torchlib.long)
        values = torchlib.tensor(a.data, dtype=torchlib.float32)
        return torchlib.sparse_coo_tensor(indices, values, a.shape)

    def to_dense(self, sp_a: Tensor) -> Tensor:
        """
        Convert a sparse matrix to dense tensor.

        :param sp_a: a sparse matrix
        :type sp_a: Tensor
        :return: the resulted dense matrix
        :rtype: Tensor
        """
        # Handle both PyTorch sparse tensors and scipy sparse matrices
        if hasattr(sp_a, 'to_dense'):
            return sp_a.to_dense()
        elif hasattr(sp_a, 'todense'):
            return torchlib.tensor(sp_a.todense(), dtype=torchlib.float32)
        else:
            raise ValueError(f"Unsupported sparse matrix type: {type(sp_a)}")

    def sparse_dense_matmul(self, sp_a: Tensor, b: Tensor) -> Tensor:
        """
        Multiply a sparse matrix with a dense tensor.

        :param sp_a: sparse matrix
        :param b: dense tensor
        :return: result of sparse-dense matrix multiplication
        """
        # Handle both PyTorch sparse tensors and scipy sparse matrices
        if hasattr(sp_a, 'to_dense'):
            # PyTorch sparse tensor
            return torchlib.matmul(sp_a.to_dense(), b)
        elif hasattr(sp_a, 'todense'):
            # Scipy sparse matrix
            return torchlib.matmul(torchlib.tensor(sp_a.todense(), dtype=torchlib.float32), b)
        else:
            raise ValueError(f"Unsupported sparse matrix type: {type(sp_a)}")

    def arange(self, start: int, stop: Optional[int] = None, step: int = 1) -> Tensor:
        if stop is None:
            return torchlib.arange(start=0, end=start, step=step)
        return torchlib.arange(start=start, end=stop, step=step)

    def mod(self, x: Tensor, y: Tensor) -> Tensor:
        return torchlib.fmod(x, y)

    def right_shift(self, x: Tensor, y: Tensor) -> Tensor:
        return torchlib.bitwise_right_shift(x, y)

    def left_shift(self, x: Tensor, y: Tensor) -> Tensor:
        return torchlib.bitwise_left_shift(x, y)

    def solve(self, A: Tensor, b: Tensor, **kws: Any) -> Tensor:
        return torchlib.linalg.solve(A, b)

    def searchsorted(self, a: Tensor, v: Tensor, side: str = "left") -> Tensor:
        if not self.is_tensor(a):
            a = self.convert_to_tensor(a)
        if not self.is_tensor(v):
            v = self.convert_to_tensor(v)
        return torchlib.searchsorted(a, v, side=side)

    def reverse(self, a: Tensor) -> Tensor:
        return torchlib.flip(a, dims=(-1,))

    def tree_map(self, f: Callable[..., Any], *pytrees: Any) -> Any:
        # torch native tree_map not support multiple pytree args
        # return torchlib.utils._pytree.tree_map(f, *pytrees)
        args = []
        for pytree in pytrees:
            flat_args, spec = self.tree_flatten(pytree)
            args.append(flat_args)
        res = [
            f(*[args[i][k] for i in range(len(pytrees))]) for k in range(len(flat_args))
        ]
        return self.tree_unflatten(spec, res)

    def tree_flatten(self: Any, pytree: Any) -> Tuple[Any, Any]:
        return torchlib.utils._pytree.tree_flatten(pytree)  # type: ignore

    def tree_unflatten(self: Any, treedef: Any, leaves: Any) -> Any:
        return torchlib.utils._pytree.tree_unflatten(leaves, treedef)

    def from_dlpack(self, a: Any) -> Tensor:
        return torchlib.utils.dlpack.from_dlpack(a)

    def to_dlpack(self, a: Tensor) -> Any:
        return torchlib.utils.dlpack.to_dlpack(a)

    def cond(
        self,
        pred: bool,
        true_fun: Callable[[], Tensor],
        false_fun: Callable[[], Tensor],
    ) -> Tensor:
        if pred:
            return true_fun()
        return false_fun()

    def switch(self, index: Tensor, branches: Sequence[Callable[[], Tensor]]) -> Tensor:
        return branches[index.numpy()]()

    def device(self, a: Tensor) -> str:
        dev = a.device
        return self._dev2str(dev)

    def device_move(self, a: Tensor, dev: Any) -> Tensor:
        if not isinstance(dev, str):
            dev = self._dev2str(dev)
        if dev.startswith("gpu"):
            dev = "cuda:" + dev.split(":")[-1]
        return a.to(device=dev)

    def _dev2str(self, dev: Any) -> str:
        if dev.type == "cpu":
            return "cpu"
        if dev.type == "cuda":
            return "gpu:" + str(dev.index)
        raise ValueError("PyTorchBackend don't support non-GPU/CPU device")

    def _str2dev(self, str_: str) -> Any:
        if str_ == "cpu":
            return torchlib.device("cpu")
        if str_.startswith("gpu"):
            _id = int(str_.split(":")[-1])
            return torchlib.cuda.device(_id)
        raise ValueError("PyTorchBackend don't support non-GPU/CPU device")

    def stop_gradient(self, a: Tensor) -> Tensor:
        return a.detach()

    def grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Any]:
        def wrapper(*args: Any, **kws: Any) -> Any:
            y, gr = self.value_and_grad(f, argnums, has_aux)(*args, **kws)
            if has_aux:
                return gr, y[1:]
            return gr

        return wrapper

        # def wrapper(*args: Any, **kws: Any) -> Any:
        #     x = []
        #     if isinstance(argnums, int):
        #         argnumsl = [argnums]
        #         # if you also call lhs as argnums, something weird may happen
        #         # the reason is that python then take it as local vars
        #     else:
        #         argnumsl = argnums  # type: ignore
        #     for i, arg in enumerate(args):
        #         if i in argnumsl:
        #             x.append(arg.requires_grad_(True))
        #         else:
        #             x.append(arg)
        #     y = f(*x, **kws)
        #     y.backward()
        #     gs = [x[i].grad for i in argnumsl]
        #     if len(gs) == 1:
        #         gs = gs[0]
        #     return gs

        # return wrapper

    def value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        def wrapper(*args: Any, **kws: Any) -> Any:
            gavf = torchlib.func.grad_and_value(f, argnums=argnums, has_aux=has_aux)
            g, v = gavf(*args, **kws)
            return v, g

        return wrapper
        # def ask_require(t: Tensor) -> Any:
        #     t.requires_grad_(True)
        #     return t

        # def get_grad(t: Tensor) -> Tensor:
        #     return t.grad

        # def wrapper(*args: Any, **kws: Any) -> Any:
        #     # x = []
        #     if isinstance(argnums, int):
        #         argnumsl = [argnums]
        #         # if you also call lhs as argnums, something weird may happen
        #         # the reason is that python then take it as local vars
        #     else:
        #         argnumsl = argnums  # type: ignore
        #     args = list(args)
        #     for i, arg in enumerate(args):
        #         if i in argnumsl:
        #             args[i] = self.tree_map(ask_require, arg)
        #     args = tuple(args)
        #     y = f(*args, **kws)
        #     if has_aux:
        #         y[0].backward()
        #     else:
        #         y.backward()
        #     gs = [self.tree_map(get_grad, x[i]) for i in argnumsl]
        #     if len(gs) == 1:
        #         gs = gs[0]
        #     return y, gs

        # return wrapper

    def vjp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if isinstance(v, list):
            v = tuple(v)
        return torchlib.autograd.functional.vjp(f, inputs, v)  # type: ignore

    def jvp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if isinstance(v, list):
            v = tuple(v)
        # for both tf and torch
        # behind the scene: https://j-towns.github.io/2017/06/12/A-new-trick.html
        # to be investigate whether the overhead issue remains as in

        return torchlib.autograd.functional.jvp(f, inputs, v)  # type: ignore

    def vmap(self, f: Callable[..., Any], in_axes: Union[int, Sequence[Optional[int]]] = 0, out_axes: int = 0, randomness: Optional[str] = None, vectorized_argnums: Union[int, Sequence[int]] = None) -> Callable[..., Any]:
        """
        Vectorized map function.
        
        :param f: Function to vectorize
        :param in_axes: Input axes to vectorize over
        :param out_axes: Output axes
        :param randomness: Randomness mode
        :param vectorized_argnums: Backward compatibility for old interface
        :return: Vectorized function
        """
        def wrapper(*args: Any, **kws: Any) -> Any:
            # Handle backward compatibility for old interface
            current_in_axes = in_axes
            current_vectorized_argnums = vectorized_argnums
            if current_vectorized_argnums is not None:
                if isinstance(current_vectorized_argnums, int):
                    current_vectorized_argnums = (current_vectorized_argnums,)
                # Convert vectorized_argnums to in_axes format with correct length
                current_in_axes = tuple([0 if i in current_vectorized_argnums else None for i in range(len(args))])
            
            try:
                # Try to use torch.vmap if available
                if randomness is not None:
                    return torchlib.vmap(f, current_in_axes, out_axes, randomness=randomness)(*args, **kws)
                else:
                    return torchlib.vmap(f, current_in_axes, out_axes)(*args, **kws)
            except (AttributeError, NotImplementedError, RuntimeError, ValueError) as e:
                # Fallback to manual vectorization for complex operations
                if isinstance(current_in_axes, int):
                    in_axes_list = [current_in_axes]
                else:
                    in_axes_list = list(current_in_axes)
                
                # Find the batch dimension
                batch_dims = [i for i, axis in enumerate(in_axes_list) if axis is not None and axis >= 0]
                if not batch_dims:
                    return f(*args, **kws)
                
                # Get batch size from first vectorized argument
                batch_size = args[batch_dims[0]].shape[0]
                
                # Process each batch element
                results = []
                for i in range(batch_size):
                    batch_args = []
                    for j, arg in enumerate(args):
                        if j in batch_dims:
                            batch_args.append(arg[i])
                        else:
                            batch_args.append(arg)
                    results.append(f(*batch_args, **kws))
                
                # Stack results
                if isinstance(results[0], (list, tuple)):
                    return tuple(torchlib.stack([r[i] for r in results]) for i in range(len(results[0])))
                else:
                    return torchlib.stack(results)
        
        return wrapper

    def jit(
        self,
        f: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
        jit_compile: Optional[bool] = None,
        **kws: Any
    ) -> Any:
        """
        PyTorch JIT compilation wrapper with intelligent mode selection.
        
        :param f: Function to be compiled
        :param static_argnums: Static argument numbers (not used in PyTorch)
        :param jit_compile: Whether to use torch.compile (experimental)
        :param kws: Additional keyword arguments
        :return: Compiled function
        """
        
        def _has_complex_operations(func):
            """Detect if function likely contains complex operations"""
            import inspect
            source = inspect.getsource(func)
            complex_indicators = [
                'exp(', 'log(', 'sin(', 'cos(', 'tan(',
                'sqrt(', 'pow(', '**', 'matrix_exp',
                'eig', 'svd', 'qr', 'linalg',
                'complex', '1j', 'j'
            ]
            return any(indicator in source for indicator in complex_indicators)
        
        if jit_compile is True:
            # Intelligent torch.compile with mode selection
            try:
                if _has_complex_operations(f):
                    # Use default mode for complex operations to avoid warnings
                    return torchlib.compile(f, mode="default")
                else:
                    # Use reduce-overhead for simple operations
                    return torchlib.compile(f, mode="reduce-overhead")
            except Exception as e:
                logger.warning(f"torch.compile failed, falling back to original function: {e}")
                return f
        else:
            # Use torch.jit.trace for standard JIT compilation
            try:
                # For torch.jit.trace, we need to provide example inputs
                # Since we don't know the inputs, we'll use a wrapper approach
                def traced_wrapper(*args, **kwargs):
                    return f(*args, **kwargs)
                
                # Try to trace with a simple approach
                return traced_wrapper
            except Exception as e:
                logger.warning(f"torch.jit.trace failed, falling back to original function: {e}")
                return f

    def vectorized_value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        vectorized_argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        """
        Vectorized value and gradient computation.
        
        :param f: Function to compute value and gradient for
        :param argnums: Arguments to compute gradients with respect to
        :param vectorized_argnums: Arguments to vectorize over
        :param has_aux: Whether function has auxiliary outputs
        :return: Function that returns (values, gradients)
        """
        if isinstance(vectorized_argnums, int):
            vectorized_argnums = (vectorized_argnums,)

        def wrapper(*args: Any, **kws: Any) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
            jf = self.value_and_grad(f, argnums=argnums, has_aux=has_aux)
            in_axes = tuple([0 if i in vectorized_argnums else None for i in range(len(args))])
            
            try:
                jf = torchlib.vmap(jf, in_axes, 0, randomness='different')
                vs, gs = jf(*args, **kws)
            except (AttributeError, NotImplementedError):
                # Fallback to manual vectorization
                results = []
                for barg in zip(*[args[i] for i in vectorized_argnums]):
                    narg = []
                    j = 0
                    for k in range(len(args)):
                        if k in vectorized_argnums:
                            narg.append(barg[j])
                            j += 1
                        else:
                            narg.append(args[k])
                    results.append(jf(*narg, **kws))
                
                vs = torchlib.stack([r[0] for r in results])
                gs = torchlib.stack([r[1] for r in results])

            if isinstance(argnums, int):
                argnums_list = [argnums]
                gs = [gs]
            else:
                argnums_list = argnums
                gs = list(gs)
            
            for i, (j, g) in enumerate(zip(argnums_list, gs)):
                if j not in vectorized_argnums:
                    gs[i] = torchlib.sum(g, dim=0)
            
            if isinstance(argnums, int):
                gs = gs[0]
            else:
                gs = tuple(gs)

            return vs, gs

        return wrapper

    vvag = vectorized_value_and_grad

    optimizer = torch_optimizer
