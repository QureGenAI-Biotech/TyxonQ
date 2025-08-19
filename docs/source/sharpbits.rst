=================================
TyxonQ: The Sharp Bits 🔪
=================================

Being fast is never for free, though much cheaper in TyxonQ, but you have to be cautious especially in terms of AD, JIT compatibility.
We will go through the main sharp edges 🔪 in this note.

Jit Compatibility
---------------------

Non-tensor input or varying shape tensor input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The input must be in tensor form and the input tensor shape must be fixed otherwise the recompilation is incurred which is time-consuming.
Therefore, if there are input args that are non-tensor or varying shape tensors and frequently change, jit is not recommend.

.. code-block:: python

    K = tq.set_backend("pytorch")

    @K.jit
    def f(a):
        print("compiling")
        return 2*a

    f(K.ones([2]))
    # compiling
    # tensor([2.+0.j, 2.+0.j], dtype=torch.complex64)

    f(K.zeros([2]))
    # tensor([0.+0.j, 0.+0.j], dtype=torch.complex64)

    f(K.ones([3]))
    # compiling
    # tensor([2.+0.j, 2.+0.j, 2.+0.j], dtype=torch.complex64)

Mix use of numpy and ML backend APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To make the function jittable and ad-aware, every ops in the function should be called via ML backend (``tq.backend`` API or direct API for PyTorch ``torch``).
This is because the ML backend has to create the computational graph to carry out AD and JIT transformation. For numpy ops, they will be only called in jit staging time (the first run).

.. code-block:: python

    K = tq.set_backend("pytorch")

    @K.jit
    def f(a):
        return np.dot(a, a)

    f(K.ones([2]))
    # TypeError: an unsupported object was captured by the tracer

Numpy call inside jitted function can be helpful if you are sure of the behavior is what you expect.

.. code-block:: python

    K = tq.set_backend("pytorch")

    @K.jit
    def f(a):
        print("compiling")
        n = a.shape[0]
        m = int(np.log(n)/np.log(2))
        return K.reshape(a, [2 for _ in range(m)])

    f(K.ones([4]))
    # compiling
    # tensor([[1.+0.j, 1.+0.j],
    #        [1.+0.j, 1.+0.j]], dtype=torch.complex64)

    f(K.zeros([4]))
    # tensor([[0.+0.j, 0.+0.j],
    #        [0.+0.j, 0.+0.j]], dtype=torch.complex64)

    f(K.zeros([2]))
    # compiling
    # tensor([0.+0.j, 0.+0.j], dtype=torch.complex64)

list append under if
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Appending something to a Python list within an if whose condition is based on tensor values will lead to issues with JIT compilation in PyTorch. The tracer will attempt to evaluate both branches, leading to unexpected behavior.

.. code-block:: python

    K = tq.set_backend("pytorch")

    @K.jit
    def f(a):
        l = []
        one = K.ones([])
        zero = K.zeros([])
        if a > 0:
            l.append(one)
        else:
            l.append(zero)
        return l

    f(-K.ones([], dtype="float32"))
    # The tracer will likely raise an error here.

Similarly, conditional gate application must be takend carefully.

.. code-block:: python

    K = tq.set_backend("pytorch")

    # The correct implementation is

    @K.jit
    def f():
        c = tq.Circuit(1)
        c.h(0)
        a = c.cond_measure(0)
        c.conditional_gate(a, [tq.gates.z(), tq.gates.x()], 0)
        return c.state()

    f()
    # tensor([1.+0.j, 0.+0.j], dtype=torch.complex64)


Tensor variables consistency
-------------------------------------------------------


All tensor variables' backend (torch vs numpy vs ..), dtype (float vs complex), shape and device (cpu vs gpu) must be compatible/consistent.

Inspect the backend, dtype, shape and device using the following codes.

.. code-block:: python

    for backend in ["numpy", "pytorch"]:
        with tq.runtime_backend(backend):
            a = tq.backend.ones([2, 3])
            print("tensor backend:", tq.interfaces.which_backend(a))
            print("tensor dtype:", tq.backend.dtype(a))
            print("tensor shape:", tq.backend.shape_tuple(a))
            print("tensor device:", tq.backend.device(a))

If the backend is inconsistent, one can convert the tensor backend via :py:meth:`tyxonq.interfaces.tensortrans.general_args_to_backend`.

.. code-block:: python

    for backend in ["numpy", "pytorch"]:
        with tq.runtime_backend(backend):
            a = tq.backend.ones([2, 3])
            print("tensor backend:", tq.interfaces.which_backend(a))
            b = tq.interfaces.general_args_to_backend(a, target_backend="pytorch", enable_dlpack=False)
            print("tensor backend:", tq.interfaces.which_backend(b))

If the dtype is inconsistent, one can convert the tensor dtype using ``tq.backend.cast``.

.. code-block:: python

    for backend in ["numpy", "pytorch"]:
        with tq.runtime_backend(backend):
            a = tq.backend.ones([2, 3])
            print("tensor dtype:", tq.backend.dtype(a))
            b = tq.backend.cast(a, dtype="float64")
            print("tensor dtype:", tq.backend.dtype(b))


If the shape is not consistent, one can convert the shape by ``tq.backend.reshape``.

If the device is not consistent, one can move the tensor between devices by ``tq.backend.device_move``.


AD Consistency
---------------------

In TyxonQ, the AD behavior for complex-valued functions is determined by the underlying PyTorch backend. PyTorch's automatic differentiation for complex numbers follows the principles outlined in the paper "Analytic derivatives of complex-valued functions". All AD relevant ops such as ``grad`` or ``jacrev`` may be affected. Therefore, the user must be careful when dealing with AD on complex valued function in a backend agnostic way in TyxonQ.

See example script on computing Jacobian with different modes: `jacobian_cal.py <https://github.com/QureGenAI-Biotech/TyxonQ/blob/master/examples/jacobian_cal.py>`_.
Also see the code below for a reference:

.. code-block:: python

    import torch

    bks = ["pytorch"]
    n = 2
    for bk in bks:
        print(bk, "backend")
        with tq.runtime_backend(bk) as K:
            def wfn(params):
                c = tq.Circuit(n)
                for i in range(n):
                    c.H(i)
                for i in range(n):
                    c.rz(i, theta=params[i])
                    c.rx(i, theta=params[i])
                return K.real(c.expectation_ps(z=[0])+c.expectation_ps(z=[1]))
            
            params_c = K.ones([n], dtype="complex64")
            params_c.requires_grad_()
            loss_c = wfn(params_c)
            loss_c.backward()
            print(params_c.grad)

            params_f = K.ones([n], dtype="float32")
            params_f.requires_grad_()
            loss_f = wfn(params_f)
            loss_f.backward()
            print(params_f.grad)

    # pytorch backend
    # tensor([0.9093+0.j, 0.9093+0.j], dtype=torch.complex64)
    # tensor([0.9093, 0.9093])