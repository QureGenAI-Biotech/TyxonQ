================
Advanced Usage
================

MPS Simulator
----------------

(Still experimental support)

Very simple, we provide the same set of API for ``MPSCircuit`` as ``Circuit``, 
the only new line is to set the bond dimension for the new simulator.

.. code-block:: python

    c = tq.MPSCircuit(n)
    c.set_split_rules({"max_singular_values": 50})

The larger bond dimension we set, the better approximation ratio (of course the more computational cost we pay)

Split Two-qubit Gates
-------------------------

The two-qubit gates applied on the circuit can be decomposed via SVD, which may further improve the optimality of the contraction pathfinding.

`split` configuration can be set at circuit-level or gate-level.

.. code-block:: python

    split_conf = {
        "max_singular_values": 2,  # how many singular values are kept
        "fixed_choice": 1, # 1 for normal one, 2 for swapped one
    }

    c = tq.Circuit(nwires, split=split_conf)

    # or

    c.exp1(
            i,
            (i + 1) % nwires,
            theta=paramc[2 * j, i],
            unitary=tq.gates._zz_matrix,
            split=split_conf
        )

Note ``max_singular_values`` must be specified to make the whole procedure static and thus jittable.


Jitted Function Save/Load
-----------------------------

To reuse the jitted function, we can save it on the disk via support from PyTorch's `TorchScript <https://pytorch.org/docs/stable/jit.html>`_.

We provide easy-to-use functions :py:meth:`tyxonq.torchnn.save_func` and :py:meth:`tyxonq.torchnn.load_func`.

Parameterized Measurements
-----------------------------

For plain measurements API on a ``tq.Circuit``, eg. `c = tq.Circuit(n=3)`, if we want to evaluate the expectation :math:`<Z_1Z_2>`, we need to call the API as ``c.expectation((tq.gates.z(), [1]), (tq.gates.z(), [2]))``.

In some cases, we may want to tell the software what to measure but in a tensor fashion. For example, if we want to get the above expectation, we can use the following API: :py:meth:`tyxonq.templates.measurements.parameterized_measurements`.

.. code-block:: python

    c = tq.Circuit(3)
    z1z2 = tq.templates.measurements.parameterized_measurements(c, tq.array_to_tensor([0, 3, 3, 0]), onehot=True) # 1

This API corresponds to measure :math:`I_0Z_1Z_2I_3` where 0, 1, 2, 3 are for local I, X, Y, and Z operators respectively.

Sparse Matrix
----------------

We support COO format sparse matrix as most backends only support this format, and some common backend methods for sparse matrices are listed below:

.. code-block:: python

    def sparse_test():
        m = tq.backend.coo_sparse_matrix(indices=np.array([[0, 1],[1, 0]]), values=np.array([1.0, 1.0]), shape=[2, 2])
        n = tq.backend.convert_to_tensor(np.array([[1.0], [0.0]]))
        print("is sparse: ", tq.backend.is_sparse(m), tq.backend.is_sparse(n))
        print("sparse matmul: ", tq.backend.sparse_dense_matmul(m, n))

    for K in ["pytorch", "numpy"]:
        with tq.runtime_backend(K):
            print("using backend: ", K)
            sparse_test()

The sparse matrix is specifically useful to evaluate Hamiltonian expectation on the circuit, where sparse matrix representation has a good tradeoff between space and time.
Please refer to :py:meth:`tyxonq.templates.measurements.sparse_expectation` for more detail.

For different representations to evaluate Hamiltonian expectation in tyxonq, please refer to :doc:`tutorials/tfim_vqe_diffreph`.

Randomness, Jit, and Their Interplay
--------------------------------------------------------

The interplay between randomness and JIT compilation requires careful handling, especially when aiming for reproducibility. PyTorch uses a stateful pseudo-random number generator (PRNG). To ensure reproducibility in a JIT-compiled function, the random state must be managed explicitly.

.. code-block:: python

    import tyxonq as tq
    import torch
    K = tq.set_backend("pytorch")

    @K.jit
    def r(generator):
        return torch.randn(1, generator=generator)

    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    print(r(g1), r(g1)) # same, correct
    print(r(g2)) # same as first call, correct

To get different random numbers, you must use different generator states.

.. code-block:: python

    g = torch.Generator().manual_seed(42)
    print(r(g), r(g))  # Two calls with the same generator will produce the same result if the function is jitted
    
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(43)
    print(r(g1), r(g2)) # different, correct

TyxonQ's backend provides helper functions to manage this. ``K.get_random_state`` will return a `torch.Generator` instance, and ``K.random_split`` can be used to create new independent generator objects.

.. code-block:: python

    key = K.get_random_state(42)

    @K.jit
    def r(key):
        # We don't need K.set_random_state inside, as we pass the generator
        return K.implicit_randn(generator=key)

    key1, key2 = K.random_split(key)

    print(r(key1), r(key2))

This paradigm is crucial when using stochastic elements in your circuits, such as with ``Circuit.unitary_kraus`` and ``Circuit.general_kraus``, inside a JIT-compiled function.