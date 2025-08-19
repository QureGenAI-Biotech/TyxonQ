"""
reducing jit compiling time by general scan magic
"""

import numpy as np
import tyxonq as tq

n = 10
nlayers = 16
param_np = np.random.normal(size=[nlayers, n, 2])

for backend in ["pytorch"]:
    with tq.runtime_backend(backend) as K:
        print("running %s" % K.name)

        def energy_reference(param, n, nlayers):
            c = tq.Circuit(n)
            for i in range(n):
                c.h(i)
            for i in range(nlayers):
                for j in range(n - 1):
                    c.rzz(j, j + 1, theta=param[i, j, 0])
                for j in range(n):
                    c.rx(j, theta=param[i, j, 1])
            return K.real(c.expectation_ps(z=[0, 1]) + c.expectation_ps(x=[2]))

        # warning pytorch might be unable to do this exactly
        vg_reference = K.jit(
            K.value_and_grad(energy_reference, argnums=0), static_argnums=(1, 2)
        )

        # a jit efficient way to utilize scan

        def energy(param, n, nlayers, each):
            def loop_f(s_, param_):
                c_ = tq.Circuit(n, inputs=s_)
                for i in range(each):
                    for j in range(n - 1):
                        c_.rzz(j, j + 1, theta=param_[i, j, 0])
                    for j in range(n):
                        c_.rx(j, theta=param_[i, j, 1])
                s_ = c_.state()
                return s_

            c = tq.Circuit(n)
            for i in range(n):
                c.h(i)
            s = c.state()
            # warning pytorch might be unable to do this exactly
            s1 = K.scan(loop_f, K.reshape(param, [nlayers // each, each, n, 2]), s)
            c1 = tq.Circuit(n, inputs=s1)
            return K.real(c1.expectation_ps(z=[0, 1]) + c1.expectation_ps(x=[2]))

        # warning pytorch might be unable to do this exactly
        vg = K.jit(
            K.value_and_grad(energy, argnums=0),
            static_argnums=(1, 2, 3),
            jit_compile=True,
        )
        # set to False can improve compile time for pytorch

        param = K.convert_to_tensor(param_np)

        for each in [1, 2, 4]:
            print("  scan impl with each=%s" % str(each))
            r1 = tq.utils.benchmark(vg, param, n, nlayers, each)
            print(r1[0][0])

        print("  plain impl")
        r0 = tq.utils.benchmark(vg_reference, param, n, nlayers)  # too slow
        np.testing.assert_allclose(r0[0][0].detach().cpu().numpy(), r1[0][0].detach().cpu().numpy(), atol=1e-5)
        np.testing.assert_allclose(r0[0][1].detach().cpu().numpy(), r1[0][1].detach().cpu().numpy(), atol=1e-5)
        # correctness check


# jit_compile=True icrease runtime while degrades jit time for pytorch
# and in general pytorch improves better with scan methodology,
# both compile time and running time can outperform other backends
