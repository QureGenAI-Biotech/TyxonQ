"""
example showcasing how circuit can be load from and dump to json:
useful for storage or restful api
"""

import numpy as np
import tyxonq as tq

tq.set_dtype("complex128")


def make_circuit():
    c = tq.Circuit(3)
    c.h(0)
    c.h(2)
    c.cnot(1, 2)
    c.rxx(0, 2, theta=0.3)
    c.u(2, theta=0.2, lbd=-1.2, phi=0.5)
    c.cu(1, 0, lbd=1.0)
    c.crx(0, 1, theta=-0.8)
    c.r(1, theta=tq.backend.ones([]), alpha=0.2)
    c.toffoli(0, 2, 1)
    c.ccnot(0, 1, 2)
    c.any(0, 1, unitary=tq.gates._xx_matrix)
    c.multicontrol(1, 2, 0, ctrl=[0, 1], unitary=tq.gates._x_matrix)
    return c


if __name__ == "__main__":
    c = make_circuit()
    s = c.to_json(simplified=True)
    print(s)
    c.to_json(file="circuit.json")
    # load from json string
    c2 = tq.Circuit.from_json(s)
    print("\n", c2.draw())
    s1 = tq.backend.numpy(c.state())
    s2 = tq.backend.numpy(c2.state())
    np.testing.assert_allclose(s1, s2, atol=1e-5)
    print("test correctness 1")
    # load from json file
    c3 = tq.Circuit.from_json_file("circuit.json")
    print("\n", c3.draw())
    s3 = tq.backend.numpy(c3.state())
    np.testing.assert_allclose(s1, s3, atol=1e-5)
    print("test correctness 2")
