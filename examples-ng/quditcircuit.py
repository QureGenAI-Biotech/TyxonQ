"""
Basic features of ``tq.Circuit`` class support qudits natively
"""

import numpy as np
import tyxonq as tq

K = tq.set_backend("pytorch")

n = 3

# d=3 qudits
ns = [tq.gates.Gate(np.array([1.0, 0.0, 0.0])) for _ in range(n)]
mps = tq.quantum.QuVector([nd[0] for nd in ns])

c = tq.Circuit(n, mps_inputs=mps)

ctrl1switch02 = np.kron(
    np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]), np.eye(3)
) + np.kron(
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
)
print("two-qudit gate: \n", ctrl1switch02)

# use unitary gate with ``Gate`` as input to avoid the matrix autoreshape to 2-base
c.unitary(0, unitary=tq.gates.Gate(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])))
c.unitary(0, 2, unitary=tq.gates.Gate(np.array(ctrl1switch02.reshape([3, 3, 3, 3]))))

print(c.state())
for i in range(n):
    print(i)
    print(
        c.expectation(
            [tq.gates.Gate(np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])), [i]]
        )
    )
