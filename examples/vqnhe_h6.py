"""
H6 molecule VQNHE with code from applications
"""

import sys

sys.path.insert(0, "../")
import numpy as np
import tyxonq as tq
import os
from tyxonq.applications.vqes import VQNHE, JointSchedule, construct_matrix_v3

K = tq.set_backend("pytorch")
K.set_dtype("complex64")

h6h = np.load(os.path.join(os.path.dirname(__file__), "h6_hamiltonian.npy"))  # robust path
hamiltonian = construct_matrix_v3(h6h.tolist())
# Densify to avoid sparse+functorch kernel issues during optimization
hamiltonian = K.to_dense(hamiltonian)
# Infer qubit count from term length in the data
n = int(len(h6h[0]) - 1)


vqeinstance = VQNHE(
    n,
    hamiltonian,
    {"width": 16, "stddev": 0.001, "choose": "complex-rbm"},  # model parameter
    {"filled_qubit": [i for i in [0, 1, 3, 4, 5, 6, 8, 9] if i < n], "epochs": 1},  # circuit parameter
    shortcut=True,  # enable shortcut for full Hamiltonian matrix evaluation
)
# 1110011100

rs = vqeinstance.multi_training(
    tries=1,
    maxiter=50,
    threshold=0.5e-8,
    optq=JointSchedule(50, 0.01, 200, 0.002, 200),
    optc=JointSchedule(50, 0.0006, 200, 0.008, 200),
    onlyq=0,
    debug=200,
    checkpoints=[],
)
print(rs)
