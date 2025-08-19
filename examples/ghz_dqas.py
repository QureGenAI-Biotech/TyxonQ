"""
DQAS for GHZ state preparation circuit, deprecated DQAS implementation
"""

import sys

sys.path.insert(0, "../")
import numpy as np
import torch
import cirq

import tyxonq as tq
from tyxonq.applications.vags import double_qubits_initial, GHZ_vag, GHZ_vag_tfq
from tyxonq.applications.dqas import (
    set_op_pool,
    get_preset,
    DQAS_search,
)

tq.set_backend("pytorch")


def main_tn():
    """
    DQAS with the tyxonq engine backend by TensorNetwork
    state preparation example

    :return:
    """
    # multi start may be necessary
    ghz_pool = [
        ("ry", 0),
        ("ry", 1),
        ("ry", 2),
        ("cnot", 0, 1),
        ("cnot", 1, 0),
        ("cnot", 0, 2),
        ("cnot", 2, 0),
        ("h", 0),
        ("h", 1),
        ("h", 2),
    ]
    set_op_pool(ghz_pool)
    c = len(ghz_pool)
    p = 4
    stp, nnp, _ = DQAS_search(
        GHZ_vag,
        p=p,
        batch=128,
        epochs=20,
        verbose=True,
        parallel_num=0,
        nnp_initial_value=np.zeros([p, c]),
        structure_opt=torch.optim.Adam([torch.nn.Parameter(torch.zeros([p, c]))], lr=0.15),
    )
    preset = get_preset(stp).detach().cpu().numpy()
    GHZ_vag(None, nnp, preset, verbose=True)


def main_tfq():
    """
    DQAS with the pytorch quantum engine.
    Unitary learning example.

    :return:
    """
    p = 4
    cset = [
        cirq.H(cirq.GridQubit(0, 0)),
        cirq.H(cirq.GridQubit(1, 0)),
        cirq.CNOT(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
        cirq.CNOT(cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)),
        cirq.X(cirq.GridQubit(0, 0)),
        cirq.X(cirq.GridQubit(1, 0)),
    ]
    set_op_pool(cset)
    c = len(cset)
    stp, nnp, _ = DQAS_search(
        GHZ_vag_tfq,
        g=double_qubits_initial(),
        p=p,
        batch=16,
        epochs=5,
        verbose=False,
        parallel_num=0,
        nnp_initial_value=np.zeros([p, c]),
        structure_opt=torch.optim.Adam([torch.nn.Parameter(torch.zeros([p, c]))], lr=0.15),
    )
    preset = get_preset(stp).detach().cpu().numpy()
    GHZ_vag_tfq(double_qubits_initial().send(None), nnp, preset, verbose=True)


if __name__ == "__main__":
    main_tfq()
    # main_tn()
