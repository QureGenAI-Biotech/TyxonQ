"""
Old-fashioned DQAS code on QAOA ansatz design, now deprecated.
"""

# pylint: disable=wildcard-import

import sys

sys.path.insert(0, "../")

from collections import namedtuple
from pickle import dump
from matplotlib import pyplot as plt
import numpy as np
import torch
import tyxonq as tq
from tyxonq.applications.dqas import *
from tyxonq.applications.vags import *
from tyxonq.applications.layers import *
from tyxonq.applications.graphdata import regular_graph_generator

# qaoa_block_vag_energy = partial(qaoa_block_vag, f=(_identity, _neg))

tq.set_backend("pytorch")


def main_layerwise_encoding():
    p = 5
    c = 7

    def noise():
        n = np.random.normal(loc=0.0, scale=0.002, size=[p, c])
        return torch.tensor(n, dtype=torch.float32)

    def penalty_gradient(stp, nnp, lbd=0.15, lbd2=0.01):
        c = stp.shape[1]
        p = stp.shape[0]
        cost = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 27 / 2 * 2.0, 27 / 2 * 2.0, 15 / 2.0], dtype=torch.float32
        )
        stp.requires_grad_(True)
        prob = torch.exp(stp) / torch.sum(torch.exp(stp), dim=1, keepdim=True)
        penalty = 0.0
        for i in range(p - 1):
            penalty += lbd * torch.tensordot(prob[i], prob[i + 1], dims=1)
            penalty += lbd2 * torch.tensordot(cost, prob[i], dims=1)
        penalty += lbd2 * torch.tensordot(cost, prob[p - 1], dims=1)
        penalty.backward()
        penalty_g = stp.grad.clone()
        stp.grad.zero_()
        return penalty_g

    # warning: pytorch scheduler is different from tensorflow
    learning_rate_sch = 0.12

    DQAS_search(
        qaoa_vag_energy,
        g=regular_graph_generator(n=8, d=3),
        p=p,
        batch=8,
        prethermal=0,
        prethermal_preset=[0, 6, 1, 6, 1],
        epochs=3,
        parallel_num=2,
        pertubation_func=noise,
        nnp_initial_value=np.random.normal(loc=0.23, scale=0.06, size=[p, c]),
        stp_regularization=penalty_gradient,
        network_opt=torch.optim.Adam([], lr=0.03),
        prethermal_opt=torch.optim.Adam([], lr=0.04),
        structure_opt=torch.optim.SGD([], lr=learning_rate_sch),
    )


result = namedtuple("result", ["epoch", "cand", "loss"])


def main_block_encoding():
    # warning: pytorch scheduler is different from tensorflow
    learning_rate_sch = 0.3
    p = 6
    c = 8

    def record():
        return result(
            get_var("epoch"), get_var("cand_preset_repr"), get_var("avcost1").detach().cpu().item()
        )

    def noise():
        # p = 6
        # c = 6
        n = np.random.normal(loc=0.0, scale=0.2, size=[2 * p, c])
        return torch.tensor(n, dtype=torch.float32)

    stp, nnp, h = DQAS_search(
        qaoa_block_vag_energy,
        g=regular_graph_generator(n=8, d=3),
        batch=8,
        prethermal=0,
        prethermal_preset=[0, 6, 1, 6, 1],
        epochs=3,
        parallel_num=2,
        pertubation_func=noise,
        p=p,
        history_func=record,
        nnp_initial_value=np.random.normal(loc=0.23, scale=0.06, size=[2 * p, c]),
        network_opt=torch.optim.Adam([], lr=0.04),
        prethermal_opt=torch.optim.Adam([], lr=0.04),
        structure_opt=torch.optim.SGD([], lr=learning_rate_sch),
    )

    with open("qaoa_block.result", "wb") as f:
        dump([stp.detach().cpu().numpy(), nnp.detach().cpu().numpy(), h], f)

    epochs = np.arange(len(h))
    data = np.array([r.loss for r in h])
    plt.plot(epochs, data)
    plt.xlabel("epoch")
    plt.ylabel("objective")
    plt.savefig("qaoa_block.pdf")


layer_pool = [Hlayer, rxlayer, rylayer, rzlayer, xxlayer, yylayer, zzlayer]
block_pool = [
    Hlayer,
    rx_zz_block,
    zz_ry_block,
    zz_rx_block,
    zz_rz_block,
    xx_rz_block,
    yy_rx_block,
    rx_rz_block,
]
set_op_pool(block_pool)

if __name__ == "__main__":
    # main_layerwise_encoding()
    main_block_encoding()
