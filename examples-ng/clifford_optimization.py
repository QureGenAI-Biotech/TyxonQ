"""
DQAS-style optimization for discrete Clifford type circuit
"""

import numpy as np
import torch

import tyxonq as tq

ctype, rtype = tq.set_dtype("complex64")
K = tq.set_backend("pytorch")

n = 6
nlayers = 6


def ansatz(structureo, structuret, preprocess="direct"):
    c = tq.Circuit(n)
    if preprocess == "softmax":
        structureo = K.softmax(structureo, axis=-1)
        structuret = K.softmax(structuret, axis=-1)
    elif preprocess == "most":
        structureo = K.onehot(K.argmax(structureo, axis=-1), num=7)
        structuret = K.onehot(K.argmax(structuret, axis=-1), num=3)
    elif preprocess == "direct":
        pass

    structureo = K.cast(structureo, ctype)
    structuret = K.cast(structuret, ctype)

    structureo = torch.reshape(structureo, shape=[nlayers, n, 7])
    structuret = torch.reshape(structuret, shape=[nlayers, n, 3])

    for i in range(n):
        c.h(i)
    for j in range(nlayers):
        for i in range(n):
            c.unitary(
                i,
                unitary=structureo[j, i, 0] * tq.gates.i().tensor
                + structureo[j, i, 1] * tq.gates.x().tensor
                + structureo[j, i, 2] * tq.gates.y().tensor
                + structureo[j, i, 3] * tq.gates.z().tensor
                + structureo[j, i, 4] * tq.gates.h().tensor
                + structureo[j, i, 5] * tq.gates.s().tensor
                + structureo[j, i, 6] * tq.gates.sd().tensor,
            )
        for i in range(n - 1):
            c.unitary(
                i,
                i + 1,
                unitary=structuret[j, i, 0] * tq.gates.ii().tensor
                + structuret[j, i, 1] * tq.gates.cnot().tensor
                + structuret[j, i, 2] * tq.gates.cz().tensor,
            )
    # loss = K.real(
    #     sum(
    #         [c.expectation_ps(z=[i, i + 1]) for i in range(n - 1)]
    #         + [c.expectation_ps(x=[i]) for i in range(n)]
    #     )
    # )
    s = c.state()
    loss = -K.real(tq.quantum.entropy(tq.quantum.reduced_density_matrix(s, cut=n // 2)))
    return loss


def sampling_from_structure(structures, batch=1):
    ch = structures.shape[-1]
    prob = K.softmax(K.real(structures), axis=-1)
    prob = K.reshape(prob, [-1, ch])
    p = prob.shape[0]
    r = np.stack(
        np.array(
            [np.random.choice(ch, p=K.numpy(prob[i]), size=[batch]) for i in range(p)]
        )
    )
    return r.transpose()


# warning pytorch might be unable to do this exactly
@K.jit
def best_from_structure(structures):
    return K.argmax(structures, axis=-1)


def nmf_gradient(structures, oh):
    """
    compute the Monte Carlo gradient with respect of naive mean-field probabilistic model
    """
    choice = K.argmax(oh, axis=-1)
    prob = K.softmax(K.real(structures), axis=-1)
    
    # In vmap, structures shape is [nlayers * n, 7], oh shape is [nlayers * n, 7]
    # choice shape is [nlayers * n]
    # prob shape is [nlayers * n, 7]
    
    # Create indices for gathering
    seq_len = structures.shape[0]
    seq_indices = torch.arange(seq_len, device=structures.device)
    indices = torch.stack([seq_indices, choice], dim=-1)
    
    # Gather the selected probabilities
    prob_gathered = torch.gather(prob, 1, choice.unsqueeze(-1)).squeeze(-1)
    
    prob_gathered = K.reshape(prob_gathered, [-1, 1])
    prob_gathered = K.tile(prob_gathered, [1, structures.shape[-1]])

    # warning pytorch might be unable to do this exactly
    result = torch.zeros_like(structures, dtype=torch.complex64)
    
    # Use scatter_add_ to accumulate gradients
    result.scatter_add_(1, choice.unsqueeze(-1), torch.ones_like(choice, dtype=torch.complex64).unsqueeze(-1))
    
    return K.real(result - prob_gathered)


# warning pytorch might be unable to do this exactly
nmf_gradient_vmap = K.jit(K.vmap(nmf_gradient, vectorized_argnums=1))
# warning pytorch might be unable to do this exactly
vf = K.jit(K.vmap(ansatz, vectorized_argnums=(0, 1)), static_argnums=2)


def main(stddev=0.05, lr=None, epochs=2000, debug_step=50, batch=256, verbose=False):
    so = K.implicit_randn([nlayers * n, 7], stddev=stddev)
    st = K.implicit_randn([nlayers * n, 3], stddev=stddev)
    if lr is None:
        lr = 0.06  # Simplified learning rate
    structure_opt = torch.optim.Adam([so, st], lr=lr)

    avcost = 0
    avcost2 = 0
    for epoch in range(epochs):  # iteration to update strcuture param
        batched_stuctureo = K.onehot(
            sampling_from_structure(so, batch=batch),
            num=so.shape[-1],
        )
        batched_stucturet = K.onehot(
            sampling_from_structure(st, batch=batch),
            num=st.shape[-1],
        )
        vs = vf(batched_stuctureo, batched_stucturet, "direct")
        avcost = K.mean(vs)
        go = nmf_gradient_vmap(so, batched_stuctureo)  # \nabla lnp
        gt = nmf_gradient_vmap(st, batched_stucturet)  # \nabla lnp
        go = K.mean(K.reshape(vs - avcost2, [-1, 1, 1]) * go, axis=0)
        gt = K.mean(K.reshape(vs - avcost2, [-1, 1, 1]) * gt, axis=0)

        # go = [(vs[i] - avcost2) * go[i] for i in range(batch)]
        # gt = [(vs[i] - avcost2) * gt[i] for i in range(batch)]
        # go = torch.math.reduce_mean(go, axis=0)
        # gt = torch.math.reduce_mean(gt, axis=0)
        avcost2 = avcost

        # Update parameters using PyTorch optimizer
        structure_opt.zero_grad()
        so.grad = go
        st.grad = gt
        structure_opt.step()
        
        # so -= K.reshape(K.mean(so, axis=-1), [-1, 1])
        # st -= K.reshape(K.mean(st, axis=-1), [-1, 1])
        if epoch % debug_step == 0 or epoch == epochs - 1:
            print("----------epoch %s-----------" % epoch)
            print(
                "batched average loss: ",
                np.mean(vs.detach().cpu().numpy()),
                "minimum candidate loss: ",
                np.min(vs.detach().cpu().numpy()),
            )
            minp1 = torch.min(torch.max(torch.softmax(st, dim=-1), dim=-1)[0])
            minp2 = torch.min(torch.max(torch.softmax(so, dim=-1), dim=-1)[0])
            if minp1 > 0.3 and minp2 > 0.6:
                print("probability converged")
                break

            if verbose:
                print(gt)
                print(st)
                print(
                    "strcuture parameter: \n",
                    so.detach().cpu().numpy(),
                    "\n",
                    st.detach().cpu().numpy(),
                )

            cand_preseto = best_from_structure(so)
            cand_presett = best_from_structure(st)
            print(
                K.reshape(cand_preseto, [nlayers, n]),
                K.reshape(cand_presett, [nlayers, n]),
            )
            print("current recommendation loss: ", ansatz(so, st, "most"))
    return ansatz(so, st, "most"), so, st


if __name__ == "__main__":
    tries = 1  # 减少尝试次数
    rs = []
    for _ in range(tries):
        ee, _, _ = main(epochs=10, batch=32)  # 减少epochs和batch size
        rs.append(-K.numpy(ee))
    print(np.min(rs))
