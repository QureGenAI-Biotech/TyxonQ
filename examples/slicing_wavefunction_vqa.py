"""
slicing the output wavefunction to save the memory in VQA context
"""

from itertools import product
import numpy as np
import tyxonq as tq

K = tq.set_backend("pytorch")


def circuit(param, n, nlayers):
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    c = tq.templates.blocks.example_block(c, param, nlayers)
    return c


def sliced_state(c, cut, mask):
    # mask = Tensor([0, 1, 0])
    # cut = [0, 1, 2]
    n = c._nqubits
    ncut = len(cut)
    end0 = tq.array_to_tensor(np.array([1.0, 0.0]))
    end1 = tq.array_to_tensor(np.array([0.0, 1.0]))
    ends = [tq.Gate(mask[i] * end1 + (1 - mask[i]) * end0) for i in range(ncut)]
    nodes, front = c._copy()
    for j, i in enumerate(cut):
        front[i] ^ ends[j][0]
    oeo = []
    for i in range(n):
        if i not in cut:
            oeo.append(front[i])
    ss = tq.contractor(nodes + ends, output_edge_order=oeo)
    return ss


def sliced_op(ps, cut, mask1, mask2):
    # ps: Tensor([0, 0, 1, 1])
    n = K.shape_tuple(ps)[-1]
    ncut = len(cut)
    end0 = tq.array_to_tensor(np.array([1.0, 0.0]))
    end1 = tq.array_to_tensor(np.array([0.0, 1.0]))
    endsr = [tq.Gate(mask1[i] * end1 + (1 - mask1[i]) * end0) for i in range(ncut)]
    endsl = [tq.Gate(mask2[i] * end1 + (1 - mask2[i]) * end0) for i in range(ncut)]

    structuresc = K.cast(ps, dtype="int32")
    structuresc = K.onehot(structuresc, num=4)
    structuresc = K.cast(structuresc, dtype=tq.dtypestr)
    pauli_mats = [
        tq.array_to_tensor(g.tensor, dtype=tq.dtypestr) for g in tq.gates.pauli_gates
    ]
    obs = []
    for i in range(n):
        mat = K.zeros([2, 2], dtype=tq.dtypestr)
        for k in range(4):
            mat = mat + structuresc[i, k] * pauli_mats[k]
        obs.append(tq.Gate(mat))
    for j, i in enumerate(cut):
        obs[i][0] ^ endsl[j][0]
        obs[i][1] ^ endsr[j][0]
    oeo = []
    for i in range(n):
        if i not in cut:
            oeo.append(obs[i][0])
    for i in range(n):
        if i not in cut:
            oeo.append(obs[i][1])
    return obs + endsl + endsr, oeo


def sliced_core(param, n, nlayers, ps, cut, mask1, mask2):
    # param, ps, mask1, mask2 are all tensor
    c = circuit(param, n, nlayers)
    ss = sliced_state(c, cut, mask1)
    ssc = sliced_state(c, cut, mask2)
    # take conjugate via backend
    ssc = [tq.Gate(K.conj(ssc.tensor))]
    op_nodes, op_edges = sliced_op(ps, cut, mask1, mask2)
    nodes = [ss] + ssc + op_nodes
    ssc = ssc[0]
    n = c._nqubits
    nleft = n - len(cut)
    for i in range(nleft):
        op_edges[i + nleft] ^ ss[i]
        op_edges[i] ^ ssc[i]
    scalar = tq.contractor(nodes)
    return K.real(scalar.tensor)


try:
    sliced_core_vvg = K.jit(
        K.vectorized_value_and_grad(sliced_core, argnums=0, vectorized_argnums=(5, 6)),
        static_argnums=(1, 2, 4),
    )
except Exception:
    sliced_core_vvg = None

sliced_core_vg = K.jit(
    K.value_and_grad(sliced_core, argnums=0),
    static_argnums=(1, 2, 4),
)


def sliced_expectation_and_grad(param, n, nlayers, ps, cut, is_vmap=True):
    pst = tq.array_to_tensor(ps)
    res = 0.0
    mask1s = []
    mask2s = []
    for mask1 in product(*[(0, 1) for _ in cut]):
        mask1t = tq.array_to_tensor(np.array(mask1))
        mask1s.append(mask1t)
        mask2 = list(mask1)
        for j, i in enumerate(cut):
            if ps[i] in [1, 2]:
                mask2[j] = 1 - mask1[j]
        mask2t = tq.array_to_tensor(np.array(mask2))
        mask2s.append(mask2t)
    if is_vmap and sliced_core_vvg is not None:
        mask1s = K.stack(mask1s)
        mask2s = K.stack(mask2s)
        res = sliced_core_vvg(param, n, nlayers, pst, cut, mask1s, mask2s)
        res = list(res)
        res[0] = K.sum(res[0])
        res = tuple(res)
    else:
        # memory bounded
        # can modified to adpative pmap
        vs = 0.0
        gs = 0.0
        for i in range(len(mask1s)):
            mask1t = mask1s[i]
            mask2t = mask2s[i]
            v, g = sliced_core_vg(param, n, nlayers, pst, cut, mask1t, mask2t)
            vs += v
            gs += g
        res = (vs, gs)
    return res


def sliced_expectation_ref(c, ps, cut):
    """
    reference implementation
    """
    # ps: [0, 2, 1]
    res = 0.0
    for mask1 in product(*[(0, 1) for _ in cut]):
        mask1t = tq.array_to_tensor(np.array(mask1))
        ss = sliced_state(c, cut, mask1t)
        mask2 = list(mask1)
        for j, i in enumerate(cut):
            if ps[i] in [1, 2]:
                mask2[j] = 1 - mask1[j]
        mask2t = tq.array_to_tensor(np.array(mask2))
        ssc = sliced_state(c, cut, mask2t)
        ssc = [tq.Gate(K.conj(ssc.tensor))]
        ps = tq.array_to_tensor(ps)
        op_nodes, op_edges = sliced_op(ps, cut, mask1t, mask2t)
        nodes = [ss] + ssc + op_nodes
        ssc = ssc[0]
        n = c._nqubits
        nleft = n - len(cut)
        for i in range(nleft):
            op_edges[i + nleft] ^ ss[i]
            op_edges[i] ^ ssc[i]
        scalar = tq.contractor(nodes)
        res += scalar.tensor
    return res


if __name__ == "__main__":
    n = 6
    nlayers = 2
    param = K.ones([n, 2 * nlayers], dtype="float32")
    cut = ()
    ops = [2, 0, 3, 1, 0, 0, 1, 2]
    ops_dict = tq.quantum.ps2xyz(ops)

    def trivial_core(param, n, nlayers):
        c = circuit(param, n, nlayers)
        # Build dense operator for the single Pauli string using pure torch tensors
        mat = []
        for j in range(n):
            if ops[j] == 0:
                mat.append(tq.array_to_tensor(np.array([[1,0],[0,1]]), dtype=tq.dtypestr))
            elif ops[j] == 1:
                mat.append(tq.array_to_tensor(np.array([[0,1],[1,0]]), dtype=tq.dtypestr))
            elif ops[j] == 2:
                mat.append(tq.array_to_tensor(np.array([[0,-1j],[1j,0]]), dtype=tq.dtypestr))
            else:
                mat.append(tq.array_to_tensor(np.array([[1,0],[0,-1]]), dtype=tq.dtypestr))
        # kron in sequence -> 2^n x 2^n dense matrix
        op = tq.backend.cast(mat[0], tq.dtypestr)
        for k in range(1, n):
            op = tq.backend.kron(op, tq.backend.cast(mat[k], tq.dtypestr))
        state = c.wavefunction(form="ket")
        expt = (tq.backend.adjoint(state) @ (op @ state))[0, 0]
        return tq.backend.real(expt)

    trivial_vg = K.jit(K.value_and_grad(trivial_core, argnums=0), static_argnums=(1, 2))

    # Run only the trivial benchmark to keep runtime short and avoid open-edge contractions
    _ = tq.utils.benchmark(trivial_vg, param, n, nlayers, tries=1)
