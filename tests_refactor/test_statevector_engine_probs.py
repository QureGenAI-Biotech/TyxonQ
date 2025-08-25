import numpy as np

from tyxonq.core.ir import Circuit
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine


def test_wavefunction_engine_state_prob_amp_sampling():
    eng = StatevectorEngine()
    # |+> on qubit 0, |0> on qubit 1
    c = Circuit(num_qubits=2, ops=[("h", 0)])

    s = eng.state(c)
    assert s.shape == (4,)
    p = eng.probability(c)
    # probabilities: 00 and 10 each 0.5 in big-endian indexing -> states 0 and 2
    assert np.isclose(np.sum(p), 1.0)
    assert np.isclose(p[0] + p[2], 1.0)

    a00 = eng.amplitude(c, "00")
    a10 = eng.amplitude(c, "10")
    a01 = eng.amplitude(c, "01")
    assert np.isclose(abs(a00) ** 2 + abs(a10) ** 2 + abs(a01) ** 2 + abs(eng.amplitude(c, "11")) ** 2, 1.0)
    assert np.isclose(abs(a01), 0.0)

    bits, prob = eng.perfect_sampling(c)
    assert bits in ("00", "10")
    assert 0.0 <= prob <= 1.0


