import math

from tyxonq.core.ir import Circuit
from tyxonq.compiler.gradients.parameter_shift import generate_shifted_circuits


def test_generate_shifted_circuits_for_rz_single_param():
    theta = 0.3
    with Circuit(num_qubits=1) as cb:
        cb.rz(0, theta)
        cb.measure_z(0)
    circ = cb.circuit()

    cplus, cminus, meta = generate_shifted_circuits(circ, match_op_name="rz")
    # Should carry same structure length
    assert len(cplus.ops) == len(cminus.ops) == len(circ.ops)
    # Parameter shifted by ±pi/2
    assert abs(cplus.ops[0][2] - (theta + math.pi / 2)) < 1e-12
    assert abs(cminus.ops[0][2] - (theta - math.pi / 2)) < 1e-12
    assert meta["coeff"] == 0.5


