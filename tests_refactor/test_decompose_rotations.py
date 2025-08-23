from tyxonq.core.ir import Circuit
from tyxonq.compiler.pipeline import build_pipeline


def test_decompose_rotations_lowers_to_native_gates():
    circ = Circuit(
        num_qubits=2,
        ops=[
            ("rx", 0, 0.5),
            ("ry", 1, 0.3),
            ("rxx", 0, 1, 0.2),
            ("ryy", 0, 1, 0.7),
            ("rzz", 0, 1, 1.1),
        ],
    )
    pl = build_pipeline(["decompose/rotations"])
    out = pl.run(circ, caps={})
    # all ops lowered to native set {h, rz, cx}
    native = {"h", "rz", "cx"}
    assert len(out.ops) > len(circ.ops)
    assert all(op[0] in native for op in out.ops)


