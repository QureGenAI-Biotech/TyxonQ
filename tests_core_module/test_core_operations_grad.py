from tyxonq.core.operations import GateSpec, registry


def test_gate_spec_gradient_metadata_defaults():
    spec = GateSpec(name="custom", num_qubits=1)
    assert spec.num_params == 0
    assert spec.is_shiftable is False
    assert spec.shift_coeffs is None
    assert spec.gradient_method is None


def test_gate_spec_parameter_shift_metadata():
    rz = GateSpec(
        name="rz",
        num_qubits=1,
        num_params=1,
        is_shiftable=True,
        shift_coeffs=(0.5,),  # typical coefficient for parameter-shift of RZ
        gradient_method="parameter-shift",
        generator="Z",
    )
    registry.register(rz)
    got = registry.get("rz")
    assert got is not None
    assert got.is_shiftable is True
    assert got.num_params == 1
    assert got.shift_coeffs == (0.5,)
    assert got.gradient_method == "parameter-shift"


