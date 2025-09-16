from tyxonq.core.operations import registry


def test_default_gate_registry_contains_h_rz_cx():
    for name in ["h", "rz", "cx"]:
        spec = registry.get(name)
        assert spec is not None
        assert spec.name == name
        assert spec.num_qubits in (1, 2)
    rz = registry.get("rz")
    assert rz is not None and rz.is_shiftable and rz.num_params == 1


