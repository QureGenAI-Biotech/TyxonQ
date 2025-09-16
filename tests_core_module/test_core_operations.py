from tyxonq.core.operations import GateSpec, Operation, registry


def test_gate_spec_and_registry_minimal():
    h = GateSpec(name="h", num_qubits=1, generator=None, differentiable=True)
    registry.register(h)
    assert registry.get("h") is h

    op = Operation(name="h", wires=(0,))
    assert op.name == "h"
    assert op.wires == (0,)


