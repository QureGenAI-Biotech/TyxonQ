"""
Compilation utilities demo (refactored to chain-style API).
"""

import tyxonq as tq


def build_demo_circuit() -> tq.Circuit:
    c = tq.Circuit(3)
    c.rx(0, theta=0.2)
    c.rz(0, theta=-0.3)
    c.h(2)
    c.cx(0, 1)
    c.measure_z(0).measure_z(1).measure_z(2)
    # Prefer text draw by default
    c._draw_output = "text"
    return c


def qiskit_compile_levels():
    c = build_demo_circuit()
    levels = [0, 1, 2, 3]
    compiled = []
    for lvl in levels:
        try:
            # compile() returns Circuit object with compiled_source stored in _compiled_source
            compiled_circuit = c.compile(
                compile_engine="default",
                output="ir",
                options={"optimization_level": lvl, "basis_gates": ["cx", "cz", "h", "rz"]},
            )
            # Extract circuit IR object
            cc = compiled_circuit
            compiled.append((lvl, cc))
        except Exception as e:
            print(f"compile failed at level {lvl}: {e}")
    for lvl, cc in compiled:
        # Directly use our Circuit.draw() which compiles to qiskit under the hood
        print(f"level {lvl} drawing:")
        print(cc.draw())


def main():
    qiskit_compile_levels()
    try:
        c = build_demo_circuit()
        # compile() returns Circuit object with compiled_source stored in _compiled_source
        compiled_circuit = c.compile(compile_engine="qiskit", output="qasm2", options={"basis_gates": ["cx", "cz", "h", "rz"]})
        # Extract compiled source (QASM2 string)
        qasm = compiled_circuit._compiled_source
        print("qasm2 length:", len(qasm))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
