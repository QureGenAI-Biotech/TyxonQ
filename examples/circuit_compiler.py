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
            # output='ir' uses native compiler by default; request qiskit artifacts explicitly below
            cc = c.compile(
                compile_engine="default",
                output="ir",
                options={"optimization_level": lvl, "basis_gates": ["cx", "cz", "h", "rz"]},
            )
            compiled.append((lvl, cc))
        except Exception as e:
            print(f"qiskit compile failed at level {lvl}: {e}")
    for lvl, cc in compiled:
        # Directly use our Circuit.draw() which compiles to qiskit under the hood
        print(f"level {lvl} drawing:")
        print(cc.draw())


def main():
    qiskit_compile_levels()
    try:
        c = build_demo_circuit()
        qasm = c.compile(compile_engine="qiskit", output="qasm2", options={"basis_gates": ["cx", "cz", "h", "rz"]})
        print("qasm2 length:", len(qasm))
    except Exception:
        pass


if __name__ == "__main__":
    main()
