import tyxonq as tq
import getpass


def quantum_hello_world():
    # Configure token and defaults via top-level API (preferred)
    tq.set_token(getpass.getpass("Input your TyxonQ API_KEY: "), provider="tyxonq", device="homebrew_s2")
    tq.device(provider="tyxonq", device="homebrew_s2", shots=100)

    c = tq.Circuit(2)
    c.h(0)
    c.cnot(0, 1)
    c.rx(1, theta=0.2)

    # Chain style, rely on global defaults; show omit of compile/postprocessing
    res = c.run()
    print("Result:", res)


if __name__ == "__main__":
    quantum_hello_world()

