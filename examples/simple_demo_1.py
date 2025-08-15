import tyxonq as tq
import getpass
from tyxonq.cloud import apis
import time
# Configure for real quantum hardware
apis.set_token(getpass.getpass("Input your TyxonQ API_KEY: "))

provider = "tyxonq"
device = "homebrew_s2"

# Create and execute quantum circuit on real hardware
def quantum_hello_world():
    c = tq.Circuit(2)
    c.H(0)                    # Hadamard gate on qubit 0
    c.CNOT(0, 1)             # CNOT gate between qubits 0 and 1
    c.rx(1, theta=0.2)       # Rotation around x-axis
    
    # Execute on real quantum hardware

    print(c.to_openqasm())
    print("Submit task to TyxonQ")

    task = apis.submit_task(provider = provider,
                        device = device,
                        circuit = c,
                        shots = 100)
    print(f"Task submitted: {task}")
    print("Wait 20 seconds to get task details")
    time.sleep(20)
    print(f"Real quantum hardware result: {task.details()}")

quantum_hello_world()

