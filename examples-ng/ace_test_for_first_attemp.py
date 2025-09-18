import tyxonq as tq
from tyxonq.cloud import apis # new feature called api
import time
# Configure for real quantum hardware

provider = "local"
device = "mps"

tq.devices
tq.complier



#可视化。  context  ，，
# 
#  任务 task  

# Create and execute quantum circuit on real hardware
def quantum_hello_world():
    c = tq.Circuit(2)
    c.H(0)                    # Hadamard gate on qubit 0
    c.CNOT(0, 1)             # CNOT gate between qubits 0 and 1
    c.rx(1, theta=0.2)       # Rotation around x-axis
    
    # Execute on real quantum hardware

    task = c.submit_task(provider = provider,
                        device = device,
                        compiler = 'qiskit',
                        auto_compile = True,
                        shots = 100)
    
    #new method
    # task = c.run(provider = provider,
    #                     device = device,
    #                     compiler = 'qiskit',
    #                     auto_compile = True,
    #                     shots = 100)

    """
    print(c.to_openqasm())
    print("Submit task to TyxonQ")
    task = apis.submit_task(provider = provider,
                    device = device,
                    circuit = c,
                    compiler = 'qiskit',
                    auto_compile = True,
                    shots = 100)
    """
    print(f"Task submitted: {task}")
    for res in task:
        print(f"Real quantum hardware result: {res.details()}")
if __name__ == "__main__":
    quantum_hello_world()
