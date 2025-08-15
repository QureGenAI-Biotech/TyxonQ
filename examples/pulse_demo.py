import sys
import os
import getpass
# Add the directory containing your module to Python's search path
module_path = ".."
sys.path.insert(0, module_path)

from tyxonq import Circuit, Param, gates, waveforms
from tyxonq.cloud import apis
import re
import time
    
shots_const = 100

token = getpass.getpass("Enter your token: ")
apis.set_token(token)
apis.set_provider("tyxonq")

#ds = apis.list_devices()
#print(ds)

# TQASM 0.2;
# QREG a[1];
# defcal rabi_test a {
# frame drive_frame = newframe(a);
# play(drive_frame, cosine_drag($formatted_t, 0.2, 0.0, 0.0)); }
# rabi_test a[0];
# MEASZ a[0];

def gen_parametric_waveform_circuit(t):
    qc = Circuit(1)
    qc.use_pulse()
    param0 = Param("a")

    builder = qc.calibrate("hello_world", [param0])
    builder.new_frame("drive_frame", param0)
    builder.play("drive_frame", waveforms.CosineDrag(t, 0.2, 0.0, 0.0))
    #print("defcal hello_world , instructions: ")
    #for instruction in builder.instructions:
    #    print(instruction)

    builder.build()
    qc.add_calibration('hello_world', ['q[0]'])    # 添加调用

    tqasm_code = qc.to_tqasm()

    return qc

def run_circuit(qc):
    device_name = "homebrew_s2"
    task = apis.submit_task(
        circuit=qc,
        shots=shots_const,
        device=device_name,
        enable_qos_gate_decomposition=False,
        enable_qos_qubit_mapping=False,
    )
    #print(task)
    print("Wait 30 seconds to get task details")
    time.sleep(30)
    #print("Get task details")
    #print(task.details())
    rf = task.results()

    return rf


def run_hello_world(tm):
    print("Run hello world with tm: ", tm)
    qc =gen_parametric_waveform_circuit(tm)
    #print("-------------------------------- QC TQASM --------------------------------")
    #print(qc.to_tqasm())
    #print("-------------------------------- QC TQASM END --------------------------------")

    result = run_circuit(qc)
    print(result)

for tm in range(1, 100, 20):
    run_hello_world(tm)
#run_hello_world(50)
