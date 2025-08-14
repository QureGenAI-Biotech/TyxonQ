import sys
import os
import getpass
import matplotlib.pyplot as plt
# Add the directory containing your module to Python's search path
module_path = ".."
sys.path.insert(0, module_path)

from tyxonq import Circuit, Param, gates, waveforms
from tyxonq.cloud import apis
import re

shots_const = 100

token = getpass.getpass("Enter your token: ")
apis.set_token(token)
apis.set_provider("tyxonq")

ds = apis.list_devices()
print(ds)



# TQASM 0.2;
# QREG a[1];
# defcal rabi_test a {
# frame drive_frame = newframe(a);
# play(drive_frame, cosine_drag($formatted_t, 0.2, 0.0, 0.0)); }
# rabi_test a[0];
# MEASZ a[0];

def gen_parametric_waveform_circuit(t):
    qc = Circuit(1)

    param0 = Param("a")

    builder = qc.calibrate("rabi_test", [param0])
    builder.new_frame("drive_frame", param0)
    builder.play("drive_frame", waveforms.CosineDrag(t, 0.2, 0.0, 0.0))
    print("defcal rabi_test , instructions: ")
    for instruction in builder.instructions:
        print(instruction)

    # 这段的代码设计有问题，！
    builder.build()   # 注册 calibration
    qc.add_calibration('rabi_test', ['q[0]'])    # 添加调用

    tqasm_code = qc.to_tqasm()

    print(tqasm_code)
    return qc

def run_circuit(qc):
    device_name = "homebrew_s2"
    t = apis.submit_task(
        circuit=qc,
        shots=shots_const,
        device=device_name,
        enable_qos_gate_decomposition=False,
        enable_qos_qubit_mapping=False,
    )
    print(t)
    import time
    time.sleep(30)
    rf = t.results()
    return rf

def exp_rabi():
    result_lst = []
    for t in range(1, 4, 2):
        qc = gen_parametric_waveform_circuit(t)
        result = run_circuit(qc)
        result['duration'] = t
        result_lst.append(result)
    return result_lst



def draw_rabi(result_lst):
    data = {
        'duration': [],
        '0': [],
        '1': []
    }

    for result in result_lst:
        data['0'].append(int(result['0']) / shots_const)
        data['1'].append(int(result['1']) / shots_const)
        data['duration'].append(result['duration'])


    plt.figure(figsize=(10,6))
    plt.plot(data['duration'], data['0'], 'b-o', label='State |0>')
    plt.plot(data['duration'], data['1'], 'r--s', label='State |1>')


    plt.title('Rabi Oscillation Experiment')
    plt.xlabel('Duration (dt)')
    plt.ylabel('Probability')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()


    plt.savefig('rabi.png', dpi=300)
    plt.show()


def run_rabi():
    qc =gen_parametric_waveform_circuit(50)
    print(qc)
    print("-------------------------------- QC TQASM --------------------------------")
    print(qc.to_tqasm())
    print("-------------------------------- QC TQASM END --------------------------------")

    result = run_circuit(qc)
    print(result)

run_rabi()
