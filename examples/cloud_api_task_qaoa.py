
import requests
import json
import getpass
import time

token = getpass.getpass("Enter your token: ")

qasm = '''OPENQASM 2.0;
include "qelib1.inc";
qreg q1[6];
h q1[0];
h q1[1];
h q1[2];
h q1[3];
h q1[4];
h q1[5];
rzz(0.457986902308) q1[0],q1[2];
rzz(0.457986902308) q1[0],q1[3];
rzz(0.457986902308) q1[0],q1[4];
rzz(0.457986902308) q1[1],q1[2];
rzz(0.457986902308) q1[1],q1[3];
rzz(0.457986902308) q1[1],q1[5];
rzz(0.457986902308) q1[2],q1[4];
rzz(0.457986902308) q1[2],q1[5];
rzz(0.457986902308) q1[3],q1[5];
rzz(0.457986902308) q1[4],q1[5];
rx(-0.937095045432) q1[0];
rx(-0.937095045432) q1[1];
rx(-0.937095045432) q1[2];
rx(-0.937095045432) q1[3];
rx(-0.937095045432) q1[4];
rx(-0.937095045432) q1[5];
rzz(0.845744593346) q1[0],q1[2];
rzz(0.845744593346) q1[0],q1[3];
rzz(0.845744593346) q1[0],q1[4];
rzz(0.845744593346) q1[1],q1[2];
rzz(0.845744593346) q1[1],q1[3];
rzz(0.845744593346) q1[1],q1[5];
rzz(0.845744593346) q1[2],q1[4];
rzz(0.845744593346) q1[2],q1[5];
rzz(0.845744593346) q1[3],q1[5];
rzz(0.845744593346) q1[4],q1[5];
rx(-3.70816562646) q1[0];
rx(-3.70816562646) q1[1];
rx(-3.70816562646) q1[2];
rx(-3.70816562646) q1[3];
rx(-3.70816562646) q1[4];
rx(-3.70816562646) q1[5];
'''

def create_task(src):
    url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/tasks/submit_task"
    headers = {"Authorization": "Bearer " + token}

    data = {
    "device": "homebrew_s2?o=3",
    "shots": 1024,
    "source": src,
    "version": "1",
    "lang": "OPENQASM",
    "prior": 1,
    "remarks": "qaoa sample"
    }
    
    response = requests.post(url, json=data, headers=headers)
    response_json = response.json()
    return response_json

def get_task_by_id(task_id):
    url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/tasks/detail"
    headers = {"Authorization": "Bearer " + token}
    response = requests.post(url, json={"task_id": task_id}, headers=headers)
    response_json = response.json()
    return response_json

res = create_task(qasm)
print(res)

print("wait 60 seconds")
time.sleep(60)

result = get_task_by_id(res['id'])
print(result)
