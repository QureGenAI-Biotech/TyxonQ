import requests
import json
import getpass
import time

token = getpass.getpass("Enter your token: ")

def get_tasks():
    
    url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/tasks/api_key/list"
    headers = {"Authorization": "Bearer " + token}
    data = {'task_type': 'quantum_api'}
            
    try:
        response = requests.post(url, json=data, headers=headers)
        
        response_json = response.json()
        return response_json
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

def get_task_by_id(task_id):
    url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/tasks/detail"
    headers = {"Authorization": "Bearer " + token}
    response = requests.post(url, json={"task_id": task_id}, headers=headers)
    response_json = response.json()
    return response_json

def create_task():
    url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/tasks/submit_task"
    headers = {"Authorization": "Bearer " + token}

    data = {
    "device": "homebrew_s2?o=3",
    "shots": 100,
    "source": "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];",
    "version": "1",
    "lang": "OPENQASM",
    "prior": 1,
    "remarks": "Bell state preparation"
    }
    
    response = requests.post(url, json=data, headers=headers)
    response_json = response.json()
    return response_json

if __name__ == "__main__":
    print("get tasks")
    tasks = get_tasks()
    print(json.dumps(tasks, indent=4))
    
    print("create task")
    task_create_result = create_task()
    print(json.dumps(task_create_result, indent=4))
    task_id = task_create_result['id']
    print(f"new task id: {task_id}")

    print("wait for task to be completed")
    time.sleep(10)
    print("get task by id")
    task_res = get_task_by_id(task_id)
    print(json.dumps(task_res, indent=4))
