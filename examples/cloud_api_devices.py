import requests
import json
import getpass

token = getpass.getpass("Enter your token: ")

url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/devices/list"
headers = {"Authorization": "Bearer " + token}
response = requests.post(url, json={}, headers=headers)
response_json = response.json()

if 'success' in response_json and response_json['success']:
    if 'devices' in response_json:
        print(json.dumps(response_json['devices'], indent=4))
    else:
        print("No devices found")
else:
    print("Error:")
    print(response_json['detail'])



