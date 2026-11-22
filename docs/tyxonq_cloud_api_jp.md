# TyxonQ Cloud API ドキュメント

## 概要

TyxonQ Cloud API は、量子計算タスクの管理、デバイス情報の取得、タスクの送信を行うための HTTP エンドポイントを提供しています。データの送受信には JSON を使用します。

サンプルコード：[examples/cloud_api_task.py](../examples/cloud_api_task.py) , [examples/cloud_api_device.py](../examples/cloud_api_device.py)

## 基本設定

- **ベース URL**: `https://api.tyxonq.com/qau-cloud/tyxonq/`
- **API バージョン**: `v1`
- **認証方法**: Authorization ヘッダに Bearer Token を指定
- **コンテンツタイプ**: `application/json`

## 認証

すべての API リクエストには Bearer Token を用いた認証が必要です：

```http
Authorization: Bearer YOUR_TOKEN
```

## API エンドポイント

### 1. デバイス管理

#### 1.1 利用可能なデバイス一覧

**エンドポイント**: `POST /api/v1/devices/list`

**リクエストボディ**:
```json
{
  // Optional filter parameters
}
```

**レスポンスボディ**:
```json
{
  "devices": [
    {
      "id": "homebrew_s2",
      "qubits": 13,
      "T1": 84.44615173339844,
        "T2": 45.41538619995117,
        "Err": {
            "SQ": 0.0007843076923076923,
            "CZ": 0.009009666666666666,
            "Readout": {
                "F0": 0.016538461538461537,
                "F1": 0.04118461538461538
            }
        },
      "state": "running"
    }
  ]
}
```

**使用例**:

サンプルプログラムファイル: examples/cloud_api_devices.py

```python
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

```

### 2. タスク管理

#### 2.1 タスク提出

**エンドポイント**: `POST /api/v1/tasks/submit_task`

**パラメータ説明**:

| パラメータ     | 型                        | 必須    | デフォルト      | 説明                     |
| --------- | ------------------------ | ----- | ---------- | ---------------------- |
| `device`  | string                   | Yes   | -          | 量子デバイス ID、最適化フラグ付き指定可能 |
| `shots`   | int \| array[int]        | No    | 1024       | 実行する測定ショット数            |
| `source`  | string \| array[string]  | Yes*  | -          | OpenQASM 形式の回路定義       |
| `lang`    | string                   | No    | "OPENQASM" | 回路コードの言語               |
| `version` | string                   | No    | "1"        | タスク送信プロトコルバージョン        |
| `prior`   | int                      | No    | 1          | 実行キュー内の優先度 (1–10)      |
| `remarks` | string                   | No    | -          | 任意の説明または備考             |
| `group`   | string                   | No    | -          | タスクグループ識別子             |


*注: source または circuit のいずれかが必須。


**リクエストボディ例**:
```json
{
  "device": "device_id?o=3",
  "shots": 1024,
  "source": "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];",
  "version": "1",
  "lang": "OPENQASM",
  "prior": 1,
  "remarks": "Optional task description",
  "group": "Optional group identifier"
}
```

**デバイス最適化オプションの説明**:

デバイスパラメータは、加算によって組み合わせ可能な最適化フラグをサポートしています：

| オプション                       | 値 | 説明                           |
| ------------------------------- | - | ---------------------------- |
| `enable_qos_qubit_mapping`      | 1 | 自動量子ビットマッピングを有効化             |
| `enable_qos_gate_decomposition` | 2 | ゲートをデバイスがサポートするネイティブゲート集合に分解 |
| `enable_qos_initial_mapping`    | 4 | 初期量子ビット配置を最適化                |

**常用デバイスパラメータの例**:

| デバイスパラメータ           | 最適化レベル     | 説明                      |
| ------------------- | ---------- | ----------------------- |
| `device_id`         | 最適化なし      | 最適化を行わない基本実行            |
| `device_id?o=0`     | 最適化なし      | 明示的にすべての最適化を無効化         |
| `device_id?o=1`     | ビットマッピングのみ | 自動ビットマッピングを有効化          |
| `device_id?o=2`     | ゲート分解のみ    | ゲートをネイティブ集合に分解          |
| `device_id?o=3`     | マッピング + 分解 | 両方を有効化 (1+2=3)          |
| `device_id?o=4`     | 初期マッピングのみ  | 初期配置を最適化                |
| `device_id?o=7`     | 全最適化       | すべての最適化戦略を有効化 (1+2+4=7) |
| `device_id?o=3&dry` | Dry run    | コンパイルのみ、実行しない（テスト用）     |

**レスポンス例**:
```json
{
    "id": "<JOB_ID>",
    "job_name": "<JOB_NAME>",
    "status": "<STATUS>",
    "user_id": "<USER_ID>",
    "success": true,
    "error": null
}
```

**使用例**:

**シングルタスクの提出**:
```python
def create_task():
    url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/tasks/submit_task"
    headers = {"Authorization": "Bearer " + token}

    data = {
    "device": "homebrew_s2?o=3",
    "shots": 100,
    "source": """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];""",
    "version": "1",
    "lang": "OPENQASM",
    "prior": 1,
    "remarks": "Bell state preparation"
    }
    
    response = requests.post(url, json=data, headers=headers)
    response_json = response.json()
    return response_json
```

**パラメータ検証**:

| パラメータ     | 検証ルール                                           |
| --------- | ----------------------------------------------- |
| `device`  | 有効なデバイス ID でなければならない。最適化フラグは正しい組み合わせである必要がある    |
| `shots`   | 正の整数。範囲: 1\~1000000                              |
| `source`  | lang="OPENQASM" の場合、有効な OpenQASM 2.0 構文である必要がある |
| `prior`   | 1\~10 の整数である必要がある (1=最高優先度)                      |
| `version` | 現在 "1" のみサポート                                   |
| `lang`    | 現在 "OPENQASM" のみサポート                            |


#### 2.2 タスク詳細取得

**エンドポイント**: `POST /api/v1/tasks/detail`

**リクエストボディ**:
```json
{
  "task_id": "task_uuid"
}
```

**レスポンスボディ**:
```json
{
    "success": true,
    "task": {
        "id": "<JOB_ID>",
        "queue": "quregenai.lab",
        "device": "homebrew_s2?o=3",
        "qubits": 2,
        "depth": 3,
        "state": "completed",
        "shots": 100,
        "at": 1754275505649825,
        "ts": {
            "completed": 1754275505649825,
            "pending": 1754275502265270,
            "scheduled": 1754275502260031
        },
        "md5": "f31a82f44a53bc8fa6e08ef0c6a34d53",
        "runAt": 1754275488761744,
        "runDur": 2532053,
        "atChip": 1754275446369691,
        "durChip": 120185,
        "result": {
            "00": 33,
            "01": 2,
            "10": 4,
            "11": 61
        },
        "task_type": "quantum_api"
    }
}
```

**タスク状態説明**:
- `pending`: キュー待機中
- `running`: 実行中
- `completed`: 正常完了
- `failed`: エラーで失敗

**使用例**:
```python
url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/tasks/detail"
data = {"task_id": "task_uuid"}
response = requests.post(url, json=data, headers=headers)
task_details = response.json()["task"]
```

#### 2.3 タスクリスト取得

**エンドポイント**: `POST /api/v1/tasks/api_key/list`

**リクエストボディ**:
```json
{
  "device": "device_id",
  "task_type": "quantum_api"
  // Optional filter parameters
}
```

**レスポンスボディ**:
```json
{
  "tasks": [
    {
      "task_id": "task_id",
      "task_type": "quantum_api",
      "status": "completed",
      "parameters": "",
      "result": "",
      "job_name": "task_uuid",
      "device": "device_id",
      "created_at": "",
      "updated_at": "",
      "completed_at": ""
    }
  ]
}
```

**使用例**:
```python
url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/tasks/api_key/list"
data = {
    "device": "quantum_processor",
    "task_type": "quantum_api"
}
response = requests.post(url, json=data, headers=headers)
tasks = response.json()["tasks"]
```


## 回路言語サポート

### OpenQASM 2.0

API は OpenQASM 2.0 フォーマットでの回路定義をサポートしています：

```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
```

### サポートされるゲート

- 単一量子ビットゲート: `h`, `x`, `y`, `z`, `s`, `t`, `rz`
- 二量子ビットゲート: `cx`, `cz`
- 測定: `measure`

## 推奨事項

1. **Token 管理**: API トークンを安全に保管し、定期的にローテーションする。  
2. **エラーハンドリング**: エラーレスポンスを常に確認し、適切に処理する。  
3. **タスク監視**: タスクの進行状況を監視するために詳細エンドポイントを利用する。  
4. **最適化**: 回路の要件に応じて適切な最適化レベルを選択する。  

## SDK 統合

TyxonQ Python SDK は、これらの API エンドポイントに対して高レベルの抽象化を提供する：

```python
import tyxonq

# トークン設定
tyxonq.set_token("YOUR_TOKEN")

# デバイス一覧
devices = tyxonq.list_devices()

# タスク提出
task = tyxonq.submit_task(
    device="quantum_processor",
    circuit=my_circuit,
    shots=1024
)

# 結果取得
results = task.results(blocked=True)
```

## サポート

API に関するサポートやご質問は、TyxonQ ドキュメントをご参照いただくか、サポートチームまでご連絡ください。