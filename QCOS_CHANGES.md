# TyxonQ QCOS Integration - Change Log

Connects TyxonQ directly to 移动云量子真机 via `wuyue_plugin.runner.Runner`.
No local QCOS Docker required.

---

## New Files

### `src/tyxonq/devices/hardware/qcos/__init__.py`

Module init, exports the driver.

### `src/tyxonq/devices/hardware/qcos/driver.py`

QCOS driver that uses `wuyue_plugin.Runner` internally.

Key components:

| Component | Description |
|-----------|-------------|
| `_ensure_license()` | Calls `License.init_license(sdk_code, access_key, secret_key)` once |
| `_get_credentials()` | Extracts `access_key`, `secret_key`, `sdk_code` from opts or env vars |
| `QCOSTask` | Task wrapper with `id`, `device`, `status`, `wuyue_result`, `async_result` |
| `run()` | Receives a Qiskit QuantumCircuit, submits via `Runner.run()` |
| `get_task_details()` | Returns unified result dict from WuYue `Result` object |
| `list_devices()` | Lists available devices via `Runner.get_eng_list()` |

---

## Modified Files

### `src/tyxonq/devices/base.py`

Added `"qcos"` provider to `resolve_driver()`:

```python
if provider == "qcos":
    from .hardware.qcos import driver as drv
    return drv
```

### `src/tyxonq/devices/hardware/config.py`

Added QCOS endpoint to `ENDPOINTS`:

```python
"qcos": {
    "base_url": "https://ecloud.10086.cn",
},
```

### `src/tyxonq/core/ir/circuit.py`

Two changes in the `run()` method:

1. **Fixed parameter priority bug**: Explicit `provider`/`device`/`shots` now correctly override `_device_opts` defaults.

2. **Added QCOS compilation path**: When `provider="qcos"`, converts TyxonQ IR directly to a Qiskit QuantumCircuit via `to_qiskit()` and passes it to the driver.

---

## Architecture

```
TyxonQ Circuit
     |  to_qiskit(circuit)
     v
Qiskit QuantumCircuit
     |  wuyue_plugin.Runner.run(qc, ...)
     v
移动云 cloud API (ecloud.10086.cn)
     |
     v
量子真机 / 模拟器
```

---

## Usage

To use wuyue and connect to 移动云量子真机, firstly we need to install wuyue_open and wuyue_plugin via

```
pip install wuyue_open-0.5-py3-none-any.whl
pip install wuyue_plugin-1.0-py3-none-any.whl
```

Then you need to prepare your access key, secret key and sdk code from China Mobile website.

```python
from tyxonq import Circuit

c = Circuit(2)
c.h(0).cx(0, 1).measure_z(0).measure_z(1)

results = c.run(
    provider="qcos",
    device="WuYue-QPUSim-FullAmpSim",
    shots=100,
    access_key="your_access_key",
    secret_key="your_secret_key",
    sdk_code="your_sdk_code",
    timeout=100,          # seconds, 0 for async
    wait_async_result=True,
)

print(results[0]["result"])  # e.g. {"00": 52, "11": 48}
```

Credentials can also be set via environment variables instead of passing them every time:

```bash
export QCOS_ACCESS_KEY="your_access_key"
export QCOS_SECRET_KEY="your_secret_key"
export QCOS_SDK_CODE="your_sdk_code"
```

```python
results = c.run(
    provider="qcos",
    device="WuYue-QPUSim-FullAmpSim",
    shots=100,
    timeout=100,
    wait_async_result=True,
)
```
