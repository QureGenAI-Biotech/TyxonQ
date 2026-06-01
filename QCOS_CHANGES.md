# TyxonQ QCOS Integration - Change Log

Connects TyxonQ directly to quantum hardware on China Mobile ecloud via `wuyue.plugin.runner.Runner`.
No local QCOS Docker required.

---

## New Files

### `src/tyxonq/devices/hardware/qcos/__init__.py`

Module init, exports the driver.

### `src/tyxonq/devices/hardware/qcos/driver.py`

QCOS driver that uses `wuyue.plugin.runner.Runner` internally.

Key components:

| Component | Description |
|-----------|-------------|
| `_get_credentials()` | Extracts `access_key`, `secret_key` from opts or env vars |
| `QCOSTask` | Task wrapper with `id`, `device`, `status`, `wuyue_result`, `async_result` |
| `run()` | Receives a Qiskit QuantumCircuit, submits via `Runner.run()` |
| `get_task_details()` | Returns unified result dict from WuYue `Result` object |
| `list_devices()` | Returns a static catalog of known WuYue device IDs |

> **Migration note (2026-04):** China Mobile removed `sdk_code` and
> `License.init_license(...)`; authenticate with `access_key` + `secret_key` only.
>
> **Migration note (2026-05):** The two-package layout (`wuyue_open` +
> `wuyue_plugin`) was replaced by a single unified `wuyue` whl. Importing
> `wuyue_plugin` will fail; the driver now imports from `wuyue.plugin.runner`.
> `Runner.get_eng_list()` was also removed, so `list_devices()` returns a
> static catalog. Install the new whl with
> `pip install wuyue-1.0-py3-none-any.whl`. The new whl pins
> `qiskit==1.4.3`; use `pip install ... --no-deps` or a dedicated env if you
> need a newer qiskit for other code.

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
     |  wuyue.plugin.runner.Runner.run(qc, ...)
     v
China Mobile ecloud API (ecloud.10086.cn)
     |
     v
Quantum hardware
```

---

## Installation

To use wuyue and connect to quantum hardware on China Mobile ecloud, install the
unified `wuyue` SDK. Due to the WuYue SDK requirements, use Python 3.11.

Firstly go to [China Mobile ecloud console](https://ecloud.10086.cn/api/page/wyqcloud/web/console/#/overview_home),
setup your China Mobile ecloud account. Then go to 编程框架本地部署 (Deploy SDK locally),
download the WuYue SDK and install the single whl:

```
pip install wuyue-1.0-py3-none-any.whl
```

The previous two-package layout (`wuyue_open-0.5` + `wuyue_plugin-1.0`) is no
longer published; both names are gone in the 2026-05 release.

Then prepare your access key and secret key from the China Mobile console.

`wuyue==1.0` hard-pins `qiskit==1.4.3`. If you need a newer qiskit for other
work in the same environment, either keep a dedicated env for QCOS or install
with `--no-deps` and pick a qiskit version yourself.

In case of package conflict, we recommend reinstalling tyxonq from source after
installing the wuyue whl.

## Usage

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
    timeout=100,          # seconds, 0 for async
    wait_async_result=True,
)

print(results[0]["result"])  # e.g. {"00": 52, "11": 48}
```

Credentials can also be set via environment variables instead of passing them every time:

```bash
export QCOS_ACCESS_KEY="your_access_key"
export QCOS_SECRET_KEY="your_secret_key"
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
