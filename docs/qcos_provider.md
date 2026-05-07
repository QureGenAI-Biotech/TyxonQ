<!-- docs/qcos_provider.md -->
# QCOS / 移动云 (WuYue) Provider

TyxonQ ships a `qcos` provider that submits circuits to **China Mobile's
WuYue (五岳) Quantum Cloud Platform** via the official `wuyue_plugin`
SDK. Both real superconducting hardware and full-amplitude simulators are
exposed through the same interface.

## Prerequisites

Install the WuYue plugin (it is NOT a TyxonQ runtime dependency — it must
be installed separately):

```bash
pip install wuyue wuyue_open wuyue_plugin
```

The `qcos` driver lazy-imports `wuyue_plugin.runner.Runner` only when you
actually submit a job, so simply having TyxonQ installed without the
plugin will not break.

## Getting credentials

1. Register at the China Mobile Cloud quantum portal and create an
   application.
2. Generate an **access_key** + **secret_key** pair from the portal's
   credentials page (this replaces the older `sdk_code` flow that was
   retired in 2026-04 — the driver explicitly rejects `sdk_code` to avoid
   silent breakage).
3. Treat the secret_key like a password.

## Configuring credentials

Two resolution paths, in order of precedence:

```python
import tyxonq as tq

# 1. Explicit kwargs on c.run() / .device() (highest precedence)
c.run(
    provider="qcos",
    device="WuYue-QPUSim-FullAmpSim",
    shots=1024,
    access_key="abc...",
    secret_key="xyz...",
)

# 2. Environment variables (preferred for CI / shells)
# export QCOS_ACCESS_KEY=abc...
# export QCOS_SECRET_KEY=xyz...
```

If both are missing, the driver raises:

```
ValueError: QCOS credentials missing: access_key (or QCOS_ACCESS_KEY env var),
            secret_key (or QCOS_SECRET_KEY env var)
```

> Note: the legacy `sdk_code` / `QCOS_SDK_CODE` flow is no longer
> supported. If you have old code that uses it, the driver raises with a
> pointer to upgrade `wuyue / wuyue_open / wuyue_plugin` to the post-2026-04
> release and switch to access_key + secret_key.

## Listing online devices

```python
import tyxonq as tq
print(tq.api.list_devices(provider="qcos"))
# ['qcos::WuYue-QPUSim-FullAmpSim', 'qcos::<chip-id>', ...]
```

Listing requires credentials (the WuYue API gates the device catalog
behind authentication). If credentials are absent, `list_devices` returns
`[]` and logs a warning rather than raising.

## Submitting a circuit (Chain API)

Unlike the `quafu` provider, the `qcos` driver accepts a Qiskit
`QuantumCircuit` object — TyxonQ's compile pipeline handles the
IR → Qiskit conversion automatically. You can use the chain API with no
explicit pre-compile:

```python
import tyxonq as tq

c = tq.Circuit(2)
c.h(0).cx(0, 1).measure_z(0).measure_z(1)

results = c.run(
    provider="qcos",
    device="WuYue-QPUSim-FullAmpSim",
    shots=1024,
    access_key="abc...",
    secret_key="xyz...",
    timeout=100,
    wait_async_result=True,
)

print(results[0]["result"])
# {'00': 510, '11': 514}
```

## Run options

| Option | Purpose | Default |
|---|---|---|
| `device` | Device id, e.g. `"WuYue-QPUSim-FullAmpSim"`. May be `qcos::<id>`. | (required) |
| `shots` | Per-task shot count. | `1024` |
| `access_key` / `secret_key` | Credentials; falls back to env vars. | — |
| `timeout` | Wait timeout in seconds; pass `0` for fully async. | `100` |
| `auto_retry` | Retry transient failures inside the WuYue runner. | `True` |
| `task_name` | Human-readable name shown in the WuYue console. | — |
| `calculate_type` | Device-specific compute mode. | — |
| `bit_info`, `qmachine_type`, `dry_run`, `initial_mapping`, ... | Forwarded directly to `wuyue_plugin.Runner`. Unknown keys are filtered by `Runner.param_map`, so future plugin params work without driver changes. | — |

## Async submission

Pass `timeout=0` to submit without waiting for results, then poll later
via the unified API:

```python
tasks = c.run(
    provider="qcos",
    device="WuYue-QPUSim-FullAmpSim",
    shots=1024,
    timeout=0,                # async submit
    wait_async_result=False,
)

# ... do other work ...

details = tq.api.get_task_details(tasks[0])
print(details["uni_status"])  # "submitted" | "queued" | "running" | "completed" | "failed"
print(details["result"])      # bitstring → count, populated only on "completed"
```

## Result shape

The driver returns TyxonQ's unified result shape:

```python
{
    "result": {"00": 510, "11": 514},        # bitstring -> count
    "result_meta": {
        "task_id": "...",                     # WuYue task id
        "device": "WuYue-QPUSim-FullAmpSim",
        "prob": [...],                        # probabilities, when available
        "amps": [...],                        # amplitudes, simulator only
        "raw": {...},                         # full upstream response
    },
    "uni_status": "completed",
    "error": "",
}
```

Status code mapping (from `wuyue_plugin` → TyxonQ unified):

| Upstream code | Meaning | Unified `uni_status` |
|---|---|---|
| 1 | Submitted | `submitted` |
| 2 | Queued | `queued` |
| 3 | Running | `running` |
| 4 | Waiting | `waiting` |
| 5 | Finished | `completed` |
| 6 | Failed | `failed` |
| any other | — | `unknown` |

