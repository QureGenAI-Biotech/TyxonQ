# BAQIS Quafu Cloud Provider

TyxonQ ships a `quafu` provider that submits OpenQASM 2.0 circuits to the
**BAQIS Quafu Superconducting Quantum Computing Cloud** at
`https://quafu-sqc.baqis.ac.cn`.

## Getting a token

1. Register at <https://quafu-sqc.baqis.ac.cn/>.
2. Copy your personal token from the SQCLab dashboard.
3. Tokens **rotate every 30 days**; refresh when you see a `token rejected`
   error.

## Configuring the token

Four resolution paths, in order of precedence:

```python
import tyxonq as tq

# 1. Explicit kwarg (highest precedence)
c.compile().device(provider="quafu", device="Dongling",
                   shots=1024, token="abc...").run()

# 2. In-memory provider-scoped via tq.set_token
tq.set_token("abc...", provider="quafu")

# 3. TyxonQ-namespaced env var (preferred for CI)
# export TYXONQ_QUAFU_TOKEN=abc...

# 4. Upstream's env var (honored for parity with quark.quafu.Task)
# export QPU_API_TOKEN=abc...
```

If none of the above is set, the driver raises a `RuntimeError` pointing
back at the SQCLab portal.

## Listing online chips

```python
import tyxonq as tq
print(tq.api.list_devices(provider="quafu"))
# ['quafu::Dongling', 'quafu::Baihua']
```

Offline chips are filtered out automatically.

## Submitting a circuit (Chain API)

The Chain API's no-arg `.compile()` is a chainable no-op that does NOT
auto-compile to QASM2. For the `quafu` provider you must compile to QASM2
first and then chain the rest:

```python
import tyxonq as tq

c = tq.Circuit(2)
c.h(0).cx(0, 1).measure_z(0).measure_z(1)
c.compile(compile_engine="qiskit", output="qasm2")  # required

results = (
    c.compile()
     .device(provider="quafu", device="Dongling", shots=1024)
     .postprocessing()
     .run()
)
```

The first `c.compile(compile_engine="qiskit", output="qasm2")` populates
`_compiled_source` with a QASM2 string. The chained `.compile()` then sees
the cached source and skips recompilation, letting `.run()` pass the QASM
straight to the driver.

Quafu requires `shots` to be a multiple of 1024. The driver warns rather
than rejects on other values; the server may round or refuse.

## Server-side recompilation

By default TyxonQ tells Quafu *not* to recompile (`compile=True`,
`options.compiler=None` — TyxonQ already lowered to runnable QASM 2.0).
If you hit gates outside a chip's native set, opt into Quafu's recompiler:

```python
c.compile().device(
    provider="quafu", device="Dongling", shots=1024,
    compiler="qsteed",  # or "quarkcircuit" or "qiskit"
).run()
```

Other forwarded options:

| Option | Purpose | Default |
|---|---|---|
| `task_name` | Display name shown in SQCLab | `"TyxonQJob"` |
| `correct` | Readout-error correction | `False` |
| `open_dd` | Dynamical decoupling (`"XY4"` / `"CPMG"`) | `None` |
| `target_qubits` | Sub-list of physical qubits to use | `[]` |

## Quotas and rate limits

Per upstream documentation: up to **1000 task submissions per day per
user**. Excess submissions are queued for the next day. The driver does
not implement a retry loop; surface the server's error message and let
the caller back off.

## Limitations (v1)

- Cloud submission only
- No pulse-level / waveform programming (Quafu's REST API does not
  surface it).
- No chip-topology visualization (would require the optional
  `quarkcircuit` package).
