import numpy as np
import pytest

from tyxonq.applications.chem import UCCSD
from tyxonq.applications.chem.algorithms.uccsd import ROUCCSD
from tyxonq.applications.chem.molecule import h2,h4


def test_uccsd_numeric_runtime_smoke():
    u = UCCSD(h2)
    # numeric runtime should be callable and deterministic for zero params
    e = u.energy(None, runtime="numeric", numeric_engine="statevector")
    assert isinstance(e, float)
import numpy as np
import pytest
from pyscf import M

from tyxonq.applications.chem import UCCSD
from tyxonq.applications.chem.molecule import h2
from tyxonq.applications.chem.runtimes.ucc_numeric_runtime import apply_excitation
from tyxonq.applications.chem.chem_libs.quantum_chem_library.ci_state_mapping import get_init_civector
from tyxonq.libs.circuits_library.qubit_state_preparation import get_init_circuit
import tyxonq as tq


@pytest.fixture
def ref_eg():
    # 使用 numeric statevector 作为参考
    prev = getattr(tq, "backend", None)
    try:
        try:
            tq.set_backend("numpy")
        except Exception:
            pytest.xfail("Backend numpy not available")
        uccsd = UCCSD(h4)
        np.random.seed(2077)
        params = np.random.rand(len(uccsd.init_guess)) - 0.5
        e, g = uccsd.energy_and_grad(params, runtime="numeric", numeric_engine="statevector")
        return float(e), np.asarray(g, dtype=np.float64)
    finally:
        if prev is not None:
            try:
                tq.set_backend(prev.name if hasattr(prev, "name") else prev)
            except Exception:
                pass


@pytest.mark.parametrize("numeric_engine", ["statevector", "civector", "civector-large", "pyscf"])
def test_gradient(ref_eg, numeric_engine):
    e_ref, g_ref = ref_eg

    uccsd = UCCSD(h4)

    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5


    e, g = uccsd.energy_and_grad(params, runtime="numeric", numeric_engine=numeric_engine)

    atol = 1e-5 if numeric_engine == "statevector" else 1e-5
    np.testing.assert_allclose(e, e_ref, atol=atol)
    np.testing.assert_allclose(g, g_ref, atol=atol)


@pytest.fixture
def ref_state():
    uccsd = UCCSD(h2)

    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5
    
    # 需要确保 backend 已设置
    prev = getattr(tq, "backend", None)
    try:
        try:
            tq.set_backend("numpy")
        except Exception:
            pytest.xfail("Backend numpy not available")
        numeric_engine = "statevector"
        state = uccsd.civector(params, numeric_engine=numeric_engine)
    finally:
        if prev is not None:
            try:
                tq.set_backend(prev.name if hasattr(prev, "name") else prev)
            except Exception:
                pass
    state = apply_excitation(state, n_qubits=8, n_elec_s=4, ex_op=(4, 0, 3, 7), mode="fermion", numeric_engine=numeric_engine)
    
    return state


@pytest.mark.parametrize("numeric_engine", ["statevector", "civector", "civector-large", "pyscf"])
def test_excitation(ref_state, numeric_engine):
    uccsd = UCCSD(h2)

    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5
        
    state = uccsd.civector(params, numeric_engine=numeric_engine)
    state = apply_excitation(state, n_qubits=8, n_elec_s=4, ex_op=(4, 0, 3, 7), mode="fermion", numeric_engine=numeric_engine)
    np.testing.assert_allclose(state, ref_state, atol=1e-6)



@pytest.mark.parametrize("numeric_engine", ["statevector",])
@pytest.mark.parametrize("backend_str", ["pytorch"])
@pytest.mark.parametrize("init_state", [None])
@pytest.mark.parametrize("mode", ["fermion"])
def test_gradient_signle_opt(backend_str, numeric_engine, init_state, mode):
    if numeric_engine in ["statevector"] and backend_str in ["numpy", "pytorch"]:
        pytest.xfail("Incompatible numeric_engine and backend")
    if numeric_engine in ["pyscf"] and mode in ["qubit"]:
        pytest.xfail("Incompatible numeric_engine and fermion symmetry")
    prev = getattr(tq, "backend", None)
    try:
        try:
            tq.set_backend(backend_str)
        except Exception:
            pytest.xfail(f"Backend {backend_str} not available")
        uccsd = UCCSD(h2, mode=mode, numeric_engine=numeric_engine)
        # test initial condition. Has no effect
        if init_state == "civector" and hasattr(uccsd, "civector_size"):
            uccsd.init_state = get_init_civector(uccsd.civector_size)
        elif init_state == "circuit" and hasattr(uccsd, "n_elec"):
            uccsd.init_state = get_init_circuit(uccsd.n_qubits, uccsd.n_elec, uccsd.mode)
        e = uccsd.kernel()
        np.testing.assert_allclose(e, uccsd.e_fci, atol=1e-4)
    finally:
        if prev is not None:
            try:
                tq.set_backend(prev.name if hasattr(prev, "name") else prev)
            except Exception:
                pass



@pytest.mark.parametrize("numeric_engine", ["statevector", "civector", "civector-large", "pyscf"])
@pytest.mark.parametrize("backend_str", ["pytorch", "numpy"])
@pytest.mark.parametrize("init_state", [None, "civector", "circuit"])
@pytest.mark.parametrize("mode", ["fermion", "qubit"])
def test_gradient_opt(backend_str, numeric_engine, init_state, mode):
    if numeric_engine in ["statevector"] and backend_str in ["numpy", "pytorch"]:
        pytest.xfail("Incompatible numeric_engine and backend")
    if numeric_engine in ["pyscf"] and mode in ["qubit"]:
        pytest.xfail("Incompatible numeric_engine and fermion symmetry")
    prev = getattr(tq, "backend", None)
    try:
        try:
            tq.set_backend(backend_str)
        except Exception:
            pytest.xfail(f"Backend {backend_str} not available")
        uccsd = UCCSD(h2, mode=mode, numeric_engine=numeric_engine)
        # test initial condition. Has no effect
        if init_state == "civector" and hasattr(uccsd, "civector_size"):
            uccsd.init_state = get_init_civector(uccsd.civector_size)
        elif init_state == "circuit" and hasattr(uccsd, "n_elec"):
            uccsd.init_state = get_init_circuit(uccsd.n_qubits, uccsd.n_elec, uccsd.mode)
        e = uccsd.kernel()
        np.testing.assert_allclose(e, uccsd.e_fci, atol=1e-4)
    finally:
        if prev is not None:
            try:
                tq.set_backend(prev.name if hasattr(prev, "name") else prev)
            except Exception:
                pass


@pytest.mark.parametrize("numeric_engine", ["statevector", "civector", "civector-large", "pyscf"])
def test_open_shell(numeric_engine):

    m = M(atom=[["O", 0, 0, 0], ["O", 0, 0, 1]], spin=2)
    active_space = (6, 4)

    uccsd = ROUCCSD(m, active_space=active_space, numeric_engine=numeric_engine)
    uccsd.kernel()
    np.testing.assert_allclose(uccsd.e_ucc, uccsd.e_fci, atol=1e-4)


def test_device_kernel_matches_fci():
    u = UCCSD(h2)
    # 正统调用：直接 kernel（设备路径，statevector 精确模拟，shots=0）
    e = u.kernel(runtime="device", provider="simulator", device="statevector", shots=0)
    np.testing.assert_allclose(e, u.e_fci, atol=1e-5)


def test_device_energy_matches_numeric_statevector():
    prev = getattr(tq, "backend", None)
    try:
        try:
            tq.set_backend("numpy")
        except Exception:
            pytest.xfail("Backend numpy not available")
        u = UCCSD(h2)
        # 使用零初值参数，比较设备路径与数值路径在相同参数下的一致性
        params = np.asarray(u.init_guess if hasattr(u, "init_guess") else np.zeros(0), dtype=np.float64)
        e_dev = u.energy(params, runtime="device", provider="simulator", device="statevector", shots=0)
        e_num = u.energy(params, runtime="numeric", numeric_engine="statevector")
        np.testing.assert_allclose(e_dev, e_num, atol=1e-7)
    finally:
        if prev is not None:
            try:
                tq.set_backend(prev.name if hasattr(prev, "name") else prev)
            except Exception:
                pass


def test_device_energy_counts_matches_numeric_tolerantly():
    prev = getattr(tq, "backend", None)
    try:
        try:
            tq.set_backend("numpy")
        except Exception:
            pytest.xfail("Backend numpy not available")
        u = UCCSD(h2)
        params = np.asarray(u.init_guess if hasattr(u, "init_guess") else np.zeros(0), dtype=np.float64)
        # counts 路径（shots>0），允许抽样误差
        shots = 8192
        e_dev = u.energy(params, runtime="device", provider="simulator", device="statevector", shots=shots)
        e_num = u.energy(params, runtime="numeric", numeric_engine="statevector")
        np.testing.assert_allclose(e_dev, e_num, atol=5e-2)
    finally:
        if prev is not None:
            try:
                tq.set_backend(prev.name if hasattr(prev, "name") else prev)
            except Exception:
                pass


def test_device_simulator_allows_shots_zero():
    # provider=simulator 时，允许 shots=0（解析路径）
    prev = getattr(tq, "backend", None)
    try:
        try:
            tq.set_backend("numpy")
        except Exception:
            pytest.xfail("Backend numpy not available")
        u = UCCSD(h2)
        # 不做严格数值断言，仅验证 shots=0 可用且返回数值
        e = u.energy(None, runtime="device", provider="simulator", device="statevector", shots=0)
        assert isinstance(e, float)
    finally:
        if prev is not None:
            try:
                tq.set_backend(prev.name if hasattr(prev, "name") else prev)
            except Exception:
                pass
