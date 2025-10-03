import numpy as np
import pytest

from tyxonq.applications.chem import UCCSD
from tyxonq.applications.chem.algorithms.uccsd import ROUCCSD
from tyxonq.applications.chem.molecule import h2,h4
import numpy as np
import pytest
from pyscf import M

from tyxonq.applications.chem import UCCSD
from tyxonq.applications.chem.runtimes.ucc_numeric_runtime import apply_excitation
from tyxonq.applications.chem.chem_libs.quantum_chem_library.ci_state_mapping import get_init_civector
#wrong version
# from tyxonq.libs.circuits_library.qubit_state_preparation import get_init_circuit
from tyxonq.applications.chem.chem_libs.quantum_chem_library.statevector_ops import get_init_circuit
import tyxonq as tq


@pytest.fixture
def ref_eg():
    # 使用 pyscf 作为金标准参考（电子能口径）
    try:
        tq.set_backend("numpy")
    except Exception:
        pytest.xfail("Backend numpy not available")
    uccsd = UCCSD(h2)
    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5
    e, g = uccsd.energy_and_grad(params, runtime="numeric", numeric_engine='pyscf')
    return float(e), np.asarray(g, dtype=np.float64)

@pytest.mark.parametrize("numeric_engine", ['statevector',"civector", "civector-large", "pyscf"])
def test_gradient(ref_eg, numeric_engine):
    # 统一 backend，验证所有 numeric_engine 与 statevector 参考一致
    e_ref, g_ref = ref_eg
    try:
        tq.set_backend("numpy")
    except Exception:
        pytest.xfail("Backend numpy not available")
    uccsd = UCCSD(h2)
    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5
    e_num, g_num = uccsd.energy_and_grad(params, runtime="numeric", numeric_engine=numeric_engine)
    np.testing.assert_allclose(e_num, e_ref, atol=1e-5)
    np.testing.assert_allclose(g_num, g_ref, atol=1e-5)


@pytest.fixture
def ref_state():
    uccsd = UCCSD(h2)

    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5
    
    # 需要确保 backend 已设置
    try:
        tq.set_backend("numpy")
    except Exception:
        pytest.xfail("Backend numpy not available")
    # numeric_engine = "statevector"
    # state = uccsd.civector(params, numeric_engine=numeric_engine)
    # state = apply_excitation(state, n_qubits=8, n_elec_s=4, ex_op=(4, 0, 3, 7), mode="fermion", numeric_engine=numeric_engine)
    # 使用 pyscf 基准 H2
    numeric_engine_ref = "pyscf"
    state = uccsd.civector(params, numeric_engine=numeric_engine_ref)
    state = apply_excitation(state, n_qubits=uccsd.n_qubits, n_elec_s=uccsd.n_elec_s, ex_op=uccsd.ex_ops[0], mode="fermion", numeric_engine=numeric_engine_ref)

    # state = np.asarray([ 0.2208986 ,  0.09394656, -0.        , -0.        ,  0.        ,
    #     0.        ,  0.09394656,  0.27237773, -0.        , -0.        ,
    #     0.        ,  0.        , -0.        , -0.        ,  0.        ,
    #     0.        , -0.        , -0.        , -0.        , -0.        ,
    #     0.        ,  0.        , -0.        , -0.        ,  0.        ,
    #     0.        , -0.        , -0.        , -0.45198544, -0.10137563,
    #     0.        ,  0.        , -0.        , -0.        , -0.10137563,
    #    -0.1918837 ])
    
    return state


@pytest.mark.parametrize("numeric_engine", ["statevector", "civector", "civector-large", "pyscf"])
def test_excitation(ref_state, numeric_engine):
    uccsd = UCCSD(h2)

    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5
        
    state = uccsd.civector(params, numeric_engine=numeric_engine)
    state = apply_excitation(state,n_qubits=uccsd.n_qubits, n_elec_s=uccsd.n_elec_s, ex_op=uccsd.ex_ops[0],mode="fermion", numeric_engine=numeric_engine)
    np.testing.assert_allclose(state, ref_state, atol=1e-6)


def test_device_matches_numeric_gradient_single():
    # 设备路径与数值路径梯度一致（单次对照，避免重复运行）
    try:
        tq.set_backend("numpy")
    except Exception:
        pytest.xfail("Backend numpy not available")
    uccsd = UCCSD(h2)
    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5
    # numeric 基准选 statevector（与 device/statevector 完全一致）
    e_num, g_num = uccsd.energy_and_grad(params, runtime="numeric", numeric_engine="statevector")
    e_dev, g_dev = uccsd.energy_and_grad(params, runtime="device", provider="simulator", device="statevector", shots=4086)
    np.testing.assert_allclose(e_dev, e_num, atol=1e-5)
    np.testing.assert_allclose(g_dev, g_num, atol=1e-5)


def test_device_matches_numeric_energy_single():
    # 设备路径与数值路径电子能一致（单次对照，避免重复运行）
    try:
        tq.set_backend("numpy")
    except Exception:
        pytest.xfail("Backend numpy not available")
    uccsd = UCCSD(h2)
    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5
    e_num = uccsd.energy(params, runtime="numeric", numeric_engine="statevector")
    e_dev = uccsd.energy(params, runtime="device", provider="simulator", device="statevector", shots=0)
    np.testing.assert_allclose(e_dev, e_num, atol=1e-7)



@pytest.mark.parametrize("numeric_engine", ["statevector",])
@pytest.mark.parametrize("backend_str", ["numpy"])
@pytest.mark.parametrize("init_state", [None])
@pytest.mark.parametrize("mode", ["fermion"])
def test_gradient_signle_opt(backend_str, numeric_engine, init_state, mode):
    if numeric_engine in ["pyscf"] and mode in ["qubit"]:
        pytest.xfail("Incompatible numeric_engine and fermion symmetry")
    prev = getattr(tq, "backend", None)
    try:
        try:
            tq.set_backend(backend_str)
        except Exception:
            pytest.xfail(f"Backend {backend_str} not available")
        uccsd = UCCSD(h2, mode=mode, numeric_engine=numeric_engine,run_fci=True)
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
    if numeric_engine in ["pyscf"] and mode in ["qubit"]:
        pytest.xfail("Incompatible numeric_engine and fermion symmetry")
    prev = getattr(tq, "backend", None)
    try:
        tq.set_backend(backend_str)
    except Exception:
        pytest.xfail(f"Backend {backend_str} not available")
    uccsd = UCCSD(h4, runtime = 'numeric',mode=mode, numeric_engine=numeric_engine,run_fci=True)
    # test initial condition. Has no effect
    if init_state == "civector" and hasattr(uccsd, "civector_size"):
        uccsd.init_state = get_init_civector(uccsd.civector_size)
    elif init_state == "circuit" and hasattr(uccsd, "n_elec"):
        uccsd.init_state = get_init_circuit(uccsd.n_qubits, uccsd.n_elec, uccsd.mode)
    e = uccsd.kernel()
    print('='*30)
    print('numeric_engine =', numeric_engine)
    print('e =', e)
    print('e_fci =', uccsd.e_fci)
    np.testing.assert_allclose(e, uccsd.e_fci, atol=1e-4)


# @pytest.mark.parametrize("numeric_engine", ["statevector", "civector", "civector-large", "pyscf"])
# def test_open_shell(numeric_engine):

#     m = M(atom=[["O", 0, 0, 0], ["O", 0, 0, 1]], spin=2)
#     active_space = (6, 4)

#     uccsd = ROUCCSD(m, active_space=active_space, numeric_engine=numeric_engine,run_fci=True)
#     uccsd.kernel(shots=0)
#     np.testing.assert_allclose(uccsd.e_ucc, uccsd.e_fci, atol=1e-4)


def test_device_kernel_matches_fci():
    u = UCCSD(h2,run_fci=True)
    # 正统调用：直接 kernel（设备路径，statevector 精确模拟，shots=0）
    e = u.kernel(runtime="device", provider="simulator", shots=0)
    np.testing.assert_allclose(e, u.e_fci, atol=1e-5)


def test_device_energy_matches_numeric_statevector():
    u = UCCSD(h2)
    # 使用零初值参数，比较设备路径与数值路径在相同参数下的一致性
    params = np.asarray(u.init_guess if hasattr(u, "init_guess") else np.zeros(0), dtype=np.float64)
    e_num = u.energy(params, runtime="numeric", numeric_engine="statevector")
    e_dev = u.energy(params, runtime="device", shots= 8192)
    
    print('='*30)
    print('e_dev = ',e_dev)
    print('e_num = ',e_num)
    print('='*30)

    np.testing.assert_allclose(e_dev, e_num, atol=1e-7)


def test_device_energy_counts_matches_numeric_tolerantly():
    u = UCCSD(h2)
    params = np.asarray(u.init_guess if hasattr(u, "init_guess") else np.zeros(0), dtype=np.float64)
    # counts 路径（shots>0），允许抽样误差
    shots = 8192
    e_dev = u.energy(params, runtime="device", provider="simulator", device="statevector", shots=shots)
    e_num = u.energy(params, runtime="numeric", numeric_engine="statevector")
    np.testing.assert_allclose(e_dev, e_num, atol=5e-2)


def test_device_counts_converges_to_pyscf():
    # shots>0 的设备路径应当随 shots 增大逐步接近 pyscf 数值解析电子能
    uccsd = UCCSD(h2)
    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5
    # pyscf 作为金标准（电子能）
    e_ref = uccsd.energy(params, runtime="numeric", numeric_engine="pyscf")
    shots_list = [128, 512, 2048, 8192]
    for s in shots_list:
        e_dev = uccsd.energy(params, runtime="device", provider="simulator", device="statevector", shots=s)
        print(e_dev)
        # 理论抽样误差 ~ O(1/sqrt(shots))，给出宽松但递减的阈值
        tol = 2.0 / np.sqrt(s) + 0.02
        np.testing.assert_allclose(e_dev, e_ref, atol=tol)


def test_device_counts_gradient_converges_to_pyscf():
    # shots>0 的设备路径梯度应当随 shots 增大逐步接近 pyscf 数值解析梯度
    uccsd = UCCSD(h2)
    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5
    # pyscf 作为金标准（电子能与梯度）
    e_ref, g_ref = uccsd.energy_and_grad(params, runtime="numeric", numeric_engine="pyscf")
    shots_list = [512, 2048, 8192,10640]
    errs = []
    for s in shots_list:
        e_dev, g_dev = uccsd.energy_and_grad(params, runtime="device", provider="simulator", device="statevector", shots=s)
        print('e_dev = ',e_dev)
        print('g_dev = ',g_dev)
        print('='*15)
        # 记录 L2 误差，观察随 shots 增大是否下降
        errs.append(float(np.linalg.norm(np.asarray(g_dev) - np.asarray(g_ref))))
    # 要求总体误差下降（最后一个不大于第一个），允许中间波动
    assert errs[-1] <= errs[0]



def test_ucc_rdm_gold_standard_h2():
    # 对齐 TCC 金标准：H2 的 UCCSD 在 MO 基的 1RDM/2RDM
    # 使用解析优化通道（shots=0）确保无采样偏差
    uccsd = UCCSD(h2)
    e = uccsd.kernel(runtime="device", provider="simulator", device="statevector", shots=0)
    # 生成 RDM（使用优化后参数）
    rdm1 = uccsd.make_rdm1(basis="MO")
    rdm2 = uccsd.make_rdm2(basis="MO")
    
    print('rdm1 = ',rdm1)
    print('rdm2 = ',rdm2)
    # TCC 金标准
    rdm1_gold = np.array([[1.97457654e+00, -2.00371787e-16],
                          [-2.00371787e-16, 2.54234643e-02]], dtype=np.float64)
    rdm2_gold = np.array(
        [[[[1.97457654e+00, -2.26018011e-16],
           [-2.26018011e-16, 2.58709350e-32]],

          [[-2.26018011e-16, -2.24054851e-01],
           [0.00000000e+00, 2.56462238e-17]]],


         [[[-2.26018011e-16, 0.00000000e+00],
           [-2.24054851e-01, 2.56462238e-17]],

          [[2.58709350e-32, 2.56462238e-17],
           [2.56462238e-17, 2.54234643e-02]]]], dtype=np.float64)


    np.testing.assert_allclose(rdm1, rdm1_gold, atol=1e-6)
    np.testing.assert_allclose(rdm2, rdm2_gold, atol=1e-6)

    e2 = uccsd.kernel(runtime="numeric")
    # 生成 RDM（使用优化后参数）
    rdm1 = uccsd.make_rdm1(basis="MO")
    rdm2 = uccsd.make_rdm2(basis="MO")
    np.testing.assert_allclose(rdm1, rdm1_gold, atol=1e-6)
    np.testing.assert_allclose(rdm2, rdm2_gold, atol=1e-6)

if __name__ == "__main__":
    # test_device_energy_matches_numeric_statevector()
    # test_device_energy_counts_matches_numeric_tolerantly()
    # test_gradient_opt('numpy','pyscf','circuit','fermion')
    test_gradient_opt('pytorch','civector' , 'civector', 'qubit')