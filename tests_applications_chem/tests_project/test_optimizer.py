
import numpy as np
from scipy.optimize import minimize

from tyxonq.applications.chem import UCCSD
from tyxonq.applications.chem.molecule import h4
from tyxonq.libs.optimizer import soap


def test_optimizer():
    # see also example/custom_optimizer.py
    ucc = UCCSD(h4)
    ucc.kernel()

    # 统一为解析路径（模拟器 shots=0），避免与 ucc.kernel() 的能量定义不一致
    opt_res = minimize(lambda v: ucc.energy(v, shots=0, provider="simulator", device="statevector"), ucc.init_guess, method=soap)
    assert np.allclose(opt_res.fun, ucc.e_ucc)
