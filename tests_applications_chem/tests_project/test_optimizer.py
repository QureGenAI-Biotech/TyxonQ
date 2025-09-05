
import numpy as np
from scipy.optimize import minimize

from tyxonq.applications.chem import UCCSD
from tyxonq.applications.chem.molecule import h4
from tyxonq.libs.optimizer import soap


def test_optimizer():
    # see also example/custom_optimizer.py
    ucc = UCCSD(h4)
    ucc.kernel()

    opt_res = minimize(ucc.energy, ucc.init_guess, method=soap)
    assert np.allclose(opt_res.fun, ucc.e_ucc)
