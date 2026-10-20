"""提供自旋链模型、基态、实时响应、谱分析和近似量子编译接口。"""

from .models import SpinChainModel
from .ground_state import GroundStateOptions, GroundStateResult, prepare_ground_state
from .circuits import build_evolution
from .aqc import AQCOptions, AQCResult, compress_dynamics
from .spectra import compute_dsf, response_from_kicks, spectrum_for_plot
from .workflow import SpinDynamicsResult, run_spin_dynamics

__all__ = ["SpinChainModel", "GroundStateOptions", "GroundStateResult", "prepare_ground_state",
           "build_evolution", "AQCOptions", "AQCResult", "compress_dynamics", "compute_dsf",
           "response_from_kicks", "spectrum_for_plot", "SpinDynamicsResult", "run_spin_dynamics"]
