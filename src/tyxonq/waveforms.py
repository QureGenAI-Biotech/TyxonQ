from typing import List, Union
from dataclasses import dataclass

ParamType = Union[float, str]

__all__ = [
    "Gaussian",
    "GaussianSquare",
    "Drag",
    "Constant",
    "Sine",
    "Cosine",
    "CosineDrag",
    "Flattop",
    "Hermite",
    "BlackmanSquare",
]

@dataclass
class Gaussian:
    amp: ParamType
    duration: int
    sigma: ParamType
    phase: ParamType = 0.0  # Phase offset (radians), default 0 for backward compatibility
    def qasm_name(self) -> str:
        return "gaussian"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.duration, self.sigma, self.phase]

@dataclass
class GaussianSquare:
    amp: ParamType
    duration: int
    sigma: ParamType
    width: ParamType
    def qasm_name(self) -> str:
        return "gaussian_square"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.duration, self.sigma, self.width]

@dataclass
class Drag:
    amp: ParamType
    duration: int
    sigma: ParamType
    beta: ParamType
    phase: ParamType = 0.0  # Phase offset (radians), default 0 for backward compatibility
    def qasm_name(self) -> str:
        return "drag"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.duration, self.sigma, self.beta, self.phase]

@dataclass
class Constant:
    amp: ParamType
    duration: int
    def qasm_name(self) -> str:
        return "constant"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.duration]

@dataclass
class Sine:
    amp: ParamType
    frequency: ParamType
    duration: int
    def qasm_name(self) -> str:
        return "sine"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.frequency, self.duration]

@dataclass
class Cosine:
    amp: ParamType
    frequency: ParamType
    duration: int
    def qasm_name(self) -> str:
        return "cosine"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.frequency, self.duration]

@dataclass
class CosineDrag:
    amp: ParamType
    duration: int
    phase: ParamType
    alpha: ParamType
    def qasm_name(self) -> str:
        return "cosine_drag"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.duration, self.phase, self.alpha]

@dataclass
class Flattop:
    amp: ParamType
    width: ParamType
    duration: int
    def qasm_name(self) -> str:
        return "flattop"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.width, self.duration]

@dataclass
class Hermite:
    """Hermite polynomial envelope waveform.
    
    Uses probabilist's Hermite polynomials to create a smooth envelope
    with minimal frequency spectral leakage. Particularly useful for
    high-fidelity quantum gates.
    
    Attributes:
        amp: Pulse amplitude
        duration: Pulse duration in nanoseconds
        order: Order of Hermite polynomial (typically 2 or 3)
        phase: Phase offset in radians (default 0.0)
    """
    amp: ParamType
    duration: int
    order: ParamType = 2
    phase: ParamType = 0.0
    
    def qasm_name(self) -> str:
        return "hermite"
    
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.duration, self.order, self.phase]

@dataclass
class BlackmanSquare:
    """Blackman window with flat-top envelope.
    
    Combines Blackman window (excellent frequency domain properties)
    with a flat-top plateau for high-precision quantum gates.
    The Blackman window provides extremely low side-lobe levels,
    reducing spectral leakage and crosstalk.
    
    Attributes:
        amp: Pulse amplitude
        duration: Pulse duration in nanoseconds
        width: Flat-top plateau width (must be < duration)
        phase: Phase offset in radians (default 0.0)
    
    Notes:
        The pulse shape is: Blackman ramp-up → flat plateau → Blackman ramp-down
        Typical width: 0.5-0.8 of duration
    """
    amp: ParamType
    duration: int
    width: ParamType
    phase: ParamType = 0.0
    
    def qasm_name(self) -> str:
        return "blackman_square"
    
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.duration, self.width, self.phase]
