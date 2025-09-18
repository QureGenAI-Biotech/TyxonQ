class TyxonQError(Exception):
    """Base exception for TyxonQ."""


class CompilationError(TyxonQError):
    """Raised when compilation fails or produces invalid output."""


class DeviceExecutionError(TyxonQError):
    """Raised when device execution fails (timeouts, connectivity, etc.)."""


class VectorizationFallbackWarning(Warning):
    """Warning indicating vectorization was disabled or fell back to eager."""


