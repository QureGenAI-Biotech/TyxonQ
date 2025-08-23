"""TyxonQ package initializer (refactor minimal, side-effect free).

This initializer intentionally avoids importing legacy modules with heavy side
effects. Submodules should be imported explicitly, e.g. `from tyxonq.core.ir import Circuit`.
"""

__version__ = "0.5.0"
__author__ = "TyxonQ Authors"

__all__ = []
