"""Compiler interfaces and stages."""

from .api import CompileResult, Pass

__all__ = ["CompileResult", "Pass"]

# Legacy imports disabled in refactor to avoid side effects
