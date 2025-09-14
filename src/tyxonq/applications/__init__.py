"""Top-level applications package for TyxonQ.

Ensures that subpackages like `tyxonq.applications.chem` are importable
as regular packages (not relying on namespace package semantics), which
stabilizes test collection across environments.
"""

# Optionally expose common subpackages for convenience
try:
    from . import chem  # noqa: F401
except Exception:
    pass

__all__ = [
    "chem",
]


