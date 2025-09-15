"""Top-level applications package for TyxonQ.

Ensures that subpackages like `tyxonq.applications.chem` are importable
as regular packages (not relying on namespace package semantics), which
stabilizes test collection across environments.
"""


from . import chem

__all__ = [
    "chem",
]


