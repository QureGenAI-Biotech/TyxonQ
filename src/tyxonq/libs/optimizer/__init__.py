from __future__ import annotations

# Re-export callable optimizer entrypoints for convenient imports like:
#   from tyxonq.libs.optimizer import soap
from .soap import soap as soap

__all__ = ["soap"]


