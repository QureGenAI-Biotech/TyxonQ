from . import api as api

# Back-compat: expose `apis` as an alias to `api`
apis = api

__all__ = [
    "api",
    "apis",
]
