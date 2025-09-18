from __future__ import annotations

import contextlib
import warnings
from typing import Any, Callable, Dict, Iterable, Tuple


def safe_for_vectorization(fn: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
    """Heuristic check whether a function is likely safe to vectorize.

    Current heuristic: functions that emit warnings containing specific
    substrings (e.g., alias) are considered unsafe.
    """

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            fn(*args, **kwargs)
        except Exception:
            # If function raises, we conservatively mark as unsafe for vmap
            return False
    for w in rec:
        msg = str(w.message)
        if any(key in msg for key in ["AliasWarning", "CloneRequiredWarning", "in-place"]):
            return False
    return True


@contextlib.contextmanager
def warn_as_error(keywords: Iterable[str]):
    """Context that converts matching warnings to exceptions and records if raised.

    Yields a dict with 'raised' flag.
    """

    caught = {"raised": False}

    def _showwarning(message, category, filename, lineno, file=None, line=None):  # type: ignore[override]
        text = str(message)
        if any(k in text for k in keywords):
            caught["raised"] = True
            raise Warning(text)
        return _orig_showwarning(message, category, filename, lineno, file=file, line=line)

    _orig_showwarning = warnings.showwarning
    warnings.showwarning = _showwarning  # type: ignore[assignment]
    try:
        yield caught
    finally:
        warnings.showwarning = _orig_showwarning  # type: ignore[assignment]


__all__ = ["safe_for_vectorization", "warn_as_error"]


