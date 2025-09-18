import warnings

from tyxonq.numerics.vectorization_checks import safe_for_vectorization, warn_as_error


def _unsafe_fn(x):
    # Simulate an in-place like pattern by emitting a warning
    warnings.warn("AliasWarning: potential alias detected", category=UserWarning)
    return x


def test_safe_for_vectorization_simple():
    assert safe_for_vectorization(lambda x: x, args=(1,), kwargs={}) is True
    assert safe_for_vectorization(_unsafe_fn, args=(1,), kwargs={}) is False


def test_warn_as_error_context():
    with warn_as_error(["AliasWarning"]) as caught:
        try:
            warnings.warn("AliasWarning: test", category=UserWarning)
        except Warning:
            pass
    assert caught["raised"] is True

