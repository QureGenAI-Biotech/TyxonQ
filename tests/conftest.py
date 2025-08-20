import sys
import os
import pytest

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tyxonq as tq


@pytest.fixture(scope="function")
def npb():
    tq.set_backend("numpy")
    yield
    tq.set_backend("numpy")  # default backend


@pytest.fixture(scope="function")
def torchb():
    try:
        tq.set_backend("pytorch")
        yield
        tq.set_backend("numpy")
    except ImportError as e:
        print(e)
        tq.set_backend("numpy")
        pytest.skip("****** No torch backend found, skipping test suit *******")


@pytest.fixture(scope="function")
def cpb():
    try:
        tq.set_backend("cupy")
        yield
        tq.set_backend("numpy")
    except ImportError as e:
        print(e)
        tq.set_backend("numpy")
        pytest.skip("****** No cupy backend found, skipping test suit *******")


@pytest.fixture(scope="function")
def highp():
    tq.set_dtype("complex128")
    yield
    tq.set_dtype("complex64")
