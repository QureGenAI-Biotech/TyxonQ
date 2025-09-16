from tyxonq.plugins.registry import discover, get_compiler, get_device


def test_discover_returns_mapping():
    mapping = discover("tyxonq.devices")
    assert isinstance(mapping, dict)


class _X:
    def __call__(self):
        return "ok"


def test_get_by_path_and_cache_monotonic():
    # Use registry itself as a loadable object for a stable path
    obj1 = get_compiler("tyxonq.plugins.registry:_X")
    obj2 = get_compiler("tyxonq.plugins.registry:_X")
    assert obj1 is obj2


