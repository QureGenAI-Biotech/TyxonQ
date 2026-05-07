"""Guards the vendored Task class so re-syncs from upstream don't silently drop
methods we depend on. If upstream renames a method, this test catches it before
the driver breaks at runtime.
"""
from __future__ import annotations


EXPECTED_PUBLIC_METHODS = {
    "verify",
    "status",
    "run",
    "result",
    "cancel",
    "query",
    "delete",
    "request",
}


def test_vendor_task_public_method_set():
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    public = {n for n in dir(Task) if not n.startswith("_")}
    methods = {n for n in public if callable(getattr(Task, n))}
    missing = EXPECTED_PUBLIC_METHODS - methods
    assert not missing, f"Vendored Task is missing expected methods: {missing}"


def test_vendor_task_url_constant():
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    assert Task.URL == "https://quafu-sqc.baqis.ac.cn"


def test_vendor_task_no_backend_method():
    """We deliberately strip backend() because it depends on the optional
    quarkcircuit package which we do not vendor."""
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    assert not hasattr(Task, "backend")
