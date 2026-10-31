"""构建资源的离线校验；不要求 CUDA、MPI 或网络。"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def builder():
    path = Path(__file__).parents[1] / "src/tyxonq/applications/chem/algorithms/sqd/gpu/sbd_ext/build.py"
    spec = importlib.util.spec_from_file_location("sbd_build_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_hash_rejects_modified_input(builder, tmp_path):
    path = tmp_path / "source.h"
    path.write_text("original")
    expected = {"source.h": builder._hash(path)}
    builder._verify_source(tmp_path, expected)
    path.write_text("modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder._verify_source(tmp_path, expected)


def test_cached_source_is_checked_and_does_not_download(builder, tmp_path, monkeypatch):
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "SOURCES.json").write_text(json.dumps({"core_commit": "a"*40}))
    (resources / "nvcc.patch").write_text("local test patch")
    monkeypatch.setattr(builder, "RESOURCES", resources)
    cache = tmp_path / "cache"
    digest = builder._hash(resources / "nvcc.patch")
    source = cache / f"sbd-{'a'*12}-{digest[:12]}"
    source.mkdir(parents=True)
    (source / "test.h").write_text("verified content")
    (source / "TYXONQ_BUILD_SOURCE.json").write_text(json.dumps({
        "core_commit": "a"*40, "patch_sha256": digest,
        "patched_files_sha256": {"test.h": builder._hash(source / "test.h")},
    }))
    monkeypatch.setattr(builder.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected download"))
    assert builder.prepare_source(cache) == source
    (source / "test.h").write_text("changed cached source")
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.prepare_source(cache)


def test_build_arch_validation_precedes_optional_imports(builder, tmp_path):
    with pytest.raises(ValueError, match="sm_80"):
        builder.build(tmp_path, tmp_path, nvcc="unused", mpicxx="unused", arch="invalid")


def test_openmpi_linker_flags_survive_nvcc_forwarding(builder, tmp_path, monkeypatch):
    """nvcc 会拆分 -Xcompiler 中的逗号，链接参数必须单独传递。"""
    (tmp_path / "TYXONQ_BUILD_SOURCE.json").write_text("{}")
    monkeypatch.setattr(builder.importlib, "import_module", lambda name: SimpleNamespace(get_include=lambda: "/test/include"))

    def query(command):
        if command[-1] == "--showme:compile":
            return "-I/test/mpi/include -pthread"
        if command[-1] == "--showme:link":
            return "-pthread -L/test/mpi/lib -Wl,-rpath -Wl,/test/mpi/lib -Wl,--enable-new-dtags -lmpi"
        return "test compiler version"

    monkeypatch.setattr(builder, "_run", query)
    commands = []

    def compile_stub(command, *, check):
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"test extension")

    monkeypatch.setattr(builder.subprocess, "run", compile_stub)
    extension = builder.build(tmp_path, tmp_path, nvcc="nvcc", mpicxx="mpicxx")
    assert extension.is_file()
    command = commands[0]
    assert not any(value.startswith("-Wl") for value in command)
    assert command[command.index("--enable-new-dtags") - 1] == "-Xlinker"
    assert command[command.index("-pthread") - 1] == "-Xcompiler"
    assert "-DSBD_NON_CUDA_AWARE_MPI=1" in command
