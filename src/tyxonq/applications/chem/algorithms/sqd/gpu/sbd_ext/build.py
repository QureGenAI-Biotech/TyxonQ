"""显式准备固定版本源码并构建可选 SBD TPB 扩展，不修改系统环境。"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sysconfig
import tarfile
import tempfile
import urllib.request

RESOURCES = Path(__file__).resolve().parent
MODULE_NAME = "_sbd_tpb_cuda"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(arguments: list[str], **kwargs) -> str:
    return subprocess.check_output(arguments, text=True, **kwargs).strip()


def _verify_source(source: Path, files: dict[str, str]) -> None:
    """对固定源码逐文件校验，拒绝将别的版本冒充已验收版本。"""
    for relative, expected in files.items():
        path = source / relative
        if not path.is_file() or _hash(path) != expected:
            raise ValueError(f"SBD source checksum mismatch: {path}")


def prepare_source(cache_dir: Path, source: Path | None = None) -> Path:
    """在自己的缓存内复制或下载源码；传入的 source 始终只读。"""
    manifest = json.loads((RESOURCES / "SOURCES.json").read_text())
    commit = manifest["core_commit"]
    patch_digest = _hash(RESOURCES / "nvcc.patch")
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"sbd-{commit[:12]}-{patch_digest[:12]}"
    marker = destination / "TYXONQ_BUILD_SOURCE.json"
    if marker.is_file():
        recorded = json.loads(marker.read_text())
        if recorded["core_commit"] != commit or recorded["patch_sha256"] != patch_digest:
            raise ValueError(f"SBD cache provenance mismatch: {destination}")
        _verify_source(destination, recorded["patched_files_sha256"])
        return destination
    if destination.exists():
        raise ValueError(f"Incomplete SBD cache; choose a new cache directory: {destination}")

    with tempfile.TemporaryDirectory(prefix="sbd-prepare-", dir=cache_dir) as temporary:
        temporary = Path(temporary)
        if source is None:
            archive = temporary / "upstream.tar.gz"
            url = f"https://github.com/r-ccs-cms/sbd/archive/{commit}.tar.gz"
            with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            extracted = temporary / "extracted"
            extracted.mkdir()
            # 仅解包普通文件和目录，禁止路径穿越和链接。
            with tarfile.open(archive, "r:gz") as bundle:
                for member in bundle:
                    relative = Path(member.name)
                    if relative.is_absolute() or ".." in relative.parts or not (member.isfile() or member.isdir()):
                        raise ValueError(f"Unsafe SBD source archive entry: {member.name}")
                    target = extracted / relative
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.extractfile(member) as incoming, target.open("wb") as outgoing:
                            shutil.copyfileobj(incoming, outgoing)
            source = extracted / f"sbd-{commit}"
        source = Path(source).resolve()
        # --source 接受源码根目录或其 include 子目录。
        if source.name == "include" and (source / "sbd").is_dir():
            source = source.parent
        _verify_source(source, manifest["files_sha256"])
        prepared = temporary / "prepared"
        prepared.mkdir()
        for relative in manifest["files_sha256"]:
            target = prepared / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
        patch = str(RESOURCES / "nvcc.patch")
        subprocess.run(["git", "apply", "--check", patch], cwd=prepared, check=True)
        subprocess.run(["git", "apply", patch], cwd=prepared, check=True)
        recorded = {
            "core_commit": commit,
            "wrapper_commit": manifest["wrapper_commit"],
            "patch_sha256": patch_digest,
            "patched_files_sha256": {name: _hash(prepared / name) for name in manifest["files_sha256"]},
        }
        (prepared / marker.name).write_text(json.dumps(recorded, indent=2) + "\n")
        prepared.rename(destination)
    return destination


def build(source: Path, output_dir: Path, *, nvcc: str, mpicxx: str, arch: str = "sm_80") -> Path:
    """编译仅在这里导入可选依赖；编译命令与版本写入结果目录。"""
    if not arch.startswith("sm_") or not arch[3:].isdigit():
        raise ValueError("arch must use the form sm_80")
    pybind11 = importlib.import_module("pybind11")
    mpi4py = importlib.import_module("mpi4py")
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = output_dir / (MODULE_NAME + sysconfig.get_config_var("EXT_SUFFIX"))
    compile_flags = shlex.split(_run([mpicxx, "--showme:compile"]))
    link_flags = shlex.split(_run([mpicxx, "--showme:link"]))
    # 当前构建接口明确使用 OpenMPI，避免无意混合其他实现的头文件和库。
    command = [
        nvcc, "-x", "cu", "-std=c++17", f"-arch={arch}",
        "--expt-relaxed-constexpr", "--extended-lambda", "-O2", "-shared",
        "-Xcompiler=-fPIC,-fopenmp", "-DOMPI_SKIP_MPICXX", "-DSBD_THRUST",
        "-DSBD_NON_CUDA_AWARE_MPI=1", f'-DSBD_OFFLOAD_TARGET="cuda:{arch}-nvcc"',
        "-I", pybind11.get_include(), "-I", mpi4py.get_include(),
        "-I", sysconfig.get_path("include"), "-I", str(source / "include"),
    ]
    for flag in compile_flags:
        command.extend([flag] if flag.startswith(("-I", "-D")) else ["-Xcompiler", flag])
    command.append(str(RESOURCES / "tpb.cpp"))
    for flag in link_flags:
        if flag.startswith("-Wl,"):
            for argument in flag[4:].split(","):
                command.extend(["-Xlinker", argument])
        else:
            command.extend([flag] if flag.startswith(("-L", "-l")) else ["-Xcompiler", flag])
        if flag.startswith("-L"):
            command.extend(["-Xlinker", "-rpath", "-Xlinker", flag[2:]])
    command.extend(["-lopenblas", "-o", str(extension)])
    record = {
        "command": command,
        "nvcc_version": _run([nvcc, "--version"]),
        "mpi_version": _run([mpicxx, "--showme:version"]),
        "source": str(source), "binding_sha256": _hash(RESOURCES / "tpb.cpp"),
        "spin_kernel_sha256": _hash(RESOURCES / "spin_kernels.cuh"),
        "provenance": json.loads((source / "TYXONQ_BUILD_SOURCE.json").read_text()),
    }
    (output_dir / "sbd_build.json").write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run(command, check=True)
    record["extension_sha256"] = _hash(extension)
    (output_dir / "sbd_build.json").write_text(json.dumps(record, indent=2) + "\n")
    return extension


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="existing pristine pinned source root or include directory")
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tyxonq" / "sbd")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nvcc", default=shutil.which("nvcc") or "nvcc")
    parser.add_argument("--mpicxx", default=shutil.which("mpicxx") or "mpicxx")
    parser.add_argument("--arch", default="sm_80")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    source = prepare_source(args.cache_dir.resolve(), args.source)
    if args.prepare_only:
        print(source)
    else:
        print(build(source, args.output_dir.resolve(), nvcc=args.nvcc, mpicxx=args.mpicxx, arch=args.arch))


if __name__ == "__main__":
    main()
