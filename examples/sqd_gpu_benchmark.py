"""固定组态空间的 Torch、原生 SBD 和混合 GPU 求解基准。

父进程准备相同积分、组态与初猜，每组测量一次冷启动和默认五次稳态；
不执行量子采样或 PCM 外循环。--help 和 --summarize 不要求 CUDA/MPI。

SBD 使用前需显式构建；查看构建参数：
    python -m tyxonq.applications.chem.algorithms.sqd.gpu.sbd_ext.build --help
从仓库根目录运行（实际求解前选择空闲 GPU）：
    export PYTHONPATH=src
    python examples/sqd_gpu_benchmark.py --case lih --backend torch native hybrid --devices 0 --validation full_cpu --output /tmp/sqd_single.json
    python examples/sqd_gpu_benchmark.py --case n2 --backend hybrid --devices 0 1 --batches 4 --validation full_cpu --output /tmp/sqd_batches.json
    python examples/sqd_gpu_benchmark.py --case n2 --backend native --devices 0 1 --mpi-grid 2 1 --mpiexec /path/to/matching/mpiexec --output /tmp/sqd_mpi.json
    python examples/sqd_gpu_benchmark.py --case all --prepare-only --output /tmp/sqd_inputs.json
    python examples/sqd_gpu_benchmark.py --summarize /tmp/sqd_single.json /tmp/sqd_batches.json --output /tmp/sqd_summary.json

支持实数 restricted 积分及固定 α/β 电子数；返回物理电子能，不含核排斥、
冻结核等外部常数或自旋惩罚贡献。Torch/Hybrid 使用本方 Davidson 并支持
--spin-sq；Native 使用 SBD Davidson，不支持自旋惩罚。--spin-backend cuda_fused
可显式选择融合 S² 内核。Torch 默认 CPU RDM，另两条路径使用 SBD RDM。

full_cpu 每次独立复核算符、残差、RDM 和能量；gpu 只做所选 GPU 残差及
密度/自旋一致性检查，不表示执行了当次独立 CPU 复核。性能比较必须使用
相同输入、初猜、精度及校验模式。--reference 控制额外的固定空间最低根对照，
不替代求解器自身验收。默认初猜是排序后的第一个行列式。

--devices 指定物理卡，入口检查占用并等待，超时拒绝启动，不终止已有进程。
独立批次为一进程一卡；MPI 仅用于 Native 单任务，双/四卡网格为 2×1/2×2，
本入口自行启动 mpiexec，不要外套 launcher。启动器、mpi4py 与扩展必须使用
同一套 MPI 库；当前 MPI 经 CPU 中转，单任务加速比与批吞吐分别比较。

--npz 接受 h1e、eri、nelec、ci_strs_a、ci_strs_b，norb 可由 h1e 推得；
--strings 可替换标签，--initial-state 提供按标签投影的显式初猜。
NPZ 默认不额外计算 CPU 最低根。--prepare-only 会执行输入准备和所选参考，
不建立 CUDA 上下文。--profile 只剖析 Torch Hc/S²/Davidson，跳过 RDM 与
CPU 验收，其墙钟不能作为正式性能数字。

JSON 记录配置、源码/输入哈希、软件版本、冷启动、稳态统计和资源采样。
正式汇总要求至少五次稳态且数值全部通过；setup 合并建表和传输，通信未单独计时。
PyTorch allocator 峰值不含 SBD C++ 分配，离散采样峰值是实际峰值的下界。
结果仅代表固定空间求解，不能直接外推至完整 SQD、PCM 或大体系科学结果。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time
import traceback


CASES = {
    "lih": ("Li 0 0 0; H 0 0 1.6", "sto-3g", 0, 0, None),
    "lih_plus": ("Li 0 0 0; H 0 0 1.6", "sto-3g", 1, 1, None),
    "n2": ("N 0 0 0; N 0 0 1.1", "sto-3g", 0, 0, None),
    "n2_plus": ("N 0 0 0; N 0 0 1.1", "sto-3g", 1, 1, (70, 61)),
    "water": ("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", "6-31g", 0, 0, (80, 71)),
}


class ResourceSampler:
    """采样本基准进程组的显存；离散采样只给出峰值下界，不能替代分配器统计。"""

    def __init__(self, devices, process_group, interval=0.1):
        self.devices, self.process_group, self.interval = devices, process_group, interval
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.samples = []
        self.errors = []
        self.backend = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _owned(self, pid):
        try:
            if os.getpgid(pid) == self.process_group:
                return True
        except (ProcessLookupError, PermissionError):
            return False
        # MPI rank 可能另建进程组，但仍是 launcher 的子孙；不能只按 PGID 判断。
        seen = set()
        current = pid
        while current > 1 and current not in seen:
            if current == self.process_group:
                return True
            seen.add(current)
            try:
                status = Path(f"/proc/{current}/status").read_text()
                current = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
            except (OSError, StopIteration, ValueError):
                return False
        return False

    def _run(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            handles = [(device, pynvml.nvmlDeviceGetHandleByIndex(int(device))) for device in self.devices]
            self.backend = "NVML"
        except Exception:
            pynvml, handles = None, None
            self.backend = "nvidia-smi"
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                records = []
                if pynvml:
                    for device, handle in handles:
                        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        memory = {str(p.pid): int(p.usedGpuMemory) for p in processes
                                  if self._owned(p.pid) and 0 <= p.usedGpuMemory < 2**63}
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        records.append({"device": device, "owned_process_memory_bytes": memory,
                                        "device_gpu_utilization_percent": utilization.gpu})
                else:
                    query = subprocess.run(
                        ["nvidia-smi", "-i", ",".join(self.devices),
                         "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=3, check=True,
                    )
                    by_device = {}
                    for line in query.stdout.splitlines():
                        uuid, pid, memory = (part.strip() for part in line.split(","))
                        if self._owned(int(pid)) and memory.isdigit():
                            by_device.setdefault(uuid, {})[pid] = int(memory) * 1024**2
                    utilization = subprocess.run(
                        ["nvidia-smi", "-i", ",".join(self.devices),
                         "--query-gpu=uuid,utilization.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=3, check=True,
                    )
                    for line in utilization.stdout.splitlines():
                        uuid, value = (part.strip() for part in line.split(","))
                        records.append({"device": uuid, "owned_process_memory_bytes": by_device.get(uuid, {}),
                                        "device_gpu_utilization_percent": float(value) if value.isdigit() else None})
                with self.lock:
                    self.samples.append({"time": started, "devices": records})
            except Exception as error:
                with self.lock:
                    if not self.errors:
                        self.errors.append(str(error))
                if isinstance(error, FileNotFoundError):
                    break
            self.stop.wait(max(0, self.interval - (time.monotonic() - started)))
        if pynvml:
            pynvml.nvmlShutdown()

    def snapshot(self, since):
        with self.lock:
            samples = [sample for sample in self.samples if sample["time"] >= since]
        peak_by_pid, utilization, peak_total = {}, [], 0
        for sample in samples:
            total = 0
            for device in sample["devices"]:
                for pid, memory in device["owned_process_memory_bytes"].items():
                    peak_by_pid[pid] = max(peak_by_pid.get(pid, 0), memory)
                    total += memory
                if device["device_gpu_utilization_percent"] is not None:
                    utilization.append(device["device_gpu_utilization_percent"])
            peak_total = max(peak_total, total)
        intervals = [b["time"] - a["time"] for a, b in zip(samples, samples[1:])]
        return {"backend": self.backend, "requested_interval_seconds": self.interval,
                "sample_count": len(samples), "actual_median_interval_seconds": statistics.median(intervals) if intervals else None,
                "process_peak_memory_bytes_lower_bound": peak_by_pid,
                "simultaneous_process_memory_bytes_lower_bound": peak_total if peak_by_pid else None,
                "owned_process_memory_observed": bool(peak_by_pid),
                "device_utilization_mean_percent": statistics.mean(utilization) if utilization else None,
                "device_utilization_max_percent": max(utilization) if utilization else None,
                "scope": "process-group or launcher-descendant memory; no attributed samples means unknown, not zero; device-wide utilization may include other processes; sampled peak is a lower bound",
                "errors": list(self.errors)}


def _json_default(value):
    """仅把数值数组转换成 JSON 值，未知对象仍明确报错。"""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _save(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _fingerprint(array):
    import numpy as np

    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _metadata():
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    versions = {}
    for name in ("numpy", "torch", "pyscf", "mpi4py"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = root / "src/tyxonq/applications/chem/algorithms/sqd/gpu"
    return {
        "python": sys.version, "packages": versions, "git_commit": commit,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources.rglob("*") if p.suffix in (".py", ".cpp", ".cuh", ".patch")},
    }


def prepare_problem(name, args):
    """在 CPU 上构造固定输入；截断案例保留 HF，并使用完全确定的随机子空间。"""
    import numpy as np
    from pyscf import ao2mo, fci, gto, lib, scf

    lib.num_threads(args.threads)
    started = time.perf_counter()
    if args.npz:
        with np.load(args.npz, allow_pickle=False) as data:
            h1e, eri = data["h1e"].copy(), data["eri"].copy()
            norb = int(data["norb"]) if "norb" in data else h1e.shape[0]
            if "nelec" in data:
                nelec = tuple(map(int, data["nelec"]))
            else:
                nelec = tuple(json.loads(str(data["metadata_json"]))["nelec"])
            labels = None if args.strings else (data["ci_strs_a"].copy(), data["ci_strs_b"].copy())
        if args.strings:
            with np.load(args.strings, allow_pickle=False) as data:
                labels = (data["ci_strs_a"].copy(), data["ci_strs_b"].copy())
        meta = {"source": "fixed_npz", "input_path": str(Path(args.npz).resolve()),
                "strings_path": args.strings}
    else:
        atom, basis, charge, spin, counts = CASES[name]
        mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
        mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run(conv_tol=1e-12)
        if not mf.converged:
            raise RuntimeError(f"SCF did not converge: {name}")
        norb, nelec = mf.mo_coeff.shape[1], mol.nelec
        h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
        labels = tuple(fci.cistring.make_strings(range(norb), n) for n in nelec)
        if counts:
            rng = np.random.default_rng(args.seed)
            labels = tuple(np.sort(np.r_[s[0], rng.choice(s[1:], size=n - 1, replace=False)])
                           for s, n in zip(labels, counts))
        meta = {"source": "molecule", "atom_angstrom": atom, "basis": basis,
                "charge": charge, "spin": spin, "scf_converged": True,
                "nuclear_repulsion_hartree_excluded": float(mol.energy_nuc())}
    from tyxonq.applications.chem.algorithms.sqd.gpu.validation import normalize_problem

    labels, h1e, eri, norb, nelec = normalize_problem(labels, h1e, eri, norb, nelec)
    problem = {"name": name, "h1e": h1e, "eri": eri, "strings": labels,
               "norb": norb, "nelec": nelec, "initial": None, "reference": None}
    if args.initial_state:
        from tyxonq.applications.chem.algorithms.sqd.fermion import SCIState

        problem["initial"] = SCIState.load(args.initial_state)
    elif args.initial == "first":
        from tyxonq.applications.chem.algorithms.sqd.fermion import SCIState

        coefficients = np.zeros(tuple(map(len, labels)))
        coefficients[0, 0] = 1
        problem["initial"] = SCIState(coefficients, *labels, norb, nelec)
    meta.update(norb=norb, nelec=nelec, shape=tuple(map(len, labels)),
                dimension=len(labels[0]) * len(labels[1]), seed=args.seed,
                fingerprint={k: _fingerprint(v) for k, v in
                             {"h1e": h1e, "eri": eri, "alpha": labels[0], "beta": labels[1]}.items()},
                prepare_seconds=time.perf_counter() - started)
    if problem["initial"] is not None:
        state = problem["initial"]
        meta["initial_state_fingerprint"] = {
            "amplitudes": _fingerprint(state.amplitudes), "alpha": _fingerprint(state.ci_strs_a),
            "beta": _fingerprint(state.ci_strs_b),
        }
    if args.reference == "pyscf" or (args.reference == "auto" and not args.npz and args.spin_sq is None):
        if args.spin_sq is not None:
            raise ValueError("Independent lowest-root reference currently requires spin_sq=None; use --reference none.")
        started = time.perf_counter()
        solver = fci.selected_ci.SelectedCI()
        solver.verbose = 0
        energy, state = fci.selected_ci.kernel_fixed_space(
            solver, h1e, eri, norb, nelec, labels, tol=1e-12,
            max_cycle=200, max_space=30, tol_residual=1e-10,
        )
        absorbed = fci.direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5)
        hc = fci.selected_ci.contract_2e(absorbed, state, norb, nelec)
        residual = float(np.linalg.norm(hc - energy * state))
        if residual > 1e-7:
            raise RuntimeError(f"PySCF reference is not converged: residual={residual}")
        problem["reference"] = {
            "energy": float(energy), "rdm1": fci.selected_ci.make_rdm1(state, norb, nelec),
            "rdm2": fci.selected_ci.make_rdm2(state, norb, nelec),
        }
        meta["reference"] = {"electronic_energy": float(energy), "residual": residual,
                             "seconds": time.perf_counter() - started}
    problem["metadata"] = meta
    return problem


def _worker(problem, options, backend, messages):
    """单 GPU 或独立批次工作进程；只有本进程及其子进程会建立 CUDA 上下文。"""
    os.setsid()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(options["devices"])
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(options["threads"])
    solver = None
    try:
        import numpy as np
        from tyxonq.applications.chem.algorithms.sqd.gpu import make_cuda_sci_solver, make_sbd_sci_solver

        if options["profile"]:
            messages.put(("started", 0))
            messages.put(("profile", _profile_kernels(problem, options)))
            messages.put(("done", None))
            return
        reports = []
        config = {key: options[key] for key in
                  ("threads", "chunk_size", "tolerance", "max_cycles", "max_space",
                   "operator_residual_tolerance", "rdm_energy_tolerance", "spin_sq", "spin_shift", "validation")}
        config.update(initial_state=problem["initial"], device_text="cuda:0")
        if backend != "native":
            config["spin_backend"] = options["spin_backend"]
        factory = make_cuda_sci_solver if backend == "torch" else make_sbd_sci_solver
        if backend != "torch":
            config["mode"] = backend
        elif options["rdm_backend"]:
            config["rdm_backend"] = options["rdm_backend"]
        solver = factory(report_callback=reports.append,
                         device_ids=options["devices"] if len(options["devices"]) > 1 else None,
                         worker_timeout=options["timeout"], **config)
        batches = [problem["strings"] for _ in range(options["batches"])]
        for repeat in range(options["repeats"] + 1):
            reports.clear()
            messages.put(("started", repeat))
            started = time.perf_counter()
            results = solver(batches, problem["h1e"], problem["eri"], problem["norb"], problem["nelec"])
            elapsed = time.perf_counter() - started
            checks = []
            reference = problem["reference"]
            if reference:
                for result in results:
                    check = {"energy_abs_error": abs(result.energy - reference["energy"]),
                             "rdm1_max_abs_error": float(np.max(np.abs(result.rdm1 - reference["rdm1"]))),
                             "rdm2_max_abs_error": float(np.max(np.abs(result.rdm2 - reference["rdm2"])))}
                    check["pass"] = (check["energy_abs_error"] <= 1e-8 and
                                     max(check["rdm1_max_abs_error"], check["rdm2_max_abs_error"]) <= 1e-6)
                    checks.append(check)
            record = {"repeat": repeat, "cold": repeat == 0, "wall_seconds": elapsed,
                      "batches_per_second": len(results) / elapsed,
                      "electronic_energies": [float(result.energy) for result in results],
                      "converged": [bool(result.converged) for result in results],
                      "reports": list(reports), "reference_checks": checks}
            record["pass"] = all(record["converged"]) and all(check["pass"] for check in checks)
            messages.put(("result", record))
        messages.put(("done", None))
    except BaseException:
        messages.put(("error", traceback.format_exc()))
    finally:
        if solver is not None:
            solver.close()


def _wait_for_idle_devices(devices, timeout):
    """启动本组前只读检查目标 GPU；有任何计算进程时等待，不抢占或终止它。"""
    def query(fields, selector=None):
        command = ["nvidia-smi", f"--query-{fields}", "--format=csv,noheader,nounits"]
        if selector is not None:
            command.extend(["-i", ",".join(selector)])
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if completed.returncode:
            raise RuntimeError(f"Cannot verify idle GPUs: {completed.stderr.strip()}")
        return [[part.strip() for part in line.split(",")] for line in completed.stdout.splitlines() if line.strip()]

    selected = query("gpu=index,uuid", devices)
    if len(selected) != len(devices):
        raise RuntimeError("GPU selection did not resolve to one physical device per requested ID")
    names = {uuid: index for index, uuid in selected}
    deadline = time.monotonic() + timeout
    notified = False
    while True:
        processes = [{"device": names[uuid], "gpu_uuid": uuid, "pid": int(pid)}
                     for uuid, pid in query("compute-apps=gpu_uuid,pid") if uuid in names]
        if not processes:
            return
        if not notified:
            print(json.dumps({"status": "waiting_for_idle_gpu", "processes": processes,
                              "timeout_seconds": timeout}), flush=True)
            notified = True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Requested GPUs remain occupied; refusing to start: {processes}")
        time.sleep(min(5.0, remaining))


def _run_isolated(problem, args, backend):
    """外层看门狗终止本基准创建的进程组，避免 C++ 卡住后遗留 GPU 作业。"""
    import signal

    context = mp.get_context("spawn")
    messages = context.Queue()
    options = vars(args).copy()
    worker = context.Process(target=_worker, args=(problem, options, backend, messages))
    worker.start()
    sampler = ResourceSampler(args.devices, worker.pid)
    sampler.thread.start()
    output = {"backend": backend, "records": [], "status": "running"}
    deadline = time.monotonic() + args.timeout
    sample_started = time.monotonic()
    try:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Benchmark call exceeded {args.timeout} seconds")
            try:
                event, payload = messages.get(timeout=0.2)
            except queue.Empty:
                if worker.exitcode is not None:
                    raise RuntimeError(f"Worker exited without a result: exitcode={worker.exitcode}")
                continue
            if event == "started":
                deadline = time.monotonic() + args.timeout
                sample_started = time.monotonic()
            elif event == "result":
                payload["resource_sampling"] = sampler.snapshot(sample_started)
                output["records"].append(payload)
                print(json.dumps({"case": problem["name"], "backend": backend,
                                  "repeat": payload["repeat"], "wall_seconds": payload["wall_seconds"],
                                  "pass": payload["pass"]}), flush=True)
                deadline = time.monotonic() + args.timeout
            elif event == "done":
                output["status"] = "complete"
                break
            elif event == "profile":
                output["profile"] = payload
                output["resource_sampling"] = sampler.snapshot(sample_started)
            elif event == "error":
                raise RuntimeError(payload)
    except BaseException as error:
        output.update(status="failed", error=str(error))
    finally:
        sampler.stop.set()
        sampler.thread.join(timeout=7)
        if output["status"] != "complete" and worker.is_alive():
            try:
                os.killpg(worker.pid, signal.SIGTERM)
            except ProcessLookupError:
                worker.terminate()
        worker.join(timeout=5)
        if worker.is_alive():
            os.killpg(worker.pid, signal.SIGKILL)
            worker.join(timeout=5)
        messages.cancel_join_thread()
        messages.close()
    warm = [record["wall_seconds"] for record in output["records"] if not record["cold"]]
    output["pass"] = (output["status"] == "complete" and all(record["pass"] for record in output["records"]))
    if warm:
        output["steady_wall_seconds"] = {"median": statistics.median(warm), "min": min(warm), "max": max(warm)}
    return output


def _profile_kernels(problem, options):
    """仅剖析 GPU 数值阶段，不将 CPU RDM/验收时间混入算符热点占比。"""
    import numpy as np
    import torch
    from pyscf.fci import selected_ci
    from tyxonq.applications.chem.algorithms.sqd.gpu.torch_davidson import davidson_lowest
    from tyxonq.applications.chem.algorithms.sqd.gpu.torch_selected_ci import (
        TorchSelectedCIContractor, absorb_h1e_torch, make_hdiag_torch,
    )
    from tyxonq.applications.chem.algorithms.sqd.gpu.torch_spin_square import (
        TorchSelectedS2Contractor, make_spin_square_diagonal_torch,
    )
    from tyxonq.applications.chem.algorithms.sqd.gpu.validation import project_initial_state

    device = torch.device("cuda:0")
    torch.set_num_threads(options["threads"])
    h1 = torch.as_tensor(problem["h1e"], dtype=torch.float64, device=device)
    eri = torch.as_tensor(problem["eri"], dtype=torch.float64, device=device)
    norb, nelec, labels = problem["norb"], problem["nelec"], problem["strings"]
    links = selected_ci._all_linkstr_index(labels, norb, nelec)
    hamiltonian = TorchSelectedCIContractor(absorb_h1e_torch(h1, eri, nelec), links, nelec,
                                          chunk_size=options["chunk_size"])
    spin = TorchSelectedS2Contractor(labels, norb, nelec, device=device, dtype=torch.float64,
                                   backend=options["spin_backend"])
    shape = tuple(map(len, labels))
    random = np.random.default_rng(options["seed"]).normal(size=shape)
    random /= np.linalg.norm(random)
    vector = torch.as_tensor(random, dtype=torch.float64, device=device)
    diagonal = make_hdiag_torch(h1, eri, labels)
    spin_sq, shift = options["spin_sq"], options["spin_shift"]
    minimum_spin = 0.5 * abs(nelec[0] - nelec[1])
    linear_penalty = spin_sq is not None and spin_sq < minimum_spin * (minimum_spin + 1) + 0.1
    if linear_penalty:
        diagonal = diagonal + shift * (make_spin_square_diagonal_torch(
            labels, norb, nelec, device=device, dtype=torch.float64) - spin_sq)

    def operator(coefficients):
        matrix = coefficients.reshape(shape)
        with torch.profiler.record_function("Hc"):
            image = hamiltonian.contract(matrix)
        if spin_sq is not None:
            with torch.profiler.record_function("spin_penalty"):
                residual = spin.contract(matrix) - spin_sq * matrix
                image = image + shift * (residual if linear_penalty else spin.contract(residual) - spin_sq * residual)
        return image.reshape(-1)

    initial = project_initial_state(problem["initial"], labels, norb, nelec)
    initial = None if initial is None else torch.as_tensor(initial, dtype=torch.float64, device=device).reshape(-1)
    results = {}
    for name, operation in (("Hc", lambda: hamiltonian.contract(vector)),
                            ("S2c", lambda: spin.contract(vector)),
                            ("Davidson", lambda: davidson_lowest(
                                operator, diagonal.reshape(-1), initial_vector=initial,
                                tolerance=options["tolerance"], max_cycles=options["max_cycles"],
                                max_space=options["max_space"]))):
        # 先预热，剖析窗口之外的 setup、CPU 校验和初次 CUDA 初始化不计入热点。
        operation()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False, profile_memory=False) as profiler:
            with torch.profiler.record_function(name):
                for _ in range(options["profile_repeats"]):
                    value = operation()
            torch.cuda.synchronize(device)
        wall = time.perf_counter() - started
        events = []
        for event in profiler.key_averages():
            own = float(getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0)))
            cumulative = float(getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0)))
            if event.key.startswith("aten::"):
                events.append({"operator": event.key, "calls": event.count,
                               "self_cuda_microseconds": own, "cuda_microseconds": cumulative,
                               "self_cpu_microseconds": float(event.self_cpu_time_total)})
        total = sum(event["self_cuda_microseconds"] for event in events)
        gather = sum(event["self_cuda_microseconds"] for event in events
                     if any(word in event["operator"] for word in ("index", "gather", "scatter")))
        entry = {"profiled_wall_seconds": wall, "calls": options["profile_repeats"],
                 "aten_self_cuda_microseconds": total, "gather_scatter_self_cuda_microseconds": gather,
                 "gather_scatter_gpu_fraction": gather / total if total else None,
                 "operators": sorted(events, key=lambda event: event["self_cuda_microseconds"], reverse=True)[:25]}
        if name == "Davidson":
            fresh_residual = float(torch.linalg.vector_norm(operator(value.coefficients) - value.energy * value.coefficients))
            entry.update(converged=value.converged, iterations=value.iterations, h_matvecs=value.h_matvecs,
                         residual=value.residual_norm, fresh_residual=fresh_residual,
                         record_function_cuda_microseconds={event.key: float(getattr(event, "device_time_total", 0))
                                                            for event in profiler.key_averages()
                                                            if event.key in ("Hc", "spin_penalty", "Davidson")})
            if not value.converged or fresh_residual > options["operator_residual_tolerance"]:
                raise RuntimeError(f"Profiled Davidson failed convergence: {entry}")
        results[name] = entry
    return {"scope": "GPU Hc/S2c/Davidson only; excludes setup/RDM/CPU validation; profiler perturbs timing",
            "spin_backend": options["spin_backend"],
            "seed": options["seed"], "dtype": "float64", "stages": results}


def _mpi_worker(payload_path):
    """仅由本入口生成的临时 payload 调用；不接受外部 pickle 数据。"""
    import pickle
    import numpy as np
    from mpi4py import MPI
    import torch
    from tyxonq.applications.chem.algorithms.sqd.gpu.sbd_solver import (
        _load_extension, convert_raw_rdms, interleaved_phase,
    )
    from tyxonq.applications.chem.algorithms.sqd.gpu.validation import (
        assemble_result, compare_cpu, independent_cpu_validation, project_initial_state, validate_rdms,
    )

    with open(payload_path, "rb") as handle:
        problem, config, result_path = pickle.load(handle)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() != config["mpi_grid"][0] * config["mpi_grid"][1]:
        raise RuntimeError("MPI communicator size disagrees with grid")
    torch.cuda.set_device(rank)
    torch.set_num_threads(config["threads"])
    phase = interleaved_phase(problem["strings"], problem["norb"], device=f"cuda:{rank}").cpu().numpy()
    initial = project_initial_state(problem["initial"], problem["strings"], problem["norb"], problem["nelec"])
    raw_initial = None if initial is None else initial * phase
    report = {"backend": "native_mpi", "status": "running", "records": [],
              "grid": config["mpi_grid"], "mpi_library": MPI.Get_library_version(),
              "communication": "SBD_NON_CUDA_AWARE_MPI=1; CPU staging; solve time includes communication"}
    try:
        for repeat in range(config["repeats"] + 1):
            if rank == 0:
                report["active_repeat"] = repeat
                report["active_started"] = time.time()
                _save(Path(result_path), report)
            comm.Barrier()
            started = time.perf_counter()
            context = _load_extension().TPBContext(
                problem["h1e"], problem["eri"], *problem["strings"], problem["norb"], problem["nelec"],
                device_id=rank, comm=comm, grid=tuple(config["mpi_grid"]),
            )
            setup_seconds = comm.allreduce(time.perf_counter() - started, op=MPI.MAX)
            comm.Barrier()
            solve_started = time.perf_counter()
            raw = context.solve(tolerance=config["tolerance"], max_cycles=config["max_cycles"],
                                max_space=config["max_space"], initial_coefficients=raw_initial)
            torch.cuda.synchronize(rank)
            solve_seconds = comm.allreduce(time.perf_counter() - solve_started, op=MPI.MAX)
            stage_seconds = {key: comm.allreduce(float(value), op=MPI.MAX)
                             for key, value in raw.get("timings", {}).items()}
            diagnostics = comm.gather({"rank": rank, "residual": float(raw["residual"]),
                                       "iterations": int(raw.get("iterations", -1)),
                                       "h_matvecs": int(raw.get("h_matvecs", -1)),
                                       "converged": bool(raw["converged"])}, root=0)
            if rank == 0:
                if not all(item["converged"] and item["residual"] <= config["operator_residual_tolerance"]
                           for item in diagnostics):
                    raise RuntimeError(f"MPI native solve did not converge: {diagnostics}")
                coefficients = np.asarray(raw["coefficients"]).reshape(phase.shape) * phase
                rdms = convert_raw_rdms(raw, problem["norb"])
                physical = assemble_result(coefficients, problem["strings"], problem["h1e"], problem["eri"],
                                           problem["norb"], problem["nelec"], *rdms)
                checks = validate_rdms(physical, float(raw["energy"]), energy_tolerance=config["rdm_energy_tolerance"])
                checks.update(rank_diagnostics=diagnostics, gpu_setup_seconds_max_rank=setup_seconds,
                              gpu_solver_seconds_max_rank=solve_seconds, mpi_communication_seconds=None,
                              stages_seconds_max_rank=stage_seconds,
                              rdm_included_in_native_solve=True, independent_cpu_validation=False)
                if config["validation"] == "full_cpu":
                    cpu = independent_cpu_validation(
                        coefficients, problem["strings"], problem["h1e"], problem["eri"], problem["norb"],
                        problem["nelec"], spin_sq=None, spin_shift=config["spin_shift"], threads=config["threads"],
                    )
                    image = np.asarray(raw["hamiltonian_image"]).reshape(phase.shape) * phase
                    checks.update(compare_cpu(physical, cpu, image,
                                              residual_tolerance=config["operator_residual_tolerance"],
                                              energy_tolerance=config["rdm_energy_tolerance"]))
                    checks.update(independent_cpu_validation=True, cpu_validation_seconds=cpu["seconds"],
                                  elementwise_gpu_cpu_hc_compared=True)
                reference_checks = []
                if problem["reference"]:
                    reference = problem["reference"]
                    error = {"energy_abs_error": abs(physical.energy - reference["energy"]),
                             "rdm1_max_abs_error": float(np.max(np.abs(physical.rdm1 - reference["rdm1"]))),
                             "rdm2_max_abs_error": float(np.max(np.abs(physical.rdm2 - reference["rdm2"])))}
                    error["pass"] = error["energy_abs_error"] <= 1e-8 and max(error["rdm1_max_abs_error"], error["rdm2_max_abs_error"]) <= 1e-6
                    reference_checks.append(error)
            comm.Barrier()
            wall_seconds = comm.allreduce(time.perf_counter() - started, op=MPI.MAX)
            if rank == 0:
                report["records"].append({"repeat": repeat, "cold": repeat == 0, "wall_seconds": wall_seconds,
                                          "wall_timing": "MPI.MAX across ranks including rank-0 CPU validation",
                                          "electronic_energies": [physical.energy], "converged": [True],
                                          "reports": [{"batch": 0, "report": checks}], "reference_checks": reference_checks,
                                          "pass": all(item["pass"] for item in reference_checks)})
                _save(Path(result_path), report)
            del context
        if rank == 0:
            report["status"] = "complete"
            report["pass"] = all(item["pass"] for item in report["records"])
            _save(Path(result_path), report)
    except BaseException:
        if rank == 0:
            report.update(status="failed", error=traceback.format_exc(), **{"pass": False})
            _save(Path(result_path), report)
        comm.Abort(1)


def _run_mpi(problem, args):
    """由单个父进程启动 MPI，并对每次求解设置独立的外层墙钟上限。"""
    import pickle
    import signal
    import tempfile

    with tempfile.TemporaryDirectory(prefix="tyxonq_sbd_mpi_") as directory:
        folder = Path(directory)
        payload, result_path = folder / "payload.pkl", folder / "result.json"
        with payload.open("wb") as handle:
            pickle.dump((problem, vars(args), str(result_path)), handle)
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(args.devices)}
        command = [args.mpiexec, "--bind-to", "none", "-n", str(len(args.devices)), sys.executable,
                   str(Path(__file__).resolve()), "_mpi_worker", str(payload)]
        with (folder / "mpi.log").open("w") as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            sampler = ResourceSampler(args.devices, process.pid)
            sampler.thread.start()
            deadline, current_repeat = time.monotonic() + args.timeout, None
            report = {"backend": "native_mpi", "status": "running", "records": []}
            sample_starts, sampling = {}, {}
            try:
                while process.poll() is None:
                    if result_path.exists():
                        report = json.loads(result_path.read_text())
                        if report.get("active_repeat") != current_repeat:
                            current_repeat = report.get("active_repeat")
                            deadline = time.monotonic() + args.timeout
                            sample_starts[current_repeat] = time.monotonic()
                        for record in report["records"]:
                            number = record["repeat"]
                            if number not in sampling:
                                sampling[number] = sampler.snapshot(sample_starts.get(number, 0))
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"MPI benchmark call exceeded {args.timeout} seconds")
                    time.sleep(0.1)
                if result_path.exists():
                    report = json.loads(result_path.read_text())
                if process.returncode or report["status"] != "complete":
                    raise RuntimeError(f"MPI failed: exitcode={process.returncode}; {report.get('error', '')}")
            except BaseException as error:
                report.update(status="failed", error=str(error), **{"pass": False})
            finally:
                sampler.stop.set()
                sampler.thread.join(timeout=7)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
            for record in report["records"]:
                record["resource_sampling"] = sampling.get(record["repeat"], sampler.snapshot(sample_starts.get(record["repeat"], 0)))
        report["launcher_command"] = command[:6]
        report["launcher_binding"] = "OpenMPI --bind-to none; per-rank CPU thread limits remain explicit"
        if report["status"] != "complete":
            report["launcher_log_tail"] = (folder / "mpi.log").read_text()[-16000:]
        warm = [record["wall_seconds"] for record in report["records"] if not record["cold"]]
        if warm:
            report["steady_wall_seconds"] = {"median": statistics.median(warm), "min": min(warm), "max": max(warm)}
        return report


def summarize_results(paths):
    """只汇总现有 JSON；输入、数值设置和工作量不一致时不生成加速比。"""
    def statistics_of(values):
        return None if not values else {"median": statistics.median(values), "min": min(values), "max": max(values)}

    numeric_settings = ("threads", "chunk_size", "tolerance", "max_cycles", "max_space",
                        "operator_residual_tolerance", "rdm_energy_tolerance", "spin_sq", "spin_shift")
    rows, skipped = [], []
    for path in paths:
        source = json.loads(Path(path).read_text())
        config = source.get("configuration", {})
        for case in source.get("cases", []):
            if case.get("status") == "failed":
                skipped.append({"source": str(path), "case": case["name"], "reason": case.get("error", "input failed")})
            for run in case.get("runs", []):
                if not run.get("records"):
                    skipped.append({"source": str(path), "case": case["name"], "backend": run["backend"],
                                    "reason": run.get("error", "no normal timing records; profile/prepare results are excluded")})
                    continue
                records = run["records"]
                cold = [record["wall_seconds"] for record in records if record["cold"]]
                warm = [record for record in records if not record["cold"]]
                warm_times = [record["wall_seconds"] for record in warm]
                mpi = run["backend"] == "native_mpi"
                reports = [item["report"] for record in warm for item in record.get("reports", [])]
                stages = {}
                for report in reports:
                    for key, value in report.items():
                        if key.endswith("_seconds") and isinstance(value, (int, float)):
                            stages.setdefault(key, []).append(value)
                initial = config.get("initial", "first")
                if config.get("initial_state"):
                    initial = case.get("input", {}).get("initial_state_fingerprint")
                settings = {key: config.get(key) for key in numeric_settings}
                settings.update(validation=config.get("validation"), initial=initial,
                                batches=config.get("batches", 1), fingerprint=case.get("input", {}).get("fingerprint"),
                                package_versions=source.get("metadata", {}).get("packages"),
                                spin_backend=config.get("spin_backend", "torch"))
                reasons = []
                if not run.get("pass", False) or run.get("status") != "complete":
                    reasons.append("run did not pass verification")
                if len(warm) < 5:
                    reasons.append(f"only {len(warm)} steady repetitions; at least five are required")
                if not settings["fingerprint"] or initial is None:
                    reasons.append("input/initial-state fingerprints are unavailable")
                if any(value <= 0 for value in warm_times):
                    reasons.append("invalid nonpositive elapsed time")
                batches = int(config.get("batches", 1))
                if any(len(record.get("electronic_energies", [])) != batches for record in records):
                    reasons.append("reported result count disagrees with configured batch count")
                stage_report = reports[0] if reports else {}
                row = {"id": len(rows), "source": str(path), "case": case["name"], "backend": run["backend"],
                       "validation": config.get("validation"), "parallelism": "single_task_MPI" if mpi else "independent_batches",
                       "gpu_count": len(config.get("devices", ["0"])), "devices": config.get("devices"),
                       "batch_count": batches, "dimension": case.get("input", {}).get("dimension"),
                       "rdm_backend": stage_report.get("rdm_backend", "sbd" if mpi else config.get("rdm_backend")),
                       "spin_backend": config.get("spin_backend", "torch"),
                       "cold_wall_seconds": cold, "steady_repetitions": len(warm),
                       "steady_wall_seconds": statistics_of(warm_times),
                       "steady_throughput_batches_per_second": statistics_of([batches / value for value in warm_times if value > 0]),
                       "stage_seconds": {key: statistics_of(value) for key, value in stages.items()},
                       "eligible_for_speedup": not reasons, "exclusions": reasons,
                       "comparison_settings": settings, "implementation": source.get("metadata", {})}
                rows.append(row)
    comparisons = []
    for target in rows:
        if not target["eligible_for_speedup"]:
            continue
        for baseline in rows:
            if baseline["id"] == target["id"] or not baseline["eligible_for_speedup"]:
                continue
            same_settings = baseline["comparison_settings"] == target["comparison_settings"]
            spin_only = (baseline["spin_backend"] == "torch" and target["spin_backend"] == "cuda_fused" and
                         baseline["backend"] == target["backend"] and
                         baseline["rdm_backend"] == target["rdm_backend"] and
                         baseline["implementation"].get("source_sha256") == target["implementation"].get("source_sha256") and
                         baseline["gpu_count"] == target["gpu_count"] and
                         baseline["parallelism"] == target["parallelism"] and
                         {k: v for k, v in baseline["comparison_settings"].items() if k != "spin_backend"} ==
                         {k: v for k, v in target["comparison_settings"].items() if k != "spin_backend"})
            if not same_settings and not spin_only:
                continue
            baseline_time = baseline["steady_wall_seconds"]["median"]
            target_time = target["steady_wall_seconds"]["median"]
            same_implementation = (baseline["implementation"].get("source_sha256") ==
                                   target["implementation"].get("source_sha256"))
            kind = "spin_backend_end_to_end" if spin_only else None
            if (same_settings and baseline["backend"] == "torch" and target["parallelism"] == "independent_batches" and
                    baseline["gpu_count"] == target["gpu_count"]):
                if target["backend"] != "torch" and same_implementation:
                    kind = "backend_end_to_end"
                elif target["backend"] == "torch" and baseline["id"] < target["id"] and not same_implementation:
                    kind = "implementation_end_to_end"
                elif (target["backend"] == "torch" and same_implementation and
                      baseline["rdm_backend"] == "cpu" and target["rdm_backend"] == "sbd"):
                    kind = "gpu_rdm_end_to_end"
            if (same_settings and baseline["gpu_count"] == 1 and target["gpu_count"] > 1 and same_implementation and
                    baseline["backend"] == target["backend"].removesuffix("_mpi") and
                    baseline["rdm_backend"] == target["rdm_backend"]):
                kind = "single_task_MPI_scaling" if target["parallelism"] == "single_task_MPI" else "independent_batch_scaling"
            if kind is not None:
                speedup = baseline_time / target_time
                comparison = {"kind": kind, "baseline_row": baseline["id"], "target_row": target["id"],
                              "case": target["case"], "validation": target["validation"],
                              "batch_count": target["batch_count"], "end_to_end_speedup": speedup,
                              "baseline_spin_backend": baseline["spin_backend"], "target_spin_backend": target["spin_backend"],
                              "time_reduction_fraction": 1 - target_time / baseline_time,
                              "baseline_throughput_batches_per_second": baseline["batch_count"] / baseline_time,
                              "target_throughput_batches_per_second": target["batch_count"] / target_time}
                if kind.endswith("scaling"):
                    comparison["parallel_efficiency"] = speedup / target["gpu_count"]
                comparisons.append(comparison)
    return {"status": "complete", "sources": [str(path) for path in paths], "rows": rows,
            "comparisons": comparisons, "skipped": skipped,
            "rules": {"minimum_steady_repetitions": 5, "cold_excluded_from_speedup": True,
                      "matching": "identical input/initial-state fingerprints, numerical settings, validation and batch count",
                      "MPI": "native one-batch baseline versus one-batch MPI; all-rank maximum wall time including validation",
                      "implementation_comparison": "for same torch backend, earlier input JSON is the baseline",
                      "parallel_scaling": "requires identical implementation source hashes and the same RDM backend",
                      "timing_limit": "a small sample range is descriptive, not a confidence interval"}}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs="+", choices=[*CASES, "all"], default=["lih"])
    parser.add_argument("--backend", nargs="+", choices=["torch", "native", "hybrid"], default=["torch", "native", "hybrid"])
    parser.add_argument("--devices", nargs="+", default=["0"], help="物理 GPU 编号；每次运行前确认空闲")
    parser.add_argument("--mpi-grid", nargs=2, type=int, metavar=("ALPHA", "BETA"), help="单任务原生 SBD MPI 网格；由本入口启动 mpiexec")
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--validation", choices=["full_cpu", "gpu"], default="full_cpu")
    parser.add_argument("--rdm-backend", choices=["cpu", "sbd"], help="仅覆盖 torch 路径的 RDM 后端")
    parser.add_argument("--npz", help="固定积分和组态输入，不执行 SCF")
    parser.add_argument("--strings", help="从另一 SCIState/组态 npz 读取 ci_strs_a/b")
    parser.add_argument("--initial-state", help="显式 SCIState npz；按标签投影")
    parser.add_argument("--initial", choices=["first", "diagonal"], default="first")
    parser.add_argument("--reference", choices=["auto", "pyscf", "none"], default="auto")
    parser.add_argument("--repeats", type=int, default=5, help="冷启动以外的稳态重复数")
    parser.add_argument("--batches", type=int, default=1, help="每次测量的独立相同批次数")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--max-space", type=int, default=20)
    parser.add_argument("--operator-residual-tolerance", type=float, default=1e-7)
    parser.add_argument("--rdm-energy-tolerance", type=float, default=1e-8)
    parser.add_argument("--spin-sq", type=float)
    parser.add_argument("--spin-shift", type=float, default=0.1)
    parser.add_argument("--spin-backend", choices=["torch", "cuda_fused"], default="torch")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="仅检查输入和可选 PySCF 参考，不建立 CUDA 上下文")
    parser.add_argument("--profile", action="store_true", help="仅剖析 Torch GPU Hc/S²/Davidson，不执行 RDM 或 CPU 验收")
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--summarize", nargs="+", type=Path, metavar="JSON", help="只汇总现有基准；旧实现 JSON 放在前面")
    args = parser.parse_args()
    if args.summarize:
        if args.output.resolve() in [path.resolve() for path in args.summarize]:
            parser.error("Summary output must not overwrite an input benchmark JSON")
        return args
    if min(args.batches, args.threads, args.max_cycles) < 1 or args.repeats < 0 or args.timeout <= 0:
        parser.error("batches/threads/max_cycles/timeout must be positive; repeats must be nonnegative")
    if args.spin_sq is not None and "native" in args.backend:
        parser.error("native SBD does not support spin_sq; select torch/hybrid")
    if args.spin_backend != "torch" and "native" in args.backend:
        parser.error("native SBD does not support spin_backend; select torch/hybrid")
    if args.strings and not args.npz:
        parser.error("--strings requires --npz")
    if len(set(args.devices)) != len(args.devices):
        parser.error("--devices must be unique")
    if args.mpi_grid and (args.backend != ["native"] or args.batches != 1 or
                          args.mpi_grid not in ([2, 1], [2, 2]) or
                          len(args.devices) != args.mpi_grid[0] * args.mpi_grid[1]):
        parser.error("MPI requires --backend native --batches 1, grid 2 1 or 2 2, and exactly one GPU per rank")
    if args.profile and (args.backend != ["torch"] or args.batches != 1 or len(args.devices) != 1 or args.profile_repeats < 1):
        parser.error("Profiling requires --backend torch, one device, one batch and positive --profile-repeats")
    return args


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        report = summarize_results(args.summarize)
        _save(args.output, report)
        print(json.dumps({"rows": len(report["rows"]), "comparisons": len(report["comparisons"]),
                          "output": str(args.output)}))
        return 0
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    report = {"status": "running", "configuration": {**vars(args), "output": str(args.output)},
              "metadata": _metadata(), "energy_convention": "physical_electronic_energy_no_external_constant",
              "scope": "fixed_determinant_spaces_not_SQD_sampling_or_PCM", "cases": []}
    _save(args.output, report)
    names = [Path(args.npz).stem] if args.npz else (list(CASES) if "all" in args.case else args.case)
    for name in names:
        try:
            problem = prepare_problem(name, args)
            entry = {"name": name, "input": problem["metadata"], "runs": []}
            report["cases"].append(entry)
            _save(args.output, report)
            if not args.prepare_only:
                for backend in args.backend:
                    try:
                        _wait_for_idle_devices(args.devices, args.timeout)
                        result = _run_mpi(problem, args) if args.mpi_grid else _run_isolated(problem, args, backend)
                    except (OSError, subprocess.SubprocessError, RuntimeError, TimeoutError) as error:
                        result = {"backend": backend, "status": "failed", "pass": False,
                                  "records": [], "error": str(error), "stage": "GPU idle check / launch"}
                    entry["runs"].append(result)
                    _save(args.output, report)
        except Exception:
            report["cases"].append({"name": name, "status": "failed", "error": traceback.format_exc()})
            _save(args.output, report)
    report["status"] = "prepared" if args.prepare_only else "complete"
    report["all_pass"] = all(case.get("status") != "failed" and all(run["pass"] for run in case["runs"])
                             for case in report["cases"])
    _save(args.output, report)
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "_mpi_worker":
        _mpi_worker(sys.argv[2])
    else:
        raise SystemExit(main())
