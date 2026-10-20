"""SQD 五参数批接口；多卡时每个独立工作进程只拥有一张 GPU。"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import queue
import time
import traceback


def _solve_one(kind, strings, h1e, eri, norb, nelec, options):
    if kind == "native":
        from .sbd_solver import solve_sci_sbd

        return solve_sci_sbd(strings, h1e, eri, norb, nelec, **options)
    from .torch_sci_solver import solve_sci_cuda

    return solve_sci_cuda(strings, h1e, eri, norb, nelec, **options)


def _device_worker(device_id, tasks, messages, kind, threads):
    """必须先设置可见设备，再导入会初始化 CUDA/MPI 的求解模块。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(threads)
    while True:
        task = tasks.get()
        if task is None:
            return
        index, strings, h1e, eri, norb, nelec, options = task
        messages.put(("started", index, device_id, None))
        try:
            options = dict(options, device_text="cuda:0")
            options["progress_callback"] = lambda record: messages.put(
                ("progress", index, device_id, record)
            )
            result, report = _solve_one(kind, strings, h1e, eri, norb, nelec, options)
            messages.put(("done", index, device_id, (result, report)))
        except Exception:
            messages.put(("error", index, device_id, traceback.format_exc()))


class _DeviceBatchPool:
    """按需启动并复用进程；错误/超时时只终止本池创建的进程。"""

    def __init__(self, device_ids, kind, threads, timeout):
        self.device_ids = tuple(str(device) for device in device_ids)
        if not self.device_ids or len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("device_ids must be non-empty and unique")
        if threads < 1 or timeout <= 0:
            raise ValueError("threads and worker_timeout must be positive")
        self.kind, self.threads, self.timeout = kind, threads, timeout
        self.processes = []
        self.running = False
        atexit.register(self.close)

    def close(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        self.processes.clear()
        for name in ("tasks", "messages"):
            channel = getattr(self, name, None)
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()
                delattr(self, name)

    def _start(self):
        if self.processes:
            return
        context = mp.get_context("spawn")
        self.tasks, self.messages = context.Queue(), context.Queue()
        for device_id in self.device_ids:
            process = context.Process(
                target=_device_worker,
                args=(device_id, self.tasks, self.messages, self.kind, self.threads),
            )
            process.start()
            self.processes.append(process)

    def solve(self, batches, h1e, eri, norb, nelec, options, report_callback):
        if self.running:
            raise RuntimeError("A batch solver cannot run concurrent calls")
        if not batches:
            return []
        self.running = True
        progress_callback = options.get("progress_callback")
        worker_options = {key: value for key, value in options.items() if key != "progress_callback"}
        try:
            self._start()
            for index, strings in enumerate(batches):
                self.tasks.put((index, strings, h1e, eri, norb, nelec, worker_options))
            results = [None] * len(batches)
            reports = [None] * len(batches)
            active, completed = {}, 0
            last_message = time.monotonic()
            while completed < len(batches):
                now = time.monotonic()
                if any(process.exitcode is not None for process in self.processes):
                    raise RuntimeError("CUDA worker exited before returning all batches")
                if any(now - started > self.timeout for started in active.values()):
                    raise TimeoutError("CUDA batch exceeded worker_timeout")
                if not active and now - last_message > self.timeout:
                    raise TimeoutError("CUDA worker failed to start within worker_timeout")
                try:
                    event, index, device, payload = self.messages.get(timeout=0.2)
                except queue.Empty:
                    continue
                last_message = time.monotonic()
                if event == "started":
                    active[index] = last_message
                elif event == "progress" and progress_callback is not None:
                    progress_callback({"batch": index, "host_gpu_id": device, **payload})
                elif event == "error":
                    raise RuntimeError(f"CUDA batch {index} on GPU {device} failed:\n{payload}")
                elif event == "done":
                    results[index], report = payload
                    reports[index] = {"batch": index, "report": {**report, "host_gpu_id": device}}
                    active.pop(index, None)
                    completed += 1
            # 完成顺序不影响批次标签、返回顺序及 SQD 的最低能选择。
            if report_callback is not None:
                for record in reports:
                    report_callback(record)
            return results
        except BaseException:
            self.close()
            raise
        finally:
            self.running = False


def _make_batch_solver(kind, report_callback, device_ids, worker_timeout, options):
    if device_ids is not None and options.get("sbd_context") is not None:
        raise ValueError("A CUDA context cannot be shared across worker processes; construct it in its owner process")
    pool = None if device_ids is None else _DeviceBatchPool(
        device_ids, kind, int(options.get("threads", 2)), worker_timeout
    )

    def solve_batch(batches, h1e, eri, norb, nelec):
        batches = list(batches)
        if pool is not None:
            return pool.solve(batches, h1e, eri, norb, nelec, options, report_callback)
        results = []
        for index, strings in enumerate(batches):
            result, report = _solve_one(kind, strings, h1e, eri, norb, nelec, options)
            results.append(result)
            if report_callback is not None:
                report_callback({"batch": index, "report": report})
        return results

    solve_batch.close = pool.close if pool is not None else lambda: None
    return solve_batch


def make_cuda_sci_solver(*, report_callback=None, device_ids=None, worker_timeout=600, **solver_options):
    """返回 ``sci_solver(batches, h1e, eri, norb, nelec)``。

    数值和显存参数原样传给 ``solve_sci_cuda``；逐批执行以限制显存峰值。
    ``report_callback`` 接收批编号及 CPU/GPU 独立验收报告。
    仅实际调用时导入 PyTorch，创建配置不要求本机具有 CUDA。
    """
    return _make_batch_solver("hybrid", report_callback, device_ids, worker_timeout, dict(solver_options))


def make_sbd_sci_solver(*, mode="native", report_callback=None, device_ids=None,
                        worker_timeout=600, **solver_options):
    """可选 SBD TPB 批接口；构造函数不加载 CUDA/MPI 或编译扩展。

    ``device_ids`` 为物理设备编号；多卡按独立批次并行。使用结束调用
    返回回调的 ``close()`` 释放工作进程。单任务 MPI 使用独立基准入口。
    """
    if mode not in ("native", "hybrid"):
        raise ValueError("mode must be 'native' or 'hybrid'")
    options = dict(solver_options)
    if mode == "native" and options.get("spin_sq") is not None:
        raise ValueError("Native SBD does not implement spin_sq; use mode='hybrid'")
    if mode == "hybrid":
        options.update(hamiltonian_backend="sbd", rdm_backend="sbd")
    return _make_batch_solver(mode, report_callback, device_ids, worker_timeout, options)
