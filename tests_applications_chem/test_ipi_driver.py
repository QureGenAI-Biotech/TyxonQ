"""TyxonQDriver（i-PI）测试（用例 8）。

依据 ``MD_INTEGRATION_PLAN.md`` §3 P2 / §4 用例表：起一个最小 i-PI socket
server（本测试扮演 server 端），向子进程形式的 ``i-pi-driver-py -m custom``
发 ``POSDATA`` / ``GETFORCE``，收到的 E/F 必须与直接调 scanner 一致。

协议依据 i-PI 3.3.0 源码（driver 侧循环见其 ``i-pi-driver-py`` 入口脚本，
server 侧即本文件实现）：12 字节大写头部；POSDATA 载荷为原子单位
（Bohr）；GETFORCE 回包为 (pot Hartree, nat, force Hartree/Bohr, vir, extras)。
"""

from __future__ import annotations

import shutil
import socket
import struct
import subprocess

import numpy as np
import pytest


def _has(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


needs_all = pytest.mark.skipif(
    not (_has("pyscf") and _has("ase") and _has("ipi")),
    reason="pyscf/ase/ipi not all installed; skipping i-PI driver test.",
)

HDRLEN = 12
BOHR_TO_ANGSTROM = 0.52917721092

# 水分子几何（Å），与 test_ase_calculator 同一参考几何
H2O_POS_ANG = np.array([(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)])

XYZ = """3
water template for TyxonQDriver
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
"""


def _msg(s):
    return s.upper().ljust(HDRLEN).encode()


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        part = sock.recv(n - len(buf))
        if not part:
            raise RuntimeError("socket disconnected")
        buf += part
    return buf


def _recv_header(sock):
    return _recv_exact(sock, HDRLEN).decode().strip()


@needs_all
def test_driver_socket_roundtrip_matches_scanner(tmp_path):
    """server 发 POSDATA/GETFORCE，driver 回包与裸 scanner 一致。"""
    from tyxonq.applications.chem.interfaces import qc_scanner
    import tyxonq.applications.chem.interfaces.ipi_driver as ipi_mod

    driver_bin = shutil.which("i-pi-driver-py")
    if driver_bin is None:
        pytest.skip("i-pi-driver-py not on PATH")

    xyz = tmp_path / "water.xyz"
    xyz.write_text(XYZ)

    # ---- server 端 socket ----
    # 注意必须绑 127.0.0.1 而不是 "localhost"：driver 用 AF_INET（IPv4），
    # 而 macOS 上 "localhost" 可能解析到 ::1，双栈不一致会连不上。
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.listen(1)
    server.settimeout(120)

    proc = subprocess.Popen(
        [
            driver_bin, "-p", str(port),
            "-m", "custom", "-P", ipi_mod.__file__,
            "-o", f"{xyz},basis=sto-3g,active_space=4 4,method=uccsd",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        conn, _ = server.accept()
        conn.settimeout(120)

        # STATUS → NEEDINIT；INIT（先 12 字节头部，再 rid/len/串）；STATUS → READY
        conn.sendall(_msg("STATUS"))
        assert _recv_header(conn) == "NEEDINIT"
        init = b"tyxonq-test"
        conn.sendall(_msg("INIT"))
        conn.sendall(struct.pack("ii", 0, len(init)) + init)
        conn.sendall(_msg("STATUS"))
        assert _recv_header(conn) == "READY"

        # POSDATA（原子单位 Bohr）
        pos_bohr = H2O_POS_ANG / BOHR_TO_ANGSTROM
        nat = len(pos_bohr)
        cell = np.eye(3) * 20.0  # 任意大盒子；分子体系无 pbc，calculator 忽略
        conn.sendall(_msg("POSDATA"))
        conn.sendall(cell.astype(np.float64).tobytes())
        conn.sendall(np.linalg.inv(cell).astype(np.float64).tobytes())
        conn.sendall(struct.pack("i", nat))
        conn.sendall(pos_bohr.astype(np.float64).tobytes())

        # GETFORCE → FORCEREADY + (pot, nat, force, vir, extras)
        conn.sendall(_msg("GETFORCE"))
        assert _recv_header(conn) == "FORCEREADY"
        pot = struct.unpack("d", _recv_exact(conn, 8))[0]
        (n2,) = struct.unpack("i", _recv_exact(conn, 4))
        assert n2 == nat
        force = np.frombuffer(_recv_exact(conn, 8 * nat * 3), np.float64).reshape(nat, 3)
        _recv_exact(conn, 8 * 9)  # virial：has_stress=False，应为零，不深究
        (elen,) = struct.unpack("i", _recv_exact(conn, 4))
        _recv_exact(conn, elen)

        conn.sendall(_msg("EXIT"))
        out, _ = proc.communicate(timeout=60)
        server.close()
    except BaseException:
        # 失败时先 dump driver 子进程输出，方便定位协议/崩溃问题
        if proc.poll() is None:
            proc.kill()
        out, _ = proc.communicate()
        server.close()
        print("---- driver output ----\n" + out.decode(errors="replace"))
        raise

    # ---- 与裸 scanner 对比 ----
    scan = qc_scanner(
        "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        basis="sto-3g", active_space=(4, 4), method="uccsd",
    )
    e_ha, de_ha = scan(pos_bohr)

    # 容差说明：i-PI 在 post_process 里把 ASE 的 eV/(eV·Å⁻¹) 换算回原子单位，
    # 其 ipi.utils.units 常数只有约 8 位有效数字（如 eV→Hartree 为 0.036749326，
    # 精确值 0.03674932217…）。回转引入 ~1e-7 相对误差，对 -2040 eV 的水分子
    # 约 7.5e-6 Hartree。spy 实验证实计算器内部能量正确（-74.9704540926），
    # 偏差全部来自上游单位常数，故按 1e-5 Hartree 绝对容差断言。
    assert pot == pytest.approx(e_ha, abs=1e-5)
    # i-PI 内部力的单位是 Hartree/Bohr，方向为 -dE/dR
    np.testing.assert_allclose(force, -np.asarray(de_ha), rtol=1e-5, atol=1e-7)
    assert proc.returncode == 0
