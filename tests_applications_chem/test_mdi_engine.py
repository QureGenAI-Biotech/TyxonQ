"""mdi_engine 测试（用例 10）。

依据 ``MD_INTEGRATION_PLAN.md`` §3 P4 / §4 用例表：
双进程 TCP 回环（driver = 本进程，engine = 子进程）验证命令回环与静电嵌入：

- 无 ``>LATTICE`` 时：(E, F) 与直接调 ``qc_scanner`` 一致（原子单位，无换算）；
- 有 ``>NLATTICE``/``>CLATTICE``/``>LATTICE`` 时：能量位移量级 ≈ 1e-3 Ha
  （MM 电荷确实进了哈密顿量），(E, F) 与带 ``mm_charges`` 的 scanner 一致，
  MM 行力 = 反作用力 ``mm_gradient``；
- 二次推 ``>LATTICE``（MM 坐标变化）每步更新生效（``set_mm_charges`` 重入）；
- 先裸 ``>COORDS`` 后推 ``>LATTICE``（E9 教程时序）：首次嵌入生效且含反作用力，
  锁住「裸建后重入装饰返回新对象被丢弃」的 bug。

全程 ``method="uccsd"``；无 ``mdi``（PyPI ``pymdi``）时 skip。
"""

from __future__ import annotations

import multiprocessing
import time

import numpy as np
import pytest


def _has_pyscf():
    try:
        import pyscf  # noqa: F401
        return True
    except Exception:
        return False


def _has_mdi():
    try:
        import mdi  # noqa: F401
        return True
    except Exception:
        return False


needs_pyscf = pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping.")
needs_mdi = pytest.mark.skipif(not _has_mdi(), reason="pymdi not installed; skipping.")

BOHR_TO_ANGSTROM = 0.52917721092

# 水几何（Å → Bohr）
H2O_ANG = np.array([(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)])
H2O_BOHR = H2O_ANG / BOHR_TO_ANGSTROM
MM_SHIFT_ANG = np.array([2.9, 0.8, 0.3])
# 全体系（两水）坐标，Bohr：>COORDS 协议传全体系，引擎内部自取 QM 子集
DIMER_BOHR = np.vstack([H2O_BOHR, (H2O_ANG + MM_SHIFT_ANG) / BOHR_TO_ANGSTROM])
TIP3P_MM_CHARGES = [-0.834, 0.417, 0.417]


def _engine_proc(port, symbols, qm_indices):
    """子进程：起 TyxonQMdiEngine。异常打印后退出（父进程以超时/断言捕获）。"""
    import traceback

    try:
        from tyxonq.applications.chem.interfaces.mdi_engine import TyxonQMdiEngine

        engine = TyxonQMdiEngine(symbols, qm_indices, active_space=(4, 4),
                                 method="uccsd")
        engine.run(hostname="localhost", port=port)
    except Exception:
        traceback.print_exc()


def _spawn_engine(port, symbols, qm_indices):
    p = multiprocessing.Process(target=_engine_proc, args=(port, symbols, qm_indices))
    p.start()
    time.sleep(1.0)  # engine 先 listen
    return p


def _driver(port):
    import mdi

    mdi.MDI_Init(f"-role DRIVER -name MM -method TCP -port {port} -hostname localhost", None)
    return mdi, mdi.MDI_Accept_Communicator()


def _send_cmd(mdi, comm, cmd):
    mdi.MDI_Send_Command(cmd, comm)


@needs_pyscf
@needs_mdi
def test_mdi_loopback_bare_matches_scanner():
    """无嵌入：命令回环 + (E, F) 与裸 scanner 一致（原子单位）；元数据命令正确。"""
    from tyxonq.applications.chem.interfaces import qc_scanner

    port = 38177
    proc = _spawn_engine(port, ["O", "H", "H"], [0, 1, 2])
    try:
        mdi, comm = _driver(port)

        _send_cmd(mdi, comm, "<NATOMS")
        assert int(np.asarray(mdi.MDI_Recv(1, mdi.MDI_INT, comm)).flat[0]) == 3
        _send_cmd(mdi, comm, "<DIMENSIONS")
        assert int(np.asarray(mdi.MDI_Recv(1, mdi.MDI_INT, comm)).flat[0]) == 3
        _send_cmd(mdi, comm, "<ELEMENTS")
        assert list(mdi.MDI_Recv(3, mdi.MDI_INT, comm)) == [8, 1, 1]
        _send_cmd(mdi, comm, "<MASSES")
        masses = mdi.MDI_Recv(3, mdi.MDI_DOUBLE, comm)
        assert masses[0] == pytest.approx(15.999, abs=0.01) and masses[1] > 0.9
        _send_cmd(mdi, comm, "<TOTCHARGE")
        assert mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm) == pytest.approx(0.0)
        _send_cmd(mdi, comm, "<ELEC_MULT")
        assert int(np.asarray(mdi.MDI_Recv(1, mdi.MDI_INT, comm)).flat[0]) == 1

        _send_cmd(mdi, comm, ">COORDS")
        mdi.MDI_Send(H2O_BOHR.reshape(-1).tolist(), 9, mdi.MDI_DOUBLE, comm)
        _send_cmd(mdi, comm, "<ENERGY")
        e_mdi = mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm)
        _send_cmd(mdi, comm, "<FORCES")
        f_mdi = np.asarray(mdi.MDI_Recv(9, mdi.MDI_DOUBLE, comm)).reshape(3, 3)
        _send_cmd(mdi, comm, "EXIT")

        scan = qc_scanner([("O", tuple(H2O_ANG[0])), ("H", tuple(H2O_ANG[1])),
                           ("H", tuple(H2O_ANG[2]))],
                          basis="sto-3g", active_space=(4, 4), method="uccsd")
        e_ref, de_ref = scan(H2O_BOHR)
        assert e_mdi == pytest.approx(float(e_ref), rel=1e-6)
        np.testing.assert_allclose(f_mdi, -np.asarray(de_ref), rtol=1e-4, atol=1e-6)
    finally:
        proc.join(timeout=60)
        if proc.is_alive():
            proc.terminate()
            pytest.fail("MDI engine did not exit cleanly.")


@needs_pyscf
@needs_mdi
def test_mdi_lattice_embedding_matches_scanner():
    """静电嵌入：>NLATTICE/>CLATTICE/>LATTICE 后 E/F 与带 mm_charges 的 scanner 一致。"""
    port = 38178
    proc = _spawn_engine(port, ["O", "H", "H", "O", "H", "H"], [0, 1, 2])
    try:
        mdi, comm = _driver(port)
        mm_pos_bohr = (H2O_ANG + MM_SHIFT_ANG) / BOHR_TO_ANGSTROM

        _send_cmd(mdi, comm, ">NLATTICE")
        mdi.MDI_Send(3, 1, mdi.MDI_INT, comm)
        _send_cmd(mdi, comm, ">CLATTICE")
        mdi.MDI_Send(TIP3P_MM_CHARGES, 3, mdi.MDI_DOUBLE, comm)
        _send_cmd(mdi, comm, ">LATTICE")
        mdi.MDI_Send(mm_pos_bohr.reshape(-1).tolist(), 9, mdi.MDI_DOUBLE, comm)

        _send_cmd(mdi, comm, ">COORDS")
        mdi.MDI_Send(DIMER_BOHR.reshape(-1).tolist(), 18, mdi.MDI_DOUBLE, comm)
        _send_cmd(mdi, comm, "<ENERGY")
        e_mdi = mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm)
        _send_cmd(mdi, comm, "<FORCES")
        f_mdi = np.asarray(mdi.MDI_Recv(18, mdi.MDI_DOUBLE, comm)).reshape(6, 3)
        _send_cmd(mdi, comm, "EXIT")

        from tyxonq.applications.chem.interfaces import qc_scanner

        scan = qc_scanner([("O", tuple(H2O_ANG[0])), ("H", tuple(H2O_ANG[1])),
                           ("H", tuple(H2O_ANG[2]))],
                          basis="sto-3g", active_space=(4, 4), method="uccsd",
                          unit="Bohr",
                          mm_charges=(mm_pos_bohr, np.array(TIP3P_MM_CHARGES)))
        e_ref, de_ref = scan(H2O_BOHR)
        de_mm_ref = scan.mm_gradient()

        # 嵌入位移：与裸算差 ~3e-3 Ha（§4.1 同类量级，证明电荷进了哈密顿量）
        scan_bare = qc_scanner([("O", tuple(H2O_ANG[0])), ("H", tuple(H2O_ANG[1])),
                                ("H", tuple(H2O_ANG[2]))],
                               basis="sto-3g", active_space=(4, 4), method="uccsd",
                               unit="Bohr")
        e_bare, _ = scan_bare(H2O_BOHR)
        assert abs(e_mdi - float(e_bare)) > 1e-4

        assert e_mdi == pytest.approx(float(e_ref), rel=1e-6)
        np.testing.assert_allclose(f_mdi[:3], -np.asarray(de_ref), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(f_mdi[3:], -np.asarray(de_mm_ref), rtol=1e-4, atol=1e-6)
    finally:
        proc.join(timeout=60)
        if proc.is_alive():
            proc.terminate()
            pytest.fail("MDI engine did not exit cleanly.")


@needs_pyscf
@needs_mdi
def test_mdi_lattice_after_bare_first_frame():
    """先裸 >COORDS 后推 >LATTICE（E9 教程时序）：首次嵌入仍生效（重建路径）。"""
    port = 38180
    proc = _spawn_engine(port, ["O", "H", "H", "O", "H", "H"], [0, 1, 2])
    try:
        mdi, comm = _driver(port)
        mm_pos_bohr = (H2O_ANG + MM_SHIFT_ANG) / BOHR_TO_ANGSTROM

        # 首帧无 lattice：裸算一次（引擎已裸建 scanner）
        _send_cmd(mdi, comm, ">COORDS")
        mdi.MDI_Send(DIMER_BOHR.reshape(-1).tolist(), 18, mdi.MDI_DOUBLE, comm)
        _send_cmd(mdi, comm, "<ENERGY")
        e_bare = mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm)

        # 随后推入晶格：裸建后首次带电荷必须生效（重建而非丢弃新对象）
        _send_cmd(mdi, comm, ">NLATTICE")
        mdi.MDI_Send(3, 1, mdi.MDI_INT, comm)
        _send_cmd(mdi, comm, ">CLATTICE")
        mdi.MDI_Send(TIP3P_MM_CHARGES, 3, mdi.MDI_DOUBLE, comm)
        _send_cmd(mdi, comm, ">LATTICE")
        mdi.MDI_Send(mm_pos_bohr.reshape(-1).tolist(), 9, mdi.MDI_DOUBLE, comm)

        _send_cmd(mdi, comm, ">COORDS")
        mdi.MDI_Send(DIMER_BOHR.reshape(-1).tolist(), 18, mdi.MDI_DOUBLE, comm)
        _send_cmd(mdi, comm, "<ENERGY")
        e_mdi = mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm)
        _send_cmd(mdi, comm, "<FORCES")
        f_mdi = np.asarray(mdi.MDI_Recv(18, mdi.MDI_DOUBLE, comm)).reshape(6, 3)
        _send_cmd(mdi, comm, "EXIT")

        from tyxonq.applications.chem.interfaces import qc_scanner

        scan = qc_scanner([("O", tuple(H2O_ANG[0])), ("H", tuple(H2O_ANG[1])),
                           ("H", tuple(H2O_ANG[2]))],
                          basis="sto-3g", active_space=(4, 4), method="uccsd",
                          unit="Bohr",
                          mm_charges=(mm_pos_bohr, np.array(TIP3P_MM_CHARGES)))
        e_ref, de_ref = scan(H2O_BOHR)
        de_mm_ref = scan.mm_gradient()

        assert abs(e_mdi - e_bare) > 1e-4, "裸建后首次嵌入未生效"
        assert e_mdi == pytest.approx(float(e_ref), rel=1e-6)
        np.testing.assert_allclose(f_mdi[:3], -np.asarray(de_ref), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(f_mdi[3:], -np.asarray(de_mm_ref), rtol=1e-4, atol=1e-6)
    finally:
        proc.join(timeout=60)
        if proc.is_alive():
            proc.terminate()
            pytest.fail("MDI engine did not exit cleanly.")


@needs_pyscf
@needs_mdi
def test_mdi_lattice_per_step_update():
    """每步更新：二次推 >LATTICE（MM 水位移 0.1 Å）后能量跟随新环境（重入路径）。"""
    port = 38179
    proc = _spawn_engine(port, ["O", "H", "H", "O", "H", "H"], [0, 1, 2])
    try:
        mdi, comm = _driver(port)
        mm0 = (H2O_ANG + MM_SHIFT_ANG) / BOHR_TO_ANGSTROM
        mm1 = mm0 + np.array([0.1, 0.0, 0.0]) / BOHR_TO_ANGSTROM

        energies = []
        for mm_pos in (mm0, mm1):
            _send_cmd(mdi, comm, ">NLATTICE")
            mdi.MDI_Send(3, 1, mdi.MDI_INT, comm)
            _send_cmd(mdi, comm, ">CLATTICE")
            mdi.MDI_Send(TIP3P_MM_CHARGES, 3, mdi.MDI_DOUBLE, comm)
            _send_cmd(mdi, comm, ">LATTICE")
            mdi.MDI_Send(mm_pos.reshape(-1).tolist(), 9, mdi.MDI_DOUBLE, comm)
            _send_cmd(mdi, comm, ">COORDS")
            mdi.MDI_Send(DIMER_BOHR.reshape(-1).tolist(), 18, mdi.MDI_DOUBLE, comm)
            _send_cmd(mdi, comm, "<ENERGY")
            energies.append(mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm))
        _send_cmd(mdi, comm, "EXIT")

        from tyxonq.applications.chem.interfaces import qc_scanner

        refs = []
        for mm_pos in (mm0, mm1):
            scan = qc_scanner([("O", tuple(H2O_ANG[0])), ("H", tuple(H2O_ANG[1])),
                               ("H", tuple(H2O_ANG[2]))],
                              basis="sto-3g", active_space=(4, 4), method="uccsd",
                              unit="Bohr",
                              mm_charges=(mm_pos, np.array(TIP3P_MM_CHARGES)))
            e, _ = scan(H2O_BOHR)
            refs.append(float(e))

        assert energies[0] != pytest.approx(energies[1], abs=1e-7)  # 更新确实生效
        assert energies[0] == pytest.approx(refs[0], rel=1e-6)
        assert energies[1] == pytest.approx(refs[1], rel=1e-6)
    finally:
        proc.join(timeout=60)
        if proc.is_alive():
            proc.terminate()
            pytest.fail("MDI engine did not exit cleanly.")
