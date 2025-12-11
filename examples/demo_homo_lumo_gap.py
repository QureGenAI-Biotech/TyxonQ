#!/usr/bin/env python3
"""
HOMO-LUMO Gap 功能演示脚本

展示如何使用 UCC 类计算分子的 HOMO-LUMO 能隙
"""

import numpy as np
from tyxonq.applications.chem import UCCSD, ROUCCSD
from tyxonq.applications.chem.molecule import h2, h4, water
from pyscf import gto


def demo_basic_usage():
    """演示基本用法"""
    print("=" * 80)
    print("演示 1: 基本用法 - H2 分子")
    print("=" * 80)
    
    ucc = UCCSD(h2, init_method="zeros")
    
    # 使用 property
    gap = ucc.homo_lumo_gap
    print(f"\n快速访问: gap = {gap:.8f} Hartree = {gap*27.2114:.4f} eV")
    
    # 获取详细信息
    gap_info = ucc.get_homo_lumo_gap()
    print(f"\n详细信息:")
    print(f"  系统类型: {gap_info['system_type']}")
    print(f"  HOMO (orbital #{gap_info['homo_idx']}): {gap_info['homo_energy']:.8f} Hartree")
    print(f"  LUMO (orbital #{gap_info['lumo_idx']}): {gap_info['lumo_energy']:.8f} Hartree")
    print(f"  能隙: {gap_info['gap']:.8f} Hartree ({gap_info['gap_ev']:.4f} eV)")


def demo_molecule_comparison():
    """演示多分子对比"""
    print("\n" + "=" * 80)
    print("演示 2: 多分子 HOMO-LUMO gap 对比")
    print("=" * 80)
    
    molecules = [
        ("H2", h2),
        ("H4", h4),
        ("H2O", water(basis="sto-3g"))
    ]
    
    print(f"\n{'分子':<10} {'电子数':<8} {'轨道数':<8} {'Gap (Ha)':<12} {'Gap (eV)':<10} {'系统类型'}")
    print("-" * 80)
    
    for name, mol in molecules:
        ucc = UCCSD(mol, init_method="zeros")
        gap_info = ucc.get_homo_lumo_gap()
        
        print(f"{name:<10} {mol.nelectron:<8} {mol.nao:<8} "
              f"{gap_info['gap']:<12.8f} {gap_info['gap_ev']:<10.4f} "
              f"{gap_info['system_type']}")


def demo_open_shell():
    """演示开壳系统"""
    print("\n" + "=" * 80)
    print("演示 3: 开壳系统 (ROHF)")
    print("=" * 80)
    
    # H5 链，5个电子，spin=1
    mol = gto.M(atom='H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4; H 0 0 3.2',
                basis='sto-3g', spin=1)
    
    ucc = ROUCCSD(mol, init_method="zeros")
    gap_info = ucc.get_homo_lumo_gap()
    
    print(f"\n分子: H5 链")
    print(f"总电子数: {mol.nelectron}")
    print(f"Spin (Nalpha - Nbeta): {mol.spin}")
    print(f"系统类型: {gap_info['system_type']}")
    print(f"\nHOMO orbital #{gap_info['homo_idx']}: {gap_info['homo_energy']:.8f} Hartree")
    print(f"LUMO orbital #{gap_info['lumo_idx']}: {gap_info['lumo_energy']:.8f} Hartree")
    print(f"HOMO-LUMO gap: {gap_info['gap']:.8f} Hartree ({gap_info['gap_ev']:.4f} eV)")
    
    # 显示轨道占据情况
    mo_occ = ucc.hf.mo_occ
    print(f"\n轨道占据数: {mo_occ}")
    print(f"  双占据轨道: {np.where(mo_occ > 1.5)[0]}")
    print(f"  单占据轨道: {np.where((mo_occ > 0.5) & (mo_occ < 1.5))[0]}")
    print(f"  虚轨道: {np.where(mo_occ < 0.5)[0]}")


def demo_manual_specification():
    """演示手动指定轨道"""
    print("\n" + "=" * 80)
    print("演示 4: 手动指定 HOMO/LUMO 轨道")
    print("=" * 80)
    
    ucc = UCCSD(h4, init_method="zeros")
    
    # 自动计算
    auto_gap = ucc.get_homo_lumo_gap()
    print(f"\n自动计算:")
    print(f"  HOMO = orbital #{auto_gap['homo_idx']}, LUMO = orbital #{auto_gap['lumo_idx']}")
    print(f"  Gap = {auto_gap['gap']:.8f} Hartree")
    
    # 手动指定不同的轨道对
    print(f"\n手动指定不同轨道对:")
    for homo_idx, lumo_idx in [(0, 2), (1, 2), (1, 3)]:
        gap_info = ucc.get_homo_lumo_gap(homo_idx=homo_idx, lumo_idx=lumo_idx)
        print(f"  Orbital {homo_idx} → {lumo_idx}: "
              f"Gap = {gap_info['gap']:.8f} Hartree ({gap_info['gap_ev']:.4f} eV)")


def demo_active_space():
    """演示活性空间"""
    print("\n" + "=" * 80)
    print("演示 5: 活性空间中的 HOMO-LUMO gap")
    print("=" * 80)
    
    h2o = water(basis="sto-3g")
    
    # 全空间
    ucc_full = UCCSD(h2o, init_method="zeros")
    full_gap = ucc_full.get_homo_lumo_gap()
    
    # 活性空间 (4 电子, 4 轨道)
    ucc_cas = UCCSD(h2o, active_space=(4, 4), init_method="zeros")
    cas_gap = ucc_cas.get_homo_lumo_gap()
    
    print(f"\n水分子 H2O:")
    print(f"  全空间 ({h2o.nelectron}e, {h2o.nao}o):")
    print(f"    Gap = {full_gap['gap']:.8f} Hartree ({full_gap['gap_ev']:.4f} eV)")
    print(f"    HOMO = orbital #{full_gap['homo_idx']}, LUMO = orbital #{full_gap['lumo_idx']}")
    
    print(f"\n  活性空间 (4e, 4o):")
    print(f"    Gap = {cas_gap['gap']:.8f} Hartree ({cas_gap['gap_ev']:.4f} eV)")
    print(f"    HOMO = orbital #{cas_gap['homo_idx']}, LUMO = orbital #{cas_gap['lumo_idx']}")


def demo_pyscf_validation():
    """演示与 PySCF 的数值一致性"""
    print("\n" + "=" * 80)
    print("演示 6: 与 PySCF 原始数据的一致性验证")
    print("=" * 80)
    
    ucc = UCCSD(h2, init_method="zeros")
    gap_info = ucc.get_homo_lumo_gap()
    
    # 直接从 PySCF 计算
    mo_energy = ucc.hf.mo_energy
    mo_occ = ucc.hf.mo_occ
    
    homo_idx = np.where(mo_occ > 1.5)[0][-1]
    lumo_idx = np.where(mo_occ < 0.5)[0][0]
    
    pyscf_homo = mo_energy[homo_idx]
    pyscf_lumo = mo_energy[lumo_idx]
    pyscf_gap = pyscf_lumo - pyscf_homo
    
    print(f"\nH2 分子验证:")
    print(f"\n  TyxonQ UCC 结果:")
    print(f"    HOMO: {gap_info['homo_energy']:.12f} Hartree")
    print(f"    LUMO: {gap_info['lumo_energy']:.12f} Hartree")
    print(f"    Gap:  {gap_info['gap']:.12f} Hartree")
    
    print(f"\n  PySCF 直接结果:")
    print(f"    HOMO: {pyscf_homo:.12f} Hartree")
    print(f"    LUMO: {pyscf_lumo:.12f} Hartree")
    print(f"    Gap:  {pyscf_gap:.12f} Hartree")
    
    print(f"\n  差异:")
    print(f"    HOMO 差异: {abs(gap_info['homo_energy'] - pyscf_homo):.2e} Hartree")
    print(f"    LUMO 差异: {abs(gap_info['lumo_energy'] - pyscf_lumo):.2e} Hartree")
    print(f"    Gap 差异:  {abs(gap_info['gap'] - pyscf_gap):.2e} Hartree")
    
    # 验证一致性
    assert np.isclose(gap_info['homo_energy'], pyscf_homo, atol=1e-10)
    assert np.isclose(gap_info['lumo_energy'], pyscf_lumo, atol=1e-10)
    assert np.isclose(gap_info['gap'], pyscf_gap, atol=1e-10)
    print(f"\n  ✓ 数值完全一致! (精度 < 1e-10 Hartree)")


def main():
    """运行所有演示"""
    print("\n" + "🎯 " * 20)
    print("HOMO-LUMO Gap 计算功能演示")
    print("🎯 " * 20)
    
    demo_basic_usage()
    demo_molecule_comparison()
    demo_open_shell()
    demo_manual_specification()
    demo_active_space()
    demo_pyscf_validation()
    
    print("\n" + "=" * 80)
    print("✅ 所有演示完成!")
    print("=" * 80)
    print("\n提示: 查看 HOMO_LUMO_GAP_FEATURE.md 了解更多使用方法")
    print()


if __name__ == "__main__":
    main()
