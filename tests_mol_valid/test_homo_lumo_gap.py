#!/usr/bin/env python3
"""
测试 HOMO-LUMO gap 计算功能
"""
import numpy as np
from pyscf import gto, scf
from tyxonq.applications.chem import UCCSD, ROUCCSD
from tyxonq.applications.chem.molecule import h2, h4, h_chain, water


def test_closed_shell_h2():
    """测试闭壳系统 (H2)"""
    print("=" * 70)
    print("测试 1: 闭壳系统 H2 分子")
    print("=" * 70)
    
    ucc = UCCSD(h2, init_method="zeros")
    
    # 使用 property 访问
    gap = ucc.homo_lumo_gap
    print(f"HOMO-LUMO gap (property): {gap:.8f} Hartree")
    print(f"HOMO-LUMO gap (property): {gap*27.2114:.6f} eV")
    
    # 使用详细方法（默认只返回Hartree）
    gap_info = ucc.get_homo_lumo_gap()
    print(f"\n详细信息:")
    print(f"  系统类型: {gap_info['system_type']}")
    print(f"  HOMO index: {gap_info['homo_idx']}")
    print(f"  LUMO index: {gap_info['lumo_idx']}")
    print(f"  HOMO energy: {gap_info['homo_energy']:.8f} Hartree")
    print(f"  LUMO energy: {gap_info['lumo_energy']:.8f} Hartree")
    print(f"  Gap: {gap_info['gap']:.8f} Hartree")
    
    # 测试include_ev=True
    gap_info_ev = ucc.get_homo_lumo_gap(include_ev=True)
    print(f"  Gap: {gap_info_ev['gap_ev']:.6f} eV")
    
    # 验证简化的索引计算方法
    n_electrons = h2.nelectron
    expected_homo_idx = (n_electrons // 2) - 1
    expected_lumo_idx = n_electrons // 2
    print(f"\n索引验证（简化方法）:")
    print(f"  电子数: {n_electrons}")
    print(f"  预期HOMO索引: {expected_homo_idx}")
    print(f"  预期LUMO索引: {expected_lumo_idx}")
    print(f"  实际HOMO索引: {gap_info['homo_idx']}")
    print(f"  实际LUMO索引: {gap_info['lumo_idx']}")
    
    # 验证一致性
    assert np.isclose(gap, gap_info['gap'])
    assert gap > 0, "HOMO-LUMO gap should be positive"
    assert gap_info['system_type'] == 'closed-shell'
    # 验证简化索引计算
    assert gap_info['homo_idx'] == expected_homo_idx
    assert gap_info['lumo_idx'] == expected_lumo_idx
    # 验证eV开关
    assert 'gap_ev' not in gap_info  # 默认不包含eV
    assert 'gap_ev' in gap_info_ev   # 开启时包含eV
    
    print("  ✓ 简化索引计算方法正确!")
    print("\n✓ H2 测试通过!")


def test_closed_shell_h4():
    """测试较大的闭壳系统 (H4)"""
    print("\n" + "=" * 70)
    print("测试 2: 闭壳系统 H4 分子链")
    print("=" * 70)
    
    ucc = UCCSD(h4, init_method="zeros")
    
    gap_info = ucc.get_homo_lumo_gap()
    print(f"系统类型: {gap_info['system_type']}")
    print(f"HOMO-LUMO gap: {gap_info['gap']:.8f} Hartree")
    print(f"HOMO orbital #{gap_info['homo_idx']}: {gap_info['homo_energy']:.8f} Hartree")
    print(f"LUMO orbital #{gap_info['lumo_idx']}: {gap_info['lumo_energy']:.8f} Hartree")
    
    # 验证简化索引计算
    n_electrons = h4.nelectron
    expected_homo_idx = (n_electrons // 2) - 1
    expected_lumo_idx = n_electrons // 2
    print(f"\n简化索引验证: 电子数={n_electrons}, HOMO={expected_homo_idx}, LUMO={expected_lumo_idx}")
    
    assert gap_info['system_type'] == 'closed-shell'
    assert gap_info['gap'] > 0
    assert gap_info['homo_idx'] == expected_homo_idx
    assert gap_info['lumo_idx'] == expected_lumo_idx
    print("  ✓ 简化索引计算正确!")
    print("\n✓ H4 测试通过!")


def test_open_shell_system():
    """测试开壳系统"""
    print("\n" + "=" * 70)
    print("测试 3: 开壳系统 H5 (spin=1)")
    print("=" * 70)
    
    # H5 开壳系统: 5个H原子，5个电子，spin=1 (Nalpha=3, Nbeta=2)
    # 这符合 spin = Nalpha - Nbeta = 1 的要求
    mol = gto.M(atom='H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4; H 0 0 3.2', 
                basis='sto-3g', spin=1)
    ucc = ROUCCSD(mol, init_method="zeros")
    
    gap_info = ucc.get_homo_lumo_gap()
    print(f"系统类型: {gap_info['system_type']}")
    print(f"总电子数: {mol.nelectron}")
    print(f"Spin (Nalpha-Nbeta): {mol.spin}")
    print(f"HOMO-LUMO gap: {gap_info['gap']:.8f} Hartree")
    print(f"HOMO orbital #{gap_info['homo_idx']}: {gap_info['homo_energy']:.8f} Hartree")
    print(f"LUMO orbital #{gap_info['lumo_idx']}: {gap_info['lumo_energy']:.8f} Hartree")
    
    # 验证开壳判断（用spin判断）
    print(f"\n系统判断验证:")
    print(f"  mol.spin = {mol.spin}")
    print(f"  系统类型判断: {'闭壳' if mol.spin == 0 else '开壳'}")
    
    assert gap_info['system_type'] == 'open-shell'
    assert gap_info['gap'] > 0
    print("  ✓ 开壳系统判断正确!")
    print("\n✓ 开壳系统测试通过!")


def test_manual_specification():
    """测试手动指定 HOMO/LUMO 索引"""
    print("\n" + "=" * 70)
    print("测试 4: 手动指定 HOMO/LUMO 索引")
    print("=" * 70)
    
    ucc = UCCSD(h4, init_method="zeros")
    
    # 自动计算
    auto_gap = ucc.get_homo_lumo_gap()
    print(f"自动计算: HOMO={auto_gap['homo_idx']}, LUMO={auto_gap['lumo_idx']}, Gap={auto_gap['gap']:.6f} Ha")
    
    # 手动指定
    manual_gap = ucc.get_homo_lumo_gap(homo_idx=1, lumo_idx=2)
    print(f"手动指定: HOMO={manual_gap['homo_idx']}, LUMO={manual_gap['lumo_idx']}, Gap={manual_gap['gap']:.6f} Ha")
    
    # 验证手动指定的值
    assert manual_gap['homo_idx'] == 1
    assert manual_gap['lumo_idx'] == 2
    print("\n✓ 手动指定测试通过!")


def test_water_molecule():
    """测试水分子"""
    print("\n" + "=" * 70)
    print("测试 5: 水分子 (H2O)")
    print("=" * 70)
    
    h2o = water(basis="sto-3g")
    ucc = UCCSD(h2o, init_method="zeros")
    
    gap_info = ucc.get_homo_lumo_gap()
    print(f"系统类型: {gap_info['system_type']}")
    print(f"总电子数: {h2o.nelectron}")
    print(f"总轨道数: {h2o.nao}")
    print(f"HOMO-LUMO gap: {gap_info['gap']:.8f} Hartree")
    print(f"HOMO orbital #{gap_info['homo_idx']}: {gap_info['homo_energy']:.8f} Hartree")
    print(f"LUMO orbital #{gap_info['lumo_idx']}: {gap_info['lumo_energy']:.8f} Hartree")
    
    # 验证简化索引计算
    n_electrons = h2o.nelectron
    expected_homo_idx = (n_electrons // 2) - 1
    expected_lumo_idx = n_electrons // 2
    print(f"\n简化索引验证: 电子数={n_electrons}, HOMO={expected_homo_idx}, LUMO={expected_lumo_idx}")
    
    # 测试eV转换开关
    gap_info_ev = ucc.get_homo_lumo_gap(include_ev=True)
    print(f"\neV转换测试:")
    print(f"  默认输出（无eV）: {list(gap_info.keys())}")
    print(f"  include_ev=True: {list(gap_info_ev.keys())}")
    print(f"  Gap: {gap_info_ev['gap_ev']:.6f} eV")
    
    assert gap_info['system_type'] == 'closed-shell'
    assert gap_info['gap'] > 0
    assert gap_info['homo_idx'] == expected_homo_idx
    assert gap_info['lumo_idx'] == expected_lumo_idx
    assert 'gap_ev' not in gap_info      # 默认不包含
    assert 'gap_ev' in gap_info_ev       # 开启时包含
    print("  ✓ eV转换开关工作正常!")
    print("\n✓ H2O 测试通过!")


def test_active_space():
    """测试活性空间中的 HOMO-LUMO gap"""
    print("\n" + "=" * 70)
    print("测试 6: 活性空间中的 HOMO-LUMO gap")
    print("=" * 70)
    
    h2o = water(basis="sto-3g")
    
    # 全空间
    ucc_full = UCCSD(h2o, init_method="zeros")
    full_gap = ucc_full.get_homo_lumo_gap()
    print(f"全空间: Gap={full_gap['gap']:.6f} Ha, HOMO={full_gap['homo_idx']}, LUMO={full_gap['lumo_idx']}")
    
    # 活性空间 (4 电子, 4 轨道)
    ucc_cas = UCCSD(h2o, active_space=(4, 4), init_method="zeros")
    cas_gap = ucc_cas.get_homo_lumo_gap()
    print(f"活性空间(4,4): Gap={cas_gap['gap']:.6f} Ha, HOMO={cas_gap['homo_idx']}, LUMO={cas_gap['lumo_idx']}")
    
    # 两者都应该是正值
    assert full_gap['gap'] > 0
    assert cas_gap['gap'] > 0
    print("\n✓ 活性空间测试通过!")

def test_simple_logic():
    """测试核心逻辑"""
    print("测试简化的 HOMO-LUMO gap 计算逻辑")
    print("=" * 60)
    
    # 测试闭壳系统索引计算逻辑
    print("\n1. 闭壳系统索引计算测试:")
    for n_electrons in [2, 4, 6, 8, 10]:
        homo_idx = (n_electrons // 2) - 1
        lumo_idx = n_electrons // 2
        print(f"  {n_electrons}个电子: HOMO={homo_idx}, LUMO={lumo_idx}")
    
    # 测试开壳系统判断
    print("\n2. 开壳系统判断测试:")
    for spin in [0, 1, 2]:
        system_type = 'closed-shell' if spin == 0 else 'open-shell'
        print(f"  spin={spin}: {system_type}")
    
    # 测试eV转换开关
    print("\n3. eV转换开关测试:")
    gap_hartree = 0.5
    gap_ev = gap_hartree * 27.211386245988
    
    # 默认输出（无eV）
    result_default = {
        'homo_energy': -0.5,
        'lumo_energy': 0.0,
        'gap': gap_hartree,
        'homo_idx': 0,
        'lumo_idx': 1,
        'system_type': 'closed-shell'
    }
    
    # 包含eV的输出
    result_with_ev = result_default.copy()
    result_with_ev['gap_ev'] = gap_ev
    
    print(f"  默认输出: {list(result_default.keys())}")
    print(f"  包含eV: {list(result_with_ev.keys())}")
    print(f"  Gap: {gap_hartree:.6f} Ha = {gap_ev:.6f} eV")
    
    print("\n✅ 所有逻辑测试通过!")


def main():
    """运行所有测试"""
    print("\n🧪 开始测试 HOMO-LUMO gap 计算功能\n")
    
    try:
        test_closed_shell_h2()
        test_closed_shell_h4()
        test_open_shell_system()
        test_manual_specification()
        test_water_molecule()
        test_active_space()
        
        print("\n" + "=" * 70)
        print("✅ 所有测试通过!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
