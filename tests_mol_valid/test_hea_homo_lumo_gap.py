#!/usr/bin/env python3
"""
测试 HEA 类的 HOMO-LUMO gap 计算功能
"""
import numpy as np
from pyscf import gto
from tyxonq.applications.chem import HEA
from tyxonq.applications.chem.molecule import h2, h4, water


def test_hea_closed_shell_h2():
    """测试 HEA 闭壳系统 (H2)"""
    print("=" * 70)
    print("测试 1: HEA 闭壳系统 H2 分子")
    print("=" * 70)
    
    hea = HEA(molecule=h2, layers=1, mapping="parity")
    
    # 使用 property 访问
    gap = hea.homo_lumo_gap
    print(f"HOMO-LUMO gap (property): {gap:.8f} Hartree")
    
    # 使用详细方法（默认只返回Hartree）
    gap_info = hea.get_homo_lumo_gap()
    print(f"\n详细信息:")
    print(f"  系统类型: {gap_info['system_type']}")
    print(f"  HOMO index: {gap_info['homo_idx']}")
    print(f"  LUMO index: {gap_info['lumo_idx']}")
    print(f"  HOMO energy: {gap_info['homo_energy']:.8f} Hartree")
    print(f"  LUMO energy: {gap_info['lumo_energy']:.8f} Hartree")
    print(f"  Gap: {gap_info['gap']:.8f} Hartree")
    
    # 测试include_ev=True
    gap_info_ev = hea.get_homo_lumo_gap(include_ev=True)
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
    print("\n✓ HEA H2 测试通过!")


def test_hea_vs_ucc_consistency():
    """测试 HEA 和 UCC 的 HOMO-LUMO gap 一致性"""
    print("\n" + "=" * 70)
    print("测试 2: HEA 与 UCC 的 HOMO-LUMO gap 一致性")
    print("=" * 70)
    
    from tyxonq.applications.chem import UCCSD
    
    # 使用相同的分子
    mol = h4
    
    # 创建 HEA 和 UCCSD 实例
    hea = HEA(molecule=mol, layers=2, mapping="parity")
    ucc = UCCSD(mol, init_method="zeros")
    
    # 获取 gap 信息
    hea_gap = hea.get_homo_lumo_gap()
    ucc_gap = ucc.get_homo_lumo_gap()
    
    print(f"HEA gap: {hea_gap['gap']:.8f} Hartree")
    print(f"UCC gap: {ucc_gap['gap']:.8f} Hartree")
    print(f"HOMO indices: HEA={hea_gap['homo_idx']}, UCC={ucc_gap['homo_idx']}")
    print(f"LUMO indices: HEA={hea_gap['lumo_idx']}, UCC={ucc_gap['lumo_idx']}")
    
    # 验证一致性（应该完全相同，因为都基于相同的HF计算）
    assert np.isclose(hea_gap['gap'], ucc_gap['gap'], atol=1e-10)
    assert hea_gap['homo_idx'] == ucc_gap['homo_idx']
    assert hea_gap['lumo_idx'] == ucc_gap['lumo_idx']
    assert hea_gap['system_type'] == ucc_gap['system_type']
    
    print("  ✓ HEA 与 UCC 结果完全一致!")
    print("\n✓ 一致性测试通过!")


def test_hea_from_integral_error():
    """测试 HEA 从积分构建时无法计算 HOMO-LUMO gap"""
    print("\n" + "=" * 70)
    print("测试 3: HEA 从积分构建的错误处理")
    print("=" * 70)
    
    # 从积分构建 HEA（没有分子信息）
    int1e = np.array([[0, -1], [-1, 0]])
    int2e = np.zeros((2, 2, 2, 2))
    hea = HEA.from_integral(int1e, int2e, n_elec=2, e_core=0.0, n_layers=1)
    
    print("从积分构建的 HEA：")
    print(f"  n_qubits: {hea.n_qubits}")
    print(f"  layers: {hea.layers}")
    print(f"  _ucc_object: {hea._ucc_object}")
    
    # 尝试计算 HOMO-LUMO gap（应该失败）
    try:
        gap = hea.get_homo_lumo_gap()
        assert False, "Should raise RuntimeError"
    except RuntimeError as e:
        print(f"\n预期的错误: {e}")
        assert "HOMO-LUMO gap calculation requires HEA to be constructed from molecule" in str(e)
        print("  ✓ 错误处理正确!")
    
    print("\n✓ 从积分构建错误处理测试通过!")


def main():
    """运行所有 HEA HOMO-LUMO gap 测试"""
    print("\n🧪 开始测试 HEA HOMO-LUMO gap 计算功能\n")
    
    try:
        test_hea_closed_shell_h2()
        test_hea_vs_ucc_consistency()
        test_hea_from_integral_error()
        
        print("\n" + "=" * 70)
        print("✅ 所有 HEA HOMO-LUMO gap 测试通过!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())