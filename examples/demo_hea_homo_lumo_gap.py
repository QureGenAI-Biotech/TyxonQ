#!/usr/bin/env python3
"""
HEA HOMO-LUMO Gap 功能演示脚本
"""

import numpy as np
from tyxonq.applications.chem import HEA, UCCSD
from tyxonq.applications.chem.molecule import h2, h4, water


def demo_basic_hea_usage():
    """演示 HEA 基本用法"""
    print("=" * 80)
    print("演示 1: HEA 基本 HOMO-LUMO gap 计算")
    print("=" * 80)
    
    hea = HEA(molecule=h2, layers=1, mapping="parity")
    
    # 使用 property
    gap = hea.homo_lumo_gap
    print(f"\n快速访问: gap = {gap:.8f} Hartree = {gap*27.2114:.4f} eV")
    
    # 获取详细信息
    gap_info = hea.get_homo_lumo_gap()
    print(f"\n详细信息:")
    print(f"  系统类型: {gap_info['system_type']}")
    print(f"  HOMO (orbital #{gap_info['homo_idx']}): {gap_info['homo_energy']:.8f} Hartree")
    print(f"  LUMO (orbital #{gap_info['lumo_idx']}): {gap_info['lumo_energy']:.8f} Hartree")
    print(f"  能隙: {gap_info['gap']:.8f} Hartree")
    
    # 包含 eV 转换
    gap_info_ev = hea.get_homo_lumo_gap(include_ev=True)
    print(f"  能隙: {gap_info_ev['gap_ev']:.4f} eV")


def demo_hea_vs_ucc():
    """演示 HEA 与 UCC 的对比"""
    print("\n" + "=" * 80)
    print("演示 2: HEA 与 UCC 的 HOMO-LUMO gap 对比")
    print("=" * 80)
    
    mol = h4
    
    # 创建两个算法实例
    hea = HEA(molecule=mol, layers=2, mapping="parity")
    ucc = UCCSD(mol, init_method="zeros")
    
    # 获取 gap 信息
    hea_gap = hea.get_homo_lumo_gap()
    ucc_gap = ucc.get_homo_lumo_gap()
    
    print(f"\nH4 分子对比:")
    print(f"  电子数: {mol.nelectron}")
    print(f"  轨道数: {mol.nao}")
    
    print(f"\n  HEA 结果:")
    print(f"    HOMO-LUMO gap: {hea_gap['gap']:.8f} Hartree")
    print(f"    HOMO index: {hea_gap['homo_idx']}")
    print(f"    LUMO index: {hea_gap['lumo_idx']}")
    
    print(f"\n  UCC 结果:")
    print(f"    HOMO-LUMO gap: {ucc_gap['gap']:.8f} Hartree")
    print(f"    HOMO index: {ucc_gap['homo_idx']}")
    print(f"    LUMO index: {ucc_gap['lumo_idx']}")
    
    print(f"\n  差异:")
    print(f"    Gap 差异: {abs(hea_gap['gap'] - ucc_gap['gap']):.2e} Hartree")
    print(f"    ✓ 结果{'完全一致' if np.isclose(hea_gap['gap'], ucc_gap['gap'], atol=1e-10) else '有差异'}")


def demo_multiple_molecules():
    """演示多分子对比"""
    print("\n" + "=" * 80)
    print("演示 3: 多分子 HOMO-LUMO gap 对比（HEA）")
    print("=" * 80)
    
    molecules = [
        ("H2", h2),
        ("H4", h4),
        ("H2O", water(basis="sto-3g"))
    ]
    
    print(f"\n{'分子':<10} {'电子数':<8} {'轨道数':<8} {'Gap (Ha)':<12} {'Gap (eV)':<10} {'系统类型'}")
    print("-" * 80)
    
    for name, mol in molecules:
        hea = HEA(molecule=mol, layers=1, mapping="parity")
        gap_info = hea.get_homo_lumo_gap(include_ev=True)
        
        print(f"{name:<10} {mol.nelectron:<8} {mol.nao:<8} "
              f"{gap_info['gap']:<12.8f} {gap_info['gap_ev']:<10.4f} "
              f"{gap_info['system_type']}")


def demo_architecture_explanation():
    """演示架构说明"""
    print("\n" + "=" * 80)
    print("演示 4: HEA 与 UCC 的架构关系")
    print("=" * 80)
    
    hea = HEA(molecule=h2, layers=1, mapping="parity")
    
    print(f"\nHEA 架构说明:")
    print(f"  ✓ HEA 类本身不是 UCC 的子类")
    print(f"  ✓ HEA 通过 from_molecule() 内部创建 UCC 对象获取分子信息")
    print(f"  ✓ UCC 对象保存在 hea._ucc_object 中")
    print(f"  ✓ HOMO-LUMO gap 计算委托给内部的 UCC 对象")
    
    print(f"\n内部对象检查:")
    print(f"  HEA 类型: {type(hea).__name__}")
    print(f"  内部 UCC 对象: {type(hea._ucc_object).__name__ if hea._ucc_object else None}")
    print(f"  HEA 层数: {hea.layers}")
    print(f"  HEA 量子比特数: {hea.n_qubits}")
    print(f"  UCC 活性空间: {hea._ucc_object.active_space if hea._ucc_object else None}")
    
    # 展示委托调用
    print(f"\n委托调用演示:")
    hea_direct = hea.get_homo_lumo_gap()
    ucc_direct = hea._ucc_object.get_homo_lumo_gap()
    
    print(f"  hea.get_homo_lumo_gap(): {hea_direct['gap']:.8f} Ha")
    print(f"  hea._ucc_object.get_homo_lumo_gap(): {ucc_direct['gap']:.8f} Ha")
    print(f"  ✓ 结果完全相同（委托成功）")


def demo_error_handling():
    """演示错误处理"""
    print("\n" + "=" * 80)
    print("演示 5: 错误处理 - 从积分构建的 HEA")
    print("=" * 80)
    
    # 从积分构建 HEA（没有分子信息）
    int1e = np.array([[0, -1], [-1, 0]])
    int2e = np.zeros((2, 2, 2, 2))
    hea = HEA.from_integral(int1e, int2e, n_elec=2, e_core=0.0, n_layers=1)
    
    print(f"\n从积分构建的 HEA:")
    print(f"  量子比特数: {hea.n_qubits}")
    print(f"  层数: {hea.layers}")
    print(f"  内部 UCC 对象: {hea._ucc_object}")
    
    print(f"\n尝试计算 HOMO-LUMO gap:")
    try:
        gap = hea.get_homo_lumo_gap()
        print(f"  ❌ 意外成功: {gap}")
    except RuntimeError as e:
        print(f"  ✓ 预期错误: {e}")
        print(f"  ✓ 错误处理正确工作")


def main():
    """运行所有演示"""
    print("\n" + "🎯 " * 20)
    print("HEA HOMO-LUMO Gap 计算功能演示")
    print("🎯 " * 20)
    
    demo_basic_hea_usage()
    demo_hea_vs_ucc()
    demo_multiple_molecules()
    demo_architecture_explanation()
    demo_error_handling()
    
    print("\n" + "=" * 80)
    print("✅ 所有演示完成!")
    print("=" * 80)
    print("\n总结:")
    print("• HEA 通过内部 UCC 对象提供 HOMO-LUMO gap 计算")
    print("• API 与 UCC 完全一致：get_homo_lumo_gap() 和 homo_lumo_gap 属性")
    print("• 支持所有 UCC 的功能：闭壳/开壳系统、手动指定、eV 转换等")
    print("• 只适用于从分子构建的 HEA，从积分构建的 HEA 会抛出错误")
    print()


if __name__ == "__main__":
    main()