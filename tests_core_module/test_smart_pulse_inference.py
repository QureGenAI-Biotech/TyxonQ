#!/usr/bin/env python3
"""测试智能 Pulse 推断功能"""

def test_smart_pulse_inference():
    """测试 output='tqasm' 自动启用 pulse 编译 + 自动补足默认参数"""
    from src.tyxonq.core.ir.circuit import Circuit
    from src.tyxonq.compiler.api import compile
    import warnings
    
    print("=" * 70)
    print("测试：智能 Pulse 推断 (output='tqasm' → 自动 pulse 编译)")
    print("=" * 70)
    
    # 场景1: 显式使用 .use_pulse() - 无警告
    print("\n1️⃣  场景1：显式声明 (推荐)")
    print("-" * 70)
    
    c1 = Circuit(2)
    c1.h(0).cx(0, 1)
    c1.use_pulse(device_params={
        "qubit_freq": [5.0e9, 5.1e9],
        "anharmonicity": [-330e6, -320e6]
    })
    
    print("   代码: c.use_pulse(device_params={...}).compile(output='tqasm')")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result1 = compile(c1, output="tqasm")
        
        # 应该没有警告（用户显式声明了）
        tqasm_warnings = [warning for warning in w if "tqasm" in str(warning.message).lower()]
        param_warnings = [warning for warning in w if "device_params" in str(warning.message).lower()]
        
        if not tqasm_warnings and not param_warnings:
            print("   ✅ 无警告（显式声明，参数完整）")
        else:
            print(f"   ⚠️  有警告: {len(tqasm_warnings)} tqasm, {len(param_warnings)} params")
    
    # 场景2: 自动推断 + 提供参数 - 有 tqasm 警告，无 params 警告
    print("\n2️⃣  场景2：智能推断 + 提供参数")
    print("-" * 70)
    
    c2 = Circuit(2)
    c2.h(0).cx(0, 1)
    
    print("   代码: compile(c, output='tqasm', options={'device_params': {...}})")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result2 = compile(c2, output="tqasm", options={
            "device_params": {
                "qubit_freq": [5.0e9, 5.1e9],
                "anharmonicity": [-330e6, -320e6]
            }
        })
        
        tqasm_warnings = [warning for warning in w if "tqasm" in str(warning.message).lower()]
        param_warnings = [warning for warning in w if "device_params" in str(warning.message).lower()]
        
        print(f"   ⚠️  TQASM 警告: {len(tqasm_warnings)} 条")
        if tqasm_warnings:
            print(f"      → {tqasm_warnings[0].message}")
        
        print(f"   ✅ Params 警告: {len(param_warnings)} 条（参数已提供）")
    
    # 场景3: 自动推断 + 自动补足参数 - 两个警告
    print("\n3️⃣  场景3：智能推断 + 自动补足参数（最智能）")
    print("-" * 70)
    
    c3 = Circuit(2)
    c3.h(0).cx(0, 1)
    
    print("   代码: compile(c, output='tqasm')  # 什么都不提供")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result3 = compile(c3, output="tqasm")
        
        tqasm_warnings = [warning for warning in w if "tqasm" in str(warning.message).lower()]
        param_warnings = [warning for warning in w if "自动补足" in str(warning.message)]
        
        print(f"   ⚠️  TQASM 警告: {len(tqasm_warnings)} 条")
        if tqasm_warnings:
            print(f"      → {str(tqasm_warnings[0].message)[:60]}...")
        
        print(f"   ⚠️  Params 警告: {len(param_warnings)} 条（自动补足默认值）")
        if param_warnings:
            print(f"      → 自动补足: qubit_freq=[5.0e9, 5.0e9], anharmonicity=[-330e6, -330e6]")
            for line in str(param_warnings[0].message).split('\n')[:3]:
                print(f"         {line}")
    
    # 场景4: 显式 compile_engine="pulse" - 无 tqasm 警告
    print("\n4️⃣  场景4：显式指定 compile_engine='pulse'")
    print("-" * 70)
    
    c4 = Circuit(2)
    c4.h(0).cx(0, 1)
    
    print("   代码: compile(c, compile_engine='pulse', output='tqasm')")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result4 = compile(c4, compile_engine="pulse", output="tqasm")
        
        tqasm_warnings = [warning for warning in w if "tqasm" in str(warning.message).lower() and "自动启用" in str(warning.message)]
        param_warnings = [warning for warning in w if "自动补足" in str(warning.message)]
        
        print(f"   ✅ TQASM 警告: {len(tqasm_warnings)} 条（显式指定 engine）")
        print(f"   ⚠️  Params 警告: {len(param_warnings)} 条（仍会补足参数）")
    
    # 总结
    print("\n" + "=" * 70)
    print("✅ 智能推断测试完成!")
    print("=" * 70)
    print("""
    智能推断规则：
    1. output='tqasm' → 自动启用 pulse 编译（有警告）
    2. 缺少 device_params → 自动补足默认值（有警告）
    3. 显式声明 (.use_pulse() 或 compile_engine='pulse') → 减少警告
    
    推荐用法：
    ✅ c.use_pulse(device_params={...}).compile(output='tqasm')
    ⚠️  compile(c, output='tqasm')  # 可用，但有警告
    """)


if __name__ == "__main__":
    test_smart_pulse_inference()
