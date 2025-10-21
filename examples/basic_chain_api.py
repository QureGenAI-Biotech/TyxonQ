"""
Basic Chain API and Global Configuration Demo

This example demonstrates TyxonQ's chainable API design and global configuration system.
It showcases the core programming paradigm: circuit.compile().device().postprocessing().run()

Key Features Demonstrated:
1. Chainable API pattern
2. Global device/postprocessing defaults  
3. Automatic configuration completion
4. Simple Bell state creation and execution
5. Integration with cloud API

Migrated from: examples-ng/simple_demo_1.py
Original reference: TensorCircuit examples
"""

import tyxonq as tq
import getpass


def demo_basic_chain():
    """Demonstrate basic chainable API usage"""
    print("=== Demo 1: Basic Chain API ===")
    
    # Create a simple Bell state circuit
    c = tq.Circuit(2)
    c.h(0)
    c.cnot(0, 1)
    c.rx(1, theta=0.2)
    
    # Execute with explicit configuration
    result = (
        c.compile()
        .device(provider="simulator", device="statevector", shots=100)
        .postprocessing(method=None)
        .run()
    )
    
    print("Explicit chain result:", result)
    return result


def demo_global_defaults():
    """Demonstrate global configuration system"""
    print("\n=== Demo 2: Global Defaults ===")
    
    # Set global defaults (applies to all subsequent circuits)
    tq.device(provider="simulator", device="statevector", shots=200)
    
    # Create circuit and run with global defaults
    c = tq.Circuit(2)
    c.h(0)
    c.cnot(0, 1)
    
    # Simple run() automatically uses global defaults
    result = c.run()
    
    print("Global defaults result:", result)
    print("Current device defaults:", tq.get_device_defaults())
    
    return result


def demo_auto_completion():
    """Demonstrate automatic configuration completion"""
    print("\n=== Demo 3: Auto-Completion ===")
    
    # Only specify shots, other options auto-complete
    c = tq.Circuit(2).h(0).cnot(0, 1)
    
    result = c.device(shots=150).run()
    print("Auto-completed result:", result)
    
    return result


def demo_cloud_integration():
    """Demonstrate cloud API integration (optional, requires credentials)"""
    print("\n=== Demo 4: Cloud Integration (Optional) ===")
    
    try:
        # Uncomment and provide your API key to test cloud submission
        # api_key = getpass.getpass("Input your TyxonQ API_KEY: ")
        # tq.set_token(api_key, provider="tyxonq", device="homebrew_s2")
        
        # tq.device(provider="tyxonq", device="homebrew_s2", shots=100)
        
        # c = tq.Circuit(2)
        # c.h(0)
        # c.cnot(0, 1)
        # c.rx(1, theta=0.2)
        
        # result = c.run()
        # print("Cloud execution result:", result)
        
        print("Cloud demo skipped (requires API key)")
        print("Uncomment code and provide credentials to test cloud execution")
        
    except Exception as e:
        print(f"Cloud demo error (expected without credentials): {e}")


def demo_multi_backend():
    """Demonstrate multiple numeric backend support"""
    print("\n=== Demo 5: Multi-Backend Support ===")
    
    # Test with NumPy backend
    tq.set_backend("numpy")
    c1 = tq.Circuit(2).h(0).cnot(0, 1)
    result_np = c1.run()
    print("NumPy backend result:", result_np)
    
    # Test with PyTorch backend (if available)
    try:
        tq.set_backend("pytorch")
        c2 = tq.Circuit(2).h(0).cnot(0, 1)
        result_torch = c2.run()
        print("PyTorch backend result:", result_torch)
    except Exception as e:
        print(f"PyTorch backend not available: {e}")
        tq.set_backend("numpy")  # Fallback
    
    return result_np


if __name__ == "__main__":
    print("=" * 60)
    print("TyxonQ Basic Chain API Demo")
    print("Demonstrates the core chainable programming paradigm")
    print("=" * 60)
    
    # Run all demos
    demo_basic_chain()
    demo_global_defaults()
    demo_auto_completion()
    demo_cloud_integration()
    demo_multi_backend()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    
    print("\nKey Takeaways:")
    print("1. Chain API: compile().device().postprocessing().run()")
    print("2. Global defaults simplify repeated configurations")
    print("3. Auto-completion fills missing options intelligently")
    print("4. Seamless switching between simulator and cloud hardware")
    print("5. Flexible numeric backend support (NumPy/PyTorch/CuPyNumeric)")
