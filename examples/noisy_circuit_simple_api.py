"""Simplified Noise API Demo.

This example demonstrates the new user-friendly noise API using .with_noise()
"""

import tyxonq as tq

print("=" * 60)
print("Simplified Noise API Demo")
print("=" * 60)

# Set backend
tq.set_backend("numpy")

# Create Bell state circuit
c = tq.Circuit(2)
c.h(0).cx(0, 1)

print("\n1. Original API (verbose):")
print("-" * 60)
result_old = c.device(
    provider="simulator",
    device="density_matrix",
    use_noise=True,
    noise={"type": "depolarizing", "p": 0.05}
).run(shots=1024)
print(f"Result: {result_old[0]['result']}")

print("\n2. New with_noise() API (simplified):")
print("-" * 60)
c2 = tq.Circuit(2)
c2.h(0).cx(0, 1)
result_new = c2.with_noise("depolarizing", p=0.05).run(shots=1024)
print(f"Result: {result_new[0]['result']}")

print("\n3. Different noise types:")
print("-" * 60)

# Amplitude damping
c3 = tq.Circuit(2)
c3.h(0).cx(0, 1)
result = c3.with_noise("amplitude_damping", gamma=0.1).run(shots=1024)
print(f"Amplitude damping: {result[0]['result']}")

# Phase damping (using 'l' instead of 'lambda' to avoid Python keyword)
c4 = tq.Circuit(2)
c4.h(0).cx(0, 1)
result = c4.with_noise("phase_damping", l=0.1).run(shots=1024)
print(f"Phase damping: {result[0]['result']}")

# Pauli channel
c5 = tq.Circuit(2)
c5.h(0).cx(0, 1)
result = c5.with_noise("pauli", px=0.01, py=0.01, pz=0.05).run(shots=1024)
print(f"Pauli channel: {result[0]['result']}")

print("\n4. Chaining with other methods:")
print("-" * 60)
c6 = tq.Circuit(2)
c6.h(0).cx(0, 1)
result = c6.with_noise("depolarizing", p=0.05).device(shots=2048).run()
print(f"Chained result: {result[0]['result']}")

print("\n" + "=" * 60)
print("✓ Simplified noise API works perfectly!")
print("=" * 60)
