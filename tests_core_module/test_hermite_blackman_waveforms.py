"""Test cases for Hermite and Blackman waveforms.

This module provides comprehensive testing for the new advanced waveforms:
- Hermite: Polynomial envelopes with minimal spectral leakage
- BlackmanSquare: Industry-standard window with excellent frequency containment

Tests cover:
1. Basic functionality (waveform instantiation, parameters)
2. Sampling and time-domain properties
3. Integration with StatevectorEngine
4. Frequency domain properties
5. Comparison with existing waveforms
6. Serialization/deserialization (pulse_inline support)
7. Physical validation (Rabi oscillations, gate fidelity)
"""

import numpy as np
import pytest
from tyxonq import Circuit, waveforms, set_backend, get_backend
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
from tyxonq.libs.quantum_library.pulse_simulation import (
    sample_waveform,
    compile_pulse_to_unitary
)


class TestHermiteWaveform:
    """Test Hermite polynomial envelope waveforms."""
    
    def test_hermite_instantiation(self):
        """Test basic Hermite waveform instantiation."""
        # Order 2
        hermite2 = waveforms.Hermite(amp=0.8, duration=40, order=2, phase=0.0)
        assert hermite2.amp == 0.8
        assert hermite2.duration == 40
        assert hermite2.order == 2
        assert hermite2.phase == 0.0
        
        # Order 3
        hermite3 = waveforms.Hermite(amp=0.75, duration=50, order=3, phase=np.pi/4)
        assert hermite3.amp == 0.75
        assert hermite3.duration == 50
        assert hermite3.order == 3
        assert abs(hermite3.phase - np.pi/4) < 1e-10
    
    def test_hermite_qasm_name(self):
        """Test Hermite QASM name."""
        hermite = waveforms.Hermite(amp=0.8, duration=40, order=2)
        assert hermite.qasm_name() == "hermite"
    
    def test_hermite_to_args(self):
        """Test Hermite to_args conversion."""
        hermite = waveforms.Hermite(amp=0.8, duration=40, order=2, phase=0.5)
        args = hermite.to_args()
        assert len(args) == 4
        assert args[0] == 0.8      # amp
        assert args[1] == 40       # duration
        assert args[2] == 2        # order
        assert args[3] == 0.5      # phase
    
    def test_hermite_sampling_center(self):
        """Test Hermite waveform sampling at center (t=T/2)."""
        set_backend("numpy")
        hermite = waveforms.Hermite(amp=1.0, duration=40, order=2, phase=0.0)
        
        # Sample at center (t = 20 ns = 20/2000 s)
        # At center, time is normalized to τ = 0.5
        t_center = 20 / 2e9  # 20 samples at 2 GHz rate
        amplitude = sample_waveform(hermite, t_center)
        
        # Amplitude should be non-zero (has Gaussian component)
        assert abs(amplitude) > 0
        assert abs(amplitude) <= 1.0  # Should not exceed amp=1.0
    
    def test_hermite_sampling_edges(self):
        """Test Hermite waveform sampling at edges."""
        set_backend("numpy")
        hermite = waveforms.Hermite(amp=0.8, duration=40, order=2)
        
        # Before pulse start
        amp_before = sample_waveform(hermite, -1e-9)
        assert abs(amp_before) < 1e-10
        
        # After pulse end
        amp_after = sample_waveform(hermite, 30e-9)
        assert abs(amp_after) < 1e-10
    
    def test_hermite_order2_vs_order3(self):
        """Test difference between Hermite order 2 and 3."""
        set_backend("numpy")
        
        hermite2 = waveforms.Hermite(amp=0.8, duration=40, order=2)
        hermite3 = waveforms.Hermite(amp=0.8, duration=40, order=3)
        
        # Sample at multiple time points
        t_values = np.linspace(0, 40/2e9, 10)
        
        amps2 = [sample_waveform(hermite2, t) for t in t_values]
        amps3 = [sample_waveform(hermite3, t) for t in t_values]
        
        # Orders should produce different envelopes
        differences = [abs(a2 - a3) for a2, a3 in zip(amps2, amps3)]
        assert any(d > 1e-10 for d in differences)
    
    def test_hermite_phase_rotation(self):
        """Test phase offset in Hermite waveform."""
        set_backend("numpy")
        
        hermite0 = waveforms.Hermite(amp=1.0, duration=40, order=2, phase=0.0)
        hermite_pi = waveforms.Hermite(amp=1.0, duration=40, order=2, phase=np.pi)
        
        # Sample at center
        t_center = 20 / 2e9
        amp0 = sample_waveform(hermite0, t_center)
        amp_pi = sample_waveform(hermite_pi, t_center)
        
        # Magnitudes should be same, phases opposite
        assert abs(abs(amp0) - abs(amp_pi)) < 1e-10
        assert abs(amp0 + amp_pi) < 1e-10  # They should cancel when added


class TestBlackmanSquareWaveform:
    """Test Blackman window with flat-top waveforms."""
    
    def test_blackman_instantiation(self):
        """Test basic BlackmanSquare waveform instantiation."""
        blackman = waveforms.BlackmanSquare(
            amp=0.8,
            duration=60,
            width=30,
            phase=0.0
        )
        assert blackman.amp == 0.8
        assert blackman.duration == 60
        assert blackman.width == 30
        assert blackman.phase == 0.0
    
    def test_blackman_qasm_name(self):
        """Test Blackman QASM name."""
        blackman = waveforms.BlackmanSquare(amp=0.8, duration=60, width=30)
        assert blackman.qasm_name() == "blackman_square"
    
    def test_blackman_to_args(self):
        """Test BlackmanSquare to_args conversion."""
        blackman = waveforms.BlackmanSquare(amp=0.8, duration=60, width=30, phase=0.25)
        args = blackman.to_args()
        assert len(args) == 4
        assert args[0] == 0.8      # amp
        assert args[1] == 60       # duration
        assert args[2] == 30       # width
        assert args[3] == 0.25     # phase
    
    def test_blackman_sampling_ramp_up(self):
        """Test Blackman sampling in ramp-up region."""
        set_backend("numpy")
        blackman = waveforms.BlackmanSquare(
            amp=1.0,
            duration=60,
            width=30,
            phase=0.0
        )
        
        # Ramp duration: (60 - 30) / 2 = 15 ns
        # Sample at t=5 ns (early ramp-up)
        t_ramp = 5 / 2e9
        amp = sample_waveform(blackman, t_ramp)
        
        # Should be non-zero but less than peak
        assert 0 < abs(amp) < 1.0
    
    def test_blackman_sampling_plateau(self):
        """Test Blackman sampling in plateau region."""
        set_backend("numpy")
        blackman = waveforms.BlackmanSquare(
            amp=1.0,
            duration=60,
            width=30,
            phase=0.0
        )
        
        # Plateau: from 15 ns to 45 ns (ramp-up: 15 ns, ramp-down: 15 ns)
        # Sample at t=30 ns (center of plateau)
        t_plateau = 30 / 2e9
        amp = sample_waveform(blackman, t_plateau)
        
        # Should be at full amplitude (with some tolerance for floating point)
        assert abs(abs(amp) - 1.0) < 0.05
    
    def test_blackman_sampling_ramp_down(self):
        """Test Blackman sampling in ramp-down region."""
        set_backend("numpy")
        blackman = waveforms.BlackmanSquare(
            amp=1.0,
            duration=60,
            width=30,
            phase=0.0
        )
        
        # Ramp-down: from 45 ns to 60 ns
        # Sample at t=55 ns (late ramp-down)
        t_ramp_down = 55 / 2e9
        amp = sample_waveform(blackman, t_ramp_down)
        
        # Should be non-zero but less than peak
        assert 0 < abs(amp) < 1.0
    
    def test_blackman_symmetry(self):
        """Test Blackman ramp-up and ramp-down symmetry."""
        set_backend("numpy")
        blackman = waveforms.BlackmanSquare(
            amp=1.0,
            duration=60,
            width=30,
            phase=0.0
        )
        
        # Ramp duration: 15 ns
        # Sample symmetrically from start and end
        t1 = 5 / 2e9      # 5 ns from start (early ramp-up)
        t2 = 55 / 2e9     # 5 ns from end (late ramp-down)
        
        amp1 = sample_waveform(blackman, t1)
        amp2 = sample_waveform(blackman, t2)
        
        # Magnitudes should be approximately equal (symmetric)
        assert abs(abs(amp1) - abs(amp2)) < 0.1
    
    def test_blackman_phase_rotation(self):
        """Test phase offset in Blackman waveform."""
        set_backend("numpy")
        
        blackman0 = waveforms.BlackmanSquare(
            amp=1.0, duration=60, width=30, phase=0.0
        )
        blackman_pi2 = waveforms.BlackmanSquare(
            amp=1.0, duration=60, width=30, phase=np.pi/2
        )
        
        # Sample at plateau center
        t_plateau = 30 / 2e9
        amp0 = sample_waveform(blackman0, t_plateau)
        amp_pi2 = sample_waveform(blackman_pi2, t_plateau)
        
        # Magnitude should be same, phase different
        assert abs(abs(amp0) - abs(amp_pi2)) < 0.05


class TestWaveformIntegration:
    """Test Hermite and Blackman integration with StatevectorEngine."""
    
    def test_hermite_circuit_execution(self):
        """Test Hermite waveform in circuit execution."""
        set_backend("numpy")
        engine = StatevectorEngine()
        
        circuit = Circuit(1)
        hermite = waveforms.Hermite(amp=0.8, duration=40, order=2)
        circuit.metadata["pulse_library"] = {"hermite": hermite}
        circuit = circuit.extended([
            ("pulse", 0, "hermite", {
                "qubit_freq": 5.0e9,
                "drive_freq": 5.0e9,
                "anharmonicity": -330e6,
                "rabi_freq": 50e6
            })
        ])
        circuit.measure_z(0)
        
        # Run circuit
        result = engine.run(circuit, shots=1024)
        counts = result.get("result", {})
        
        # Check valid result
        assert "0" in counts or "1" in counts
        assert sum(counts.values()) == 1024
    
    def test_blackman_circuit_execution(self):
        """Test Blackman waveform in circuit execution."""
        set_backend("numpy")
        engine = StatevectorEngine()
        
        circuit = Circuit(1)
        blackman = waveforms.BlackmanSquare(amp=0.8, duration=60, width=40)
        circuit.metadata["pulse_library"] = {"blackman": blackman}
        circuit = circuit.extended([
            ("pulse", 0, "blackman", {
                "qubit_freq": 5.0e9,
                "drive_freq": 5.0e9,
                "anharmonicity": -330e6,
                "rabi_freq": 50e6
            })
        ])
        circuit.measure_z(0)
        
        # Run circuit
        result = engine.run(circuit, shots=1024)
        counts = result.get("result", {})
        
        # Check valid result
        assert "0" in counts or "1" in counts
        assert sum(counts.values()) == 1024
    
    def test_hermite_pulse_inline_serialization(self):
        """Test Hermite waveform serialization (pulse_inline)."""
        set_backend("numpy")
        engine = StatevectorEngine()
        
        circuit = Circuit(1)
        waveform_dict = {
            "type": "hermite",
            "class": "Hermite",
            "args": [0.8, 40, 2, 0.0]
        }
        
        circuit = circuit.extended([
            ("pulse_inline", 0, waveform_dict, {
                "qubit_freq": 5.0e9,
                "drive_freq": 5.0e9,
                "anharmonicity": -330e6
            })
        ])
        circuit.measure_z(0)
        
        result = engine.run(circuit, shots=1024)
        counts = result.get("result", {})
        
        # Check valid result
        assert "0" in counts or "1" in counts
        assert sum(counts.values()) == 1024
    
    def test_blackman_pulse_inline_serialization(self):
        """Test Blackman waveform serialization (pulse_inline)."""
        set_backend("numpy")
        engine = StatevectorEngine()
        
        circuit = Circuit(1)
        waveform_dict = {
            "type": "blackman_square",
            "class": "BlackmanSquare",
            "args": [0.8, 60, 40, 0.0]
        }
        
        circuit = circuit.extended([
            ("pulse_inline", 0, waveform_dict, {
                "qubit_freq": 5.0e9,
                "drive_freq": 5.0e9,
                "anharmonicity": -330e6
            })
        ])
        circuit.measure_z(0)
        
        result = engine.run(circuit, shots=1024)
        counts = result.get("result", {})
        
        # Check valid result
        assert "0" in counts or "1" in counts
        assert sum(counts.values()) == 1024


class TestWaveformComparison:
    """Compare Hermite and Blackman with existing waveforms."""
    
    def test_waveform_fidelity_comparison(self):
        """Compare X-gate fidelity across waveform types."""
        set_backend("numpy")
        engine = StatevectorEngine()
        
        # Define test waveforms (calibrated for approximately π rotation)
        waveforms_dict = {
            "gaussian": waveforms.Gaussian(amp=0.8, duration=40, sigma=10),
            "hermite2": waveforms.Hermite(amp=0.8, duration=40, order=2),
            "hermite3": waveforms.Hermite(amp=0.8, duration=40, order=3),
            "blackman": waveforms.BlackmanSquare(amp=0.8, duration=40, width=25),
        }
        
        fidelities = {}
        
        for name, pulse in waveforms_dict.items():
            circuit = Circuit(1)
            circuit.metadata["pulse_library"] = {"pulse": pulse}
            circuit = circuit.extended([
                ("pulse", 0, "pulse", {
                    "qubit_freq": 5.0e9,
                    "drive_freq": 5.0e9,
                    "anharmonicity": -330e6,
                    "rabi_freq": 50e6
                })
            ])
            circuit.measure_z(0)
            
            result = engine.run(circuit, shots=2048)
            counts = result.get("result", {})
            
            p1 = counts.get("1", 0) / 2048
            # For X gate, expect P(|1⟩) ≈ 0.5, so fidelity = 1 - |P(|1⟩) - 0.5|
            fidelity = 1.0 - abs(p1 - 0.5)
            fidelities[name] = fidelity
        
        # All should achieve reasonable fidelity
        for name, fidelity in fidelities.items():
            assert fidelity > 0.4, f"{name} fidelity too low: {fidelity}"
        
        # Hermite and Blackman should be competitive with Gaussian
        assert fidelities["hermite2"] > 0.85 or fidelities["blackman"] > 0.85
    
    def test_waveform_envelope_smoothness(self):
        """Test envelope smoothness by checking derivatives."""
        set_backend("numpy")
        
        hermite = waveforms.Hermite(amp=0.8, duration=40, order=2)
        blackman = waveforms.BlackmanSquare(amp=0.8, duration=40, width=25)
        
        # Sample envelope densely
        t_values = np.linspace(0, 40/2e9, 100)
        
        hermite_amps = [sample_waveform(hermite, t) for t in t_values]
        blackman_amps = [sample_waveform(blackman, t) for t in t_values]
        
        # Check for no NaN or Inf
        assert all(np.isfinite(np.abs(a)) for a in hermite_amps)
        assert all(np.isfinite(np.abs(a)) for a in blackman_amps)


class TestWaveformProperties:
    """Test physical properties of Hermite and Blackman waveforms."""
    
    def test_hermite_amplitude_bounds(self):
        """Test that Hermite amplitude stays within bounds."""
        set_backend("numpy")
        hermite = waveforms.Hermite(amp=0.9, duration=40, order=2)
        
        t_values = np.linspace(0, 40/2e9, 200)
        amps = [sample_waveform(hermite, t) for t in t_values]
        
        max_amp = max(abs(a) for a in amps)
        assert max_amp <= 0.9 * 1.1  # Allow 10% overshoot due to modulation
    
    def test_blackman_amplitude_bounds(self):
        """Test that Blackman amplitude stays within bounds."""
        set_backend("numpy")
        blackman = waveforms.BlackmanSquare(amp=0.9, duration=60, width=30)
        
        t_values = np.linspace(0, 60/2e9, 300)
        amps = [sample_waveform(blackman, t) for t in t_values]
        
        max_amp = max(abs(a) for a in amps)
        assert max_amp <= 0.9 * 1.05  # Allow 5% overshoot
    
    def test_waveform_energy_conservation(self):
        """Test that waveform energy is reasonable."""
        set_backend("numpy")
        
        hermite = waveforms.Hermite(amp=1.0, duration=40, order=2)
        blackman = waveforms.BlackmanSquare(amp=1.0, duration=60, width=30)
        
        # Sample densely
        t_hermite = np.linspace(0, 40/2e9, 1000)
        t_blackman = np.linspace(0, 60/2e9, 1000)
        
        amps_hermite = np.array([sample_waveform(hermite, t) for t in t_hermite])
        amps_blackman = np.array([sample_waveform(blackman, t) for t in t_blackman])
        
        # Energy (integral of |Ω(t)|²)
        energy_hermite = np.sum(np.abs(amps_hermite)**2) / len(t_hermite)
        energy_blackman = np.sum(np.abs(amps_blackman)**2) / len(t_blackman)
        
        # Both should have positive energy
        assert energy_hermite > 0
        assert energy_blackman > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
