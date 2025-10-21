#!/bin/bash
#
# Test all TyxonQ examples
# Usage: bash examples/test_all_examples.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

# Arrays to store results
PASSED_TESTS=()
FAILED_TESTS=()
SKIPPED_TESTS=()

echo "========================================"
echo "TyxonQ Examples Test Suite"
echo "========================================"
echo ""

# Function to run a test
run_test() {
    local test_file=$1
    local test_name=$(basename "$test_file")
    local timeout=${2:-60}  # Default 60s timeout
    
    ((TOTAL++))
    
    echo -n "Testing $test_name ... "
    
    # Run with timeout
    if timeout $timeout python "$test_file" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        ((PASSED++))
        PASSED_TESTS+=("$test_name")
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo -e "${YELLOW}TIMEOUT${NC}"
        else
            echo -e "${RED}FAIL${NC}"
        fi
        ((FAILED++))
        FAILED_TESTS+=("$test_name")
        return 1
    fi
}

# Function to skip a test
skip_test() {
    local test_name=$1
    local reason=$2
    
    ((TOTAL++))
    ((SKIPPED++))
    SKIPPED_TESTS+=("$test_name: $reason")
    echo -e "Skipping $test_name ... ${YELLOW}SKIP${NC} ($reason)"
}

cd "$(dirname "$0")/.."

echo "=== Core Examples ==="
run_test "examples/basic_chain_api.py" 30
run_test "examples/circuit_chain_demo.py" 30
run_test "examples/numeric_backend_switching.py" 60

echo ""
echo "=== VQE Examples ==="
run_test "examples/vqe_simple_hamiltonian.py" 60
run_test "examples/simple_qaoa.py" 60
skip_test "vqe_extra.py" "long running time"
skip_test "vqetfim_benchmark.py" "benchmark, long running"
skip_test "vqeh2o_benchmark.py" "benchmark, long running"

echo ""
echo "=== Simulator Examples ==="
run_test "examples/mps_approximation_benchmark.py" 90
run_test "examples/stabilizer_clifford_entropy.py" 90

echo ""
echo "=== Optimization Examples ==="
run_test "examples/quantum_natural_gradient_optimization.py" 120
skip_test "hybrid_quantum_classical_training.py" "requires torch+MNIST data"
skip_test "vqe_parallel_pmap.py" "requires JAX"

echo ""
echo "=== Compiler Examples ==="
run_test "examples/compiler_lightcone_optimization.py" 60
skip_test "circuit_compiler.py" "manual inspection needed"

echo ""
echo "=== Gradient Examples ==="
run_test "examples/autograd_vs_counts.py" 60
run_test "examples/gradient_benchmark.py" 90
skip_test "parameter_shift.py" "long running"

echo ""
echo "=== Time Evolution Examples ==="
run_test "examples/hamiltonian_time_evolution.py" 60
skip_test "timeevolution_trotter.py" "covered by hamiltonian_time_evolution"

echo ""
echo "=== Hamiltonian Examples ==="
run_test "examples/hamiltonian_building.py" 30
skip_test "hchainhamiltonian.py" "specific chemistry case"

echo ""
echo "=== Sampling Examples ==="
run_test "examples/sample_benchmark.py" 60
skip_test "sample_value_gradient.py" "long running"

echo ""
echo "=== Noise Examples ==="
skip_test "noise_controls_demo.py" "requires noise model setup"
skip_test "readout_mitigation.py" "requires calibration data"
skip_test "vqe_shot_noise.py" "long running"
skip_test "vqe_noisyopt.py" "long running"

echo ""
echo "=== Cloud Examples ==="
skip_test "cloud_*.py" "requires cloud credentials"

echo ""
echo "=== Chemistry Examples ==="
skip_test "demo_hea_homo_lumo_gap.py" "requires PySCF"
skip_test "demo_homo_lumo_gap.py" "requires PySCF"

echo ""
echo "=== Other Examples ==="
skip_test "incremental_twoqubit.py" "advanced feature"
skip_test "jacobian_cal.py" "covered by gradient tests"
skip_test "pulse_demo.py" "pulse-level control"
skip_test "pulse_demo_scan.py" "pulse-level control"
skip_test "jsonio.py" "I/O utility"

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "Total tests:   $TOTAL"
echo -e "${GREEN}Passed:        $PASSED${NC}"
echo -e "${RED}Failed:        $FAILED${NC}"
echo -e "${YELLOW}Skipped:       $SKIPPED${NC}"
echo ""

if [ $PASSED -gt 0 ]; then
    echo -e "${GREEN}Passed tests:${NC}"
    for test in "${PASSED_TESTS[@]}"; do
        echo "  ✓ $test"
    done
    echo ""
fi

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  ✗ $test"
    done
    echo ""
fi

if [ $SKIPPED -gt 0 ]; then
    echo -e "${YELLOW}Skipped tests:${NC}"
    for test in "${SKIPPED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
fi

echo "========================================"

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
