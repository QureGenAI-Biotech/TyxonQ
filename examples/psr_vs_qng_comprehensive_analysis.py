"""
PSR vs QNG 综合分析
=====================

完整对比 PSR（参数移位法）和 QNG（量子自然梯度）在 VQE 优化中的表现。

包含：
1. PSR Baseline：使用 SciPy L-BFGS-B + PSR 梯度的标准优化
2. QNG 优化：使用 4-point stencil QFI + QNG 梯度的优化
3. 详细的迭代日志和性能对比

核心原理：
  PSR 梯度：∂E/∂θ_i ≈ [E(θ_i + π/2) - E(θ_i - π/2)] / 2
  QFI 矩阵：QFI[i,i] ≈ |∂²E/∂θ_i²| / 4（4-point stencil）
  QNG 梯度：∇_nat = QFI^{-1} @ ∇_PSR
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple
from scipy.optimize import minimize
from scipy.linalg import solve
import warnings

from tyxonq.applications.chem import HEA, UCCSD
from tyxonq.applications.chem.molecule import h2


# ==============================================================================
# PSR BASELINE OPTIMIZER
# ==============================================================================

class PSROptimizer:
    """
    标准 PSR 优化：使用 SciPy L-BFGS-B + PSR 梯度。
    """
    
    def __init__(self, algorithm, verbose: bool = True):
        self.algorithm = algorithm
        self.verbose = verbose
        self.history: List[Dict] = []
        self.n_evaluations = 0
    
    def optimize(
        self,
        runtime: str = "numeric",
        shots: int = 1024,
        max_iterations: int = 100,
    ) -> Dict[str, Any]:
        """
        PSR 梯度优化。
        """
        
        params_init = self.algorithm.init_guess.copy()
        self.n_evaluations = 0
        self.history = []
        
        def objective_and_grad(x):
            """计算能量和 PSR 梯度"""
            if runtime == "numeric":
                energy, grad = self.algorithm.energy_and_grad(x, runtime="numeric")
            else:
                energy, grad = self.algorithm.energy_and_grad(x, runtime="device", shots=shots)
            
            energy = float(energy)
            grad = np.asarray(grad, dtype=np.float64)
            
            self.n_evaluations += 1
            circuit_evals = 1 + 2 * len(x)
            total_evals = self.n_evaluations * circuit_evals
            grad_norm = np.linalg.norm(grad)
            
            self.history.append({
                "n_evals": self.n_evaluations,
                "energy": energy,
                "grad_norm": grad_norm,
                "circuit_evals": total_evals,
                "shots": total_evals * shots if runtime == "device" else 0,
            })
            
            if self.verbose:
                shots_str = f"{total_evals * shots}" if runtime == "device" else "N/A"
                print(f"{self.n_evaluations:5d} {energy:16.10f} {grad_norm:14.8f} {total_evals:8d} {shots_str:>12}")
            
            return energy, grad
        
        def objective_func(x):
            e, _ = objective_and_grad(x)
            return e
        
        def gradient_func(x):
            _, g = objective_and_grad(x)
            return g
        
        t_opt_start = time.time()
        
        result = minimize(
            objective_func,
            params_init,
            jac=gradient_func,
            method="L-BFGS-B",
            options={"maxiter": max_iterations, "disp": False, "ftol": 1e-9},
        )
        
        t_opt_end = time.time()
        
        final_energy = result.fun
        n_iters = result.nit
        total_circuit_evals = self.n_evaluations * (1 + 2 * len(params_init))
        total_shots = total_circuit_evals * shots if runtime == "device" else 0
        
        self.algorithm.opt_res = {
            "x": np.asarray(result.x, dtype=np.float64),
            "fun": float(result.fun),
            "nit": int(result.nit),
        }
        self.algorithm.params = self.algorithm.opt_res["x"].copy()
        
        return {
            "final_energy": final_energy,
            "iterations": n_iters,
            "n_evaluations": self.n_evaluations,
            "total_circuit_evals": total_circuit_evals,
            "total_shots": total_shots,
            "time": t_opt_end - t_opt_start,
            "history": self.history,
        }


# ==============================================================================
# QFI ESTIMATOR (4-Point Stencil)
# ==============================================================================

class DeviceQFIEstimator:
    """
    使用 4-point stencil 在设备上计算 QFI。
    """
    
    def __init__(self, algorithm, verbose: bool = True, eps: float = np.pi / 8):
        self.algorithm = algorithm
        self.verbose = verbose
        self.eps = float(eps)
        self.n_measurements = 0
    
    def estimate(
        self,
        params: np.ndarray,
        shots: int = 1024,
        regularization: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        从设备上用 4-point stencil 计算 QFI。
        """
        
        params = np.asarray(params, dtype=np.float64)
        n_params = len(params)
        
        print(f"\n[QFI Estimation via 4-Point Stencil on Device]")
        print(f"  Using device runtime (shots={shots})")
        print(f"  Shift distance: {self.eps:.6f} rad ({np.degrees(self.eps):.2f}°)")
        print(f"  Computing {n_params} diagonal elements...")
        
        qfi = np.zeros((n_params, n_params), dtype=np.float64)
        shifts = np.array([-2*self.eps, -self.eps, self.eps, 2*self.eps])
        
        # 先计算当前点的梯度（用于 fallback）
        if self.verbose:
            print(f"  [Baseline] Computing PSR gradient at current params...", end=" ")
        
        _, grad_current = self.algorithm.energy_and_grad(
            params,
            runtime="device",
            shots=shots
        )
        grad_current = np.asarray(grad_current, dtype=np.float64)
        self.n_measurements += 1 + 2 * n_params
        
        if self.verbose:
            print("done")
        
        # 计算每个参数的 QFI 对角元
        for i in range(n_params):
            if self.verbose:
                print(f"  [{i+1}/{n_params}] Computing QFI[{i},{i}]...", end=" ")
            
            energies = []
            for shift in shifts:
                params_shifted = params.copy()
                params_shifted[i] += shift
                
                try:
                    e_shifted, _ = self.algorithm.energy_and_grad(
                        params_shifted,
                        runtime="device",
                        shots=shots
                    )
                    energies.append(float(e_shifted))
                    self.n_measurements += 1 + 2 * n_params
                except Exception:
                    energies.append(0.0)
            
            # 4-point stencil: 2nd derivative
            if len(energies) == 4:
                e_m2, e_m1, e_p1, e_p2 = energies
                d2e = (e_m2 - 2*e_m1 + 2*e_p1 - e_p2) / (2 * self.eps**2)
                d2e_abs = abs(d2e) / 4.0
                
                # Adaptive fallback
                grad_mag = abs(grad_current[i])
                if d2e_abs < regularization or grad_mag > d2e_abs * 10:
                    qfi[i, i] = max(grad_mag, regularization)
                    fallback_str = " (grad fallback)"
                else:
                    qfi[i, i] = max(d2e_abs, regularization)
                    fallback_str = ""
            else:
                qfi[i, i] = max(abs(grad_current[i]), regularization)
                fallback_str = " (fallback)"
            
            if self.verbose:
                print(f"QFI={qfi[i,i]:.6e}{fallback_str}")
        
        # 正则化以确保正定性
        qfi = qfi + regularization * np.eye(n_params)
        
        if self.verbose:
            cond = np.linalg.cond(qfi)
            print(f"\n[QFI] Condition number: {cond:.2e}")
        
        total_shots = self.n_measurements * shots
        
        return {
            "qfi": qfi,
            "cond_number": np.linalg.cond(qfi),
            "shots": total_shots,
        }


# ==============================================================================
# QNG OPTIMIZER
# ==============================================================================

class QNGOptimizer:
    """
    真机友好的 QNG 优化：device PSR 梯度 + device QFI + QNG 自然梯度。
    """
    
    def __init__(self, algorithm, verbose: bool = True):
        self.algorithm = algorithm
        self.verbose = verbose
        self.history: List[Dict] = []
        self.n_evaluations = 0
    
    def optimize(
        self,
        shots: int = 1024,
        max_iterations: int = 100,
        qfi_shots: int = 1024,
    ) -> Dict[str, Any]:
        """
        QNG 优化。
        """
        
        params_init = self.algorithm.init_guess.copy()
        n_params = len(params_init)
        
        # ===== Phase 1: QFI 预计算 =====
        print(f"\n[PHASE 1] QFI Precomputation (One-time)")
        print(f"{'-'*100}")
        
        qfi_estimator = DeviceQFIEstimator(self.algorithm, verbose=self.verbose)
        qfi_result = qfi_estimator.estimate(
            params_init,
            shots=qfi_shots,
            regularization=1e-4,
        )
        
        qfi = qfi_result["qfi"]
        shots_qfi = qfi_result["shots"]
        total_shots = shots_qfi
        
        print(f"\n[QFI] Total shots: {shots_qfi}")
        
        # ===== Phase 2: 优化循环 =====
        print(f"\n[PHASE 2] Optimization Loop (L-BFGS-B + QNG)")
        print(f"{'-'*100}")
        print(f"{'Iter':>5} {'Energy':>16} {'|∇PSR|':>14} {'|∇QNG|':>14} {'Evals':>8}")
        print(f"{'-'*100}")
        
        self.n_evaluations = 0
        self.history = []
        
        def objective_and_grad(x):
            """计算能量和 QNG 梯度"""
            energy, grad_psr = self.algorithm.energy_and_grad(
                x,
                runtime="device",
                shots=shots
            )
            
            energy = float(energy)
            grad_psr = np.asarray(grad_psr, dtype=np.float64)
            
            # 应用 QFI 预处理
            try:
                grad_qng = solve(qfi, grad_psr)
            except np.linalg.LinAlgError:
                warnings.warn(f"QFI inversion failed, using PSR gradient")
                grad_qng = grad_psr
            
            self.n_evaluations += 1
            circuit_evals = 1 + 2 * len(x)
            total_evals = self.n_evaluations * circuit_evals
            
            grad_psr_norm = np.linalg.norm(grad_psr)
            grad_qng_norm = np.linalg.norm(grad_qng)
            
            self.history.append({
                "energy": energy,
                "grad_psr_norm": grad_psr_norm,
                "grad_qng_norm": grad_qng_norm,
                "circuit_evals": total_evals,
            })
            
            if self.verbose:
                print(f"{self.n_evaluations:5d} {energy:16.10f} {grad_psr_norm:14.8f} "
                      f"{grad_qng_norm:14.8f} {total_evals:8d}")
            
            # 返回 (energy, grad_qng)
            return energy, grad_qng
        
        def objective_func(x):
            e, _ = objective_and_grad(x)
            return e
        
        def gradient_func(x):
            _, g = objective_and_grad(x)
            return g
        
        t_opt_start = time.time()
        
        result = minimize(
            objective_func,
            params_init,
            jac=gradient_func,
            method="L-BFGS-B",
            options={"maxiter": max_iterations, "disp": False, "ftol": 1e-9},
        )
        
        t_opt_end = time.time()
        
        print(f"{'-'*100}\n")
        
        final_energy = result.fun
        n_iters = result.nit
        total_circuit_evals_opt = self.n_evaluations * (1 + 2 * n_params)
        total_shots_opt = total_circuit_evals_opt * shots
        
        total_shots += total_shots_opt
        
        self.algorithm.opt_res = {
            "x": np.asarray(result.x, dtype=np.float64),
            "fun": float(result.fun),
            "nit": int(result.nit),
        }
        self.algorithm.params = self.algorithm.opt_res["x"].copy()
        
        print(f"[Optimization Summary]")
        print(f"  Final Energy: {final_energy:.10f} Ha")
        print(f"  Iterations: {n_iters}")
        print(f"  Function Evals: {self.n_evaluations}")
        print(f"  Time: {t_opt_end - t_opt_start:.4f} s")
        print(f"\n[Shot Budget]")
        print(f"  QFI Precomputation: {shots_qfi} shots")
        print(f"  Optimization Loop: {total_shots_opt} shots")
        print(f"  Total: {total_shots} shots")
        
        return {
            "final_energy": final_energy,
            "iterations": n_iters,
            "n_evaluations": self.n_evaluations,
            "shots_qfi": shots_qfi,
            "shots_optimization": total_shots_opt,
            "total_shots": total_shots,
            "time": t_opt_end - t_opt_start,
            "history": self.history,
        }


# ==============================================================================
# MAIN: COMPREHENSIVE ANALYSIS
# ==============================================================================

def main():
    """
    运行 PSR vs QNG 的综合对比分析。
    """
    
    print("\n" + "="*100)
    print("PSR vs QNG Comprehensive Analysis for H2 Molecule")
    print("="*100)
    
    fci_energy = -1.1372744055
    results = {}
    
    # ==========================================
    # 1. HEA + PSR (Numeric)
    # ==========================================
    print("\n\n" + "█"*100)
    print("1. HEA + PSR (Numeric Runtime)")
    print("█"*100)
    
    uccsd_ref = UCCSD(h2, run_fci=True)
    hea_numeric = HEA.ry(
        uccsd_ref.int1e,
        uccsd_ref.int2e,
        uccsd_ref.n_elec,
        uccsd_ref.e_core,
        n_layers=2,
        runtime="numeric"
    )
    hea_numeric.numeric_engine = "statevector"
    hea_numeric.grad = "param-shift"
    
    optimizer1 = PSROptimizer(hea_numeric, verbose=True)
    
    print(f"\n{'Iter':>5} {'Energy':>16} {'|∇E|':>14} {'Evals':>8} {'Shots':>12}")
    print(f"{'-'*100}")
    result1 = optimizer1.optimize(runtime="numeric", shots=0, max_iterations=100)
    results["HEA_PSR_Numeric"] = result1
    
    error1 = abs(result1["final_energy"] - fci_energy)
    print(f"\nFinal Energy: {result1['final_energy']:.10f} Ha")
    print(f"Error (vs FCI): {error1:.10f} Ha ({error1*1000:.3f} mHa)")
    
    # ==========================================
    # 2. HEA + PSR (Device)
    # ==========================================
    print("\n\n" + "█"*100)
    print("2. HEA + PSR (Device Runtime, Shots=0)")
    print("█"*100)
    
    hea_device = HEA.ry(
        uccsd_ref.int1e,
        uccsd_ref.int2e,
        uccsd_ref.n_elec,
        uccsd_ref.e_core,
        n_layers=2,
        runtime="device"
    )
    hea_device.grad = "param-shift"
    
    optimizer2 = PSROptimizer(hea_device, verbose=True)
    
    print(f"\n{'Iter':>5} {'Energy':>16} {'|∇E|':>14} {'Evals':>8} {'Shots':>12}")
    print(f"{'-'*100}")
    result2 = optimizer2.optimize(runtime="device", shots=0, max_iterations=100)
    results["HEA_PSR_Device"] = result2
    
    error2 = abs(result2["final_energy"] - fci_energy)
    print(f"\nFinal Energy: {result2['final_energy']:.10f} Ha")
    print(f"Error (vs FCI): {error2:.10f} Ha ({error2*1000:.3f} mHa)")
    
    # ==========================================
    # 3. HEA + QNG (Device)
    # ==========================================
    print("\n\n" + "█"*100)
    print("3. HEA + QNG (Device Runtime with QFI)")
    print("█"*100)
    
    hea_qng = HEA.ry(
        uccsd_ref.int1e,
        uccsd_ref.int2e,
        uccsd_ref.n_elec,
        uccsd_ref.e_core,
        n_layers=2,
        runtime="device"
    )
    hea_qng.grad = "param-shift"
    
    print(f"\n[QFI Precomputation]")
    optimizer3 = QNGOptimizer(hea_qng, verbose=True)
    result3 = optimizer3.optimize(shots=1024, max_iterations=30, qfi_shots=1024)
    results["HEA_QNG_Device"] = result3
    
    error3 = abs(result3["final_energy"] - fci_energy)
    print(f"\nFinal Energy: {result3['final_energy']:.10f} Ha")
    print(f"Error (vs FCI): {error3:.10f} Ha ({error3*1000:.3f} mHa)")
    
    # ==========================================
    # 4. UCCSD + PSR (Numeric)
    # ==========================================
    print("\n\n" + "█"*100)
    print("4. UCCSD + PSR (Numeric Runtime)")
    print("█"*100)
    
    uccsd_numeric = UCCSD(h2, runtime="numeric")
    
    optimizer4 = PSROptimizer(uccsd_numeric, verbose=True)
    
    print(f"\n{'Iter':>5} {'Energy':>16} {'|∇E|':>14} {'Evals':>8} {'Shots':>12}")
    print(f"{'-'*100}")
    result4 = optimizer4.optimize(runtime="numeric", shots=0, max_iterations=100)
    results["UCCSD_PSR_Numeric"] = result4
    
    error4 = abs(result4["final_energy"] - fci_energy)
    print(f"\nFinal Energy: {result4['final_energy']:.10f} Ha")
    print(f"Error (vs FCI): {error4:.10f} Ha ({error4*1000:.3f} mHa)")
    
    # ==========================================
    # 5. UCCSD + PSR (Device)
    # ==========================================
    print("\n\n" + "█"*100)
    print("5. UCCSD + PSR (Device Runtime, Shots=0)")
    print("█"*100)
    
    uccsd_device = UCCSD(h2, runtime="device")
    uccsd_device.grad = "param-shift"
    
    optimizer5 = PSROptimizer(uccsd_device, verbose=True)
    
    print(f"\n{'Iter':>5} {'Energy':>16} {'|∇E|':>14} {'Evals':>8} {'Shots':>12}")
    print(f"{'-'*100}")
    result5 = optimizer5.optimize(runtime="device", shots=0, max_iterations=100)
    results["UCCSD_PSR_Device"] = result5
    
    error5 = abs(result5["final_energy"] - fci_energy)
    print(f"\nFinal Energy: {result5['final_energy']:.10f} Ha")
    print(f"Error (vs FCI): {error5:.10f} Ha ({error5*1000:.3f} mHa)")
    
    # ==========================================
    # 6. UCCSD + QNG (Device)
    # ==========================================
    print("\n\n" + "█"*100)
    print("6. UCCSD + QNG (Device Runtime with QFI)")
    print("█"*100)
    
    uccsd_qng = UCCSD(h2, runtime="device")
    
    print(f"\n[QFI Precomputation]")
    optimizer6 = QNGOptimizer(uccsd_qng, verbose=True)
    result6 = optimizer6.optimize(shots=1024, max_iterations=30, qfi_shots=1024)
    results["UCCSD_QNG_Device"] = result6
    
    error6 = abs(result6["final_energy"] - fci_energy)
    print(f"\nFinal Energy: {result6['final_energy']:.10f} Ha")
    print(f"Error (vs FCI): {error6:.10f} Ha ({error6*1000:.3f} mHa)")
    
    # ==========================================
    # Summary Table
    # ==========================================
    print("\n\n" + "="*120)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*120)
    print(f"{'Algorithm':<12} {'Method':<8} {'Runtime':<12} {'Iterations':<12} {'Error (mHa)':<14} {'Total Shots':<15}")
    print(f"{'-'*120}")
    
    print(f"{'HEA':<12} {'PSR':<8} {'Numeric':<12} {result1['iterations']:<12} {error1*1000:<14.3f} {'N/A':<15}")
    print(f"{'HEA':<12} {'PSR':<8} {'Device':<12} {result2['iterations']:<12} {error2*1000:<14.3f} {result2['total_shots']:<15d}")
    print(f"{'HEA':<12} {'QNG':<8} {'Device+QFI':<12} {result3['iterations']:<12} {error3*1000:<14.3f} {result3['total_shots']:<15d}")
    print(f"{'UCCSD':<12} {'PSR':<8} {'Numeric':<12} {result4['iterations']:<12} {error4*1000:<14.3f} {'N/A':<15}")
    print(f"{'UCCSD':<12} {'PSR':<8} {'Device':<12} {result5['iterations']:<12} {error5*1000:<14.3f} {result5['total_shots']:<15d}")
    print(f"{'UCCSD':<12} {'QNG':<8} {'Device+QFI':<12} {result6['iterations']:<12} {error6*1000:<14.3f} {result6['total_shots']:<15d}")
    print(f"{'-'*120}")
    print(f"FCI Reference: {fci_energy:.10f} Ha")
    
    # ==========================================
    # Key Insights
    # ==========================================
    print("\n\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)
    
    print("\n1. CONVERGENCE COMPARISON (Iterations):")
    print(f"   HEA:   PSR={result1['iterations']}, QNG={result3['iterations']} "
          f"(QNG {result1['iterations']/max(result3['iterations'], 1):.1f}x faster)")
    print(f"   UCCSD: PSR={result4['iterations']}, QNG={result6['iterations']} "
          f"(QNG {result4['iterations']/max(result6['iterations'], 1):.1f}x faster)")
    
    print("\n2. ACCURACY COMPARISON (Error in mHa):")
    print(f"   HEA:   PSR={error1*1000:.3f}, QNG={error3*1000:.3f}")
    print(f"   UCCSD: PSR={error4*1000:.3f}, QNG={error6*1000:.3f}")
    
    print("\n3. SHOT EFFICIENCY (Device mode):")
    print(f"   HEA QNG:   Total shots: {result3['total_shots']:,} (QFI: {result3['shots_qfi']:,}, Opt: {result3['shots_optimization']:,})")
    print(f"   UCCSD QNG: Total shots: {result6['total_shots']:,} (QFI: {result6['shots_qfi']:,}, Opt: {result6['shots_optimization']:,})")
    
    print("\n" + "="*100)
    print("✅ Analysis Complete!")
    print("="*100)


if __name__ == "__main__":
    main()
