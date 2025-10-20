"""
Bloch Steering Experiment: Adaptive state preparation.

Task: Prepare |+x> from unknown mixed state under dephasing.
Demonstrates adaptive weak measurements outperform fixed strategies.

Reproduces predictions from Section 6.2 of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import json
import argparse

from umm.core import UMMSimulator, IRBuilder, WeakMeasurementInstruction
from umm.intent import ObjectiveParser
from umm.core.quantum_state import ket_plus


class SimpleAdaptivePolicy:
    """
    Simple adaptive policy for Bloch steering.
    
    Strategy: Measure along axis perpendicular to current Bloch vector,
    with strength proportional to distance from target.
    """
    
    def __init__(self, target_axis: np.ndarray = np.array([1, 0, 0])):
        self.target_axis = target_axis / np.linalg.norm(target_axis)
    
    def __call__(self, history, state) -> WeakMeasurementInstruction:
        """Select next measurement action."""
        # Get current Bloch vector
        bloch = state.to_bloch_vector()
        
        # Compute distance to target
        distance = np.linalg.norm(bloch - self.target_axis)
        
        # Measurement strength: higher when farther from target
        strength = min(0.8, 0.3 + 0.5 * distance)
        
        # Measurement axis: perpendicular to current position
        # (simplified: just use target axis)
        axis = self.target_axis
        
        return WeakMeasurementInstruction(
            axis=axis,
            strength=strength
        )


def run_fixed_baseline(
    simulator: UMMSimulator,
    target_state: np.ndarray,
    n_steps: int = 32,
    strength: float = 0.3
) -> Tuple[float, List[float]]:
    """Run fixed-basis measurement strategy."""
    simulator.reset()
    fidelities = []
    
    for t in range(n_steps):
        # Fixed weak measurement along X
        action = WeakMeasurementInstruction(
            axis=np.array([1, 0, 0]),
            strength=strength
        )
        simulator._execute_instruction(action)
        
        # Track fidelity
        fidelity = simulator.state.fidelity(target_state)
        fidelities.append(fidelity)
        
        # Wait
        from umm.core.ir import WaitInstruction
        simulator._execute_instruction(WaitInstruction(duration=1e-6))
    
    final_fidelity = simulator.state.fidelity(target_state)
    return final_fidelity, fidelities


def run_adaptive(
    simulator: UMMSimulator,
    target_state: np.ndarray,
    policy: SimpleAdaptivePolicy,
    n_steps: int = 32
) -> Tuple[float, List[float]]:
    """Run adaptive policy."""
    simulator.reset()
    fidelities = []
    
    for t in range(n_steps):
        # Adaptive action
        action = policy(simulator.history, simulator.state)
        simulator._execute_instruction(action)
        
        # Track fidelity
        fidelity = simulator.state.fidelity(target_state)
        fidelities.append(fidelity)
        
        # Wait
        from umm.core.ir import WaitInstruction
        simulator._execute_instruction(WaitInstruction(duration=1e-6))
    
    final_fidelity = simulator.state.fidelity(target_state)
    return final_fidelity, fidelities


def main(config_path: str = None):
    """Run Bloch steering experiment."""
    
    # Load configuration
    if config_path:
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            "n_runs": 100,
            "n_steps": 32,
            "T1": 50e-6,
            "T2": 50e-6,
            "target_delta_F": 0.02
        }
    
    print("=" * 70)
    print("Bloch Steering Experiment: Adaptive vs Fixed")
    print("=" * 70)
    print(f"Configuration: {config}")
    print()
    
    # Setup
    target_state = ket_plus()  # |+x>
    n_runs = config["n_runs"]
    n_steps = config["n_steps"]
    
    # Initialize simulator
    simulator = UMMSimulator(
        n_qubits=1,
        T1=config["T1"],
        T2=config["T2"],
        readout_error=0.01,
        gate_error=0.001
    )
    
    # Initialize policy
    policy = SimpleAdaptivePolicy(target_axis=np.array([1, 0, 0]))
    
    # Run experiments
    print(f"Running {n_runs} trials...")
    print()
    
    fixed_fidelities = []
    adaptive_fidelities = []
    
    for run in range(n_runs):
        # Fixed strategy
        f_fixed, _ = run_fixed_baseline(simulator, target_state, n_steps, strength=0.3)
        fixed_fidelities.append(f_fixed)
        
        # Adaptive strategy
        f_adaptive, _ = run_adaptive(simulator, target_state, policy, n_steps)
        adaptive_fidelities.append(f_adaptive)
        
        if (run + 1) % 20 == 0:
            print(f"  Progress: {run + 1}/{n_runs}")
    
    # Analyze results
    mean_fixed = np.mean(fixed_fidelities)
    mean_adaptive = np.mean(adaptive_fidelities)
    std_fixed = np.std(fixed_fidelities)
    std_adaptive = np.std(adaptive_fidelities)
    
    delta_F = mean_adaptive - mean_fixed
    
    print()
    print("Results:")
    print(f"  Fixed strategy:    F = {mean_fixed:.4f} ± {std_fixed:.4f}")
    print(f"  Adaptive strategy: F = {mean_adaptive:.4f} ± {std_adaptive:.4f}")
    print(f"  Improvement ΔF:    {delta_F:.4f}")
    print()
    
    # Statistical test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(adaptive_fidelities, fixed_fidelities)
    print(f"Statistical significance: t={t_stat:.3f}, p={p_value:.4e}")
    print()
    
    # Check against target
    target_delta = config["target_delta_F"]
    if delta_F >= target_delta:
        print(f"✓ SUCCESS: ΔF = {delta_F:.4f} ≥ {target_delta:.4f} (target)")
    else:
        print(f"✗ FAIL: ΔF = {delta_F:.4f} < {target_delta:.4f} (target)")
    print()
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(fixed_fidelities, bins=30, alpha=0.5, label='Fixed', density=True)
    plt.hist(adaptive_fidelities, bins=30, alpha=0.5, label='Adaptive', density=True)
    plt.xlabel('Final Fidelity')
    plt.ylabel('Density')
    plt.title('Bloch Steering: Adaptive vs Fixed Strategy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bloch_steering_results.png', dpi=150)
    print("Saved plot: bloch_steering_results.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bloch steering experiment")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    args = parser.parse_args()
    
    main(config_path=args.config)
