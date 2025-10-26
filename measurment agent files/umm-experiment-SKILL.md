# UMM Experiment Skill

## Purpose
This skill provides best practices for designing, running, and reproducing quantum measurement experiments using the Universal Measurement Machine (UMM) framework. Use this when users want to validate algorithms, benchmark performance, or reproduce paper results.

## Core Experiment Types

### 1. Single-Run Validation
Quick test to verify correctness of implementation.

### 2. Statistical Benchmarking
Multiple runs with statistics to characterize performance distributions.

### 3. Parameter Sweeps
Systematic variation of system parameters to map performance landscape.

### 4. Ablation Studies
Isolate impact of specific algorithmic components.

### 5. Comparative Analysis
Benchmark adaptive vs baseline strategies.

## Spec Invariants for UMM Experiments

Before running any experiment, verify:
- **INVARIANT 1**: n_runs ≥ 30 for statistical significance
- **INVARIANT 2**: Noise parameters are physical (T1, T2 > 0; errors ∈ [0,1])
- **INVARIANT 3**: Random seeds are set for reproducibility
- **INVARIANT 4**: Target metrics are well-defined and computable
- **INVARIANT 5**: Baseline comparisons use identical noise models
- **INVARIANT 6**: Simulation timesteps respect hardware constraints (dt ≥ 1ns)

## Experiment Configuration Format

### Standard Config Structure
```json
{
  "experiment": {
    "name": "bloch_steering_adaptive",
    "description": "Adaptive state prep vs fixed baseline",
    "type": "comparative_benchmark"
  },
  "task": {
    "type": "state_prep",
    "target_state": "|+x>",
    "constraints": {"budget_time_us": 10.0},
    "cost_weights": {"backaction": 0.5, "time": 0.2}
  },
  "simulation": {
    "n_qubits": 1,
    "T1": 50e-6,
    "T2": 50e-6,
    "readout_error": 0.01,
    "gate_error": 0.001,
    "n_runs": 100,
    "n_steps": 32,
    "seed": 42
  },
  "metrics": {
    "primary": "fidelity",
    "secondary": ["backaction", "purity", "wall_time"],
    "target_thresholds": {
      "fidelity": 0.95,
      "backaction": 10.0
    }
  },
  "output": {
    "save_trajectories": true,
    "save_histograms": true,
    "save_ir_programs": true,
    "plot_formats": ["png", "pdf"]
  }
}
```

## Experiment Patterns

### Pattern 1: Bloch Steering (State Preparation)

**Objective**: Demonstrate adaptive weak measurements outperform fixed strategies.

**Configuration**:
```python
config = {
    "task": "state_prep",
    "target_state": "|+x>",
    "n_runs": 100,
    "n_steps": 32,
    "T1": 50e-6,
    "T2": 50e-6,
    "target_delta_F": 0.02  # Minimum improvement threshold
}
```

**Implementation Template**:
```python
from umm.core import UMMSimulator
from umm.intent import ObjectiveParser
import numpy as np
import matplotlib.pyplot as plt

def run_bloch_steering_experiment(config):
    """Run Bloch steering experiment."""
    
    # Setup
    simulator = UMMSimulator(
        n_qubits=1,
        T1=config["T1"],
        T2=config["T2"],
        seed=config.get("seed", 42)
    )
    
    parser = ObjectiveParser()
    objective = {
        "task": "state_prep",
        "target_state": config["target_state"],
        "cost_weights": {"backaction": 0.5, "time": 0.2}
    }
    reward_fn, constraints = parser.compile(objective)
    
    # Target state
    from umm.core.quantum_state import ket_plus
    target = ket_plus()
    
    # Run trials
    fixed_fidelities = []
    adaptive_fidelities = []
    
    for run in range(config["n_runs"]):
        # Fixed baseline
        simulator.reset()
        for t in range(config["n_steps"]):
            simulator._execute_instruction(
                WeakMeasurementInstruction(
                    axis=np.array([1, 0, 0]),
                    strength=0.3
                )
            )
            simulator._execute_instruction(WaitInstruction(1e-6))
        
        fixed_fidelities.append(simulator.state.fidelity(target))
        
        # Adaptive policy
        simulator.reset()
        policy = SimpleAdaptivePolicy(target_axis=np.array([1, 0, 0]))
        for t in range(config["n_steps"]):
            action = policy(simulator.history, simulator.state)
            simulator._execute_instruction(action)
            simulator._execute_instruction(WaitInstruction(1e-6))
        
        adaptive_fidelities.append(simulator.state.fidelity(target))
    
    # Analysis
    results = {
        "mean_fixed": np.mean(fixed_fidelities),
        "std_fixed": np.std(fixed_fidelities),
        "mean_adaptive": np.mean(adaptive_fidelities),
        "std_adaptive": np.std(adaptive_fidelities),
        "delta_F": np.mean(adaptive_fidelities) - np.mean(fixed_fidelities)
    }
    
    # Statistical test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(adaptive_fidelities, fixed_fidelities)
    results["t_statistic"] = t_stat
    results["p_value"] = p_value
    
    # Visualization
    plot_fidelity_comparison(fixed_fidelities, adaptive_fidelities)
    
    return results
```

**Success Criteria**:
- ΔF ≥ 0.02 (2% improvement)
- p < 0.01 (statistical significance)
- Both strategies converge (no divergence)

### Pattern 2: Quantum Zeno Stabilization

**Objective**: Demonstrate learned schedules outperform fixed-rate measurements.

**Configuration**:
```python
config = {
    "task": "zeno",
    "target_subspace": "|+x>",
    "n_runs": 50,
    "protocol_duration_us": 50.0,
    "T1": 50e-6,
    "T2": 30e-6,  # Shorter T2 to test dephasing suppression
    "strategies": ["fixed_rate", "exponential", "adaptive"]
}
```

**Key Metrics**:
- Survival probability: P_survive(T) = Tr(Π_target ρ(T))
- Average measurement count: N_avg
- Effective dephasing rate: Γ_eff extracted from exponential decay

**Implementation Notes**:
1. Fixed rate: Δt = T_total / N_measurements
2. Exponential: Δt_n = Δt_0 × exp(αn), tune α
3. Adaptive: Policy learns optimal timing from reward R = P_survive - λ×N_meas

### Pattern 3: Adaptive Metrology

**Objective**: Demonstrate quantum Fisher information scales with N² vs N.

**Configuration**:
```python
config = {
    "task": "phase_est",
    "true_phase": np.pi / 4,  # Hidden parameter
    "n_probes": [10, 20, 50, 100, 200],  # Sweep probe count
    "n_runs": 30,
    "strategies": ["separable", "entangled_adaptive"]
}
```

**Key Metrics**:
- Variance: Var(φ_est)
- Quantum Fisher Information: F_Q
- Cramer-Rao bound: Var(φ) ≥ 1/(NF_Q)
- Scaling: log(Var) vs log(N) should have slope -2 for entangled

**Success Criteria**:
- Adaptive strategy achieves Heisenberg limit scaling: Var ∝ 1/N²
- Fixed strategy shows standard quantum limit: Var ∝ 1/N
- Confidence intervals contain true phase 95% of time

### Pattern 4: Noise Robustness Study

**Objective**: Characterize performance degradation under realistic noise.

**Configuration**:
```python
config = {
    "task": "state_prep",
    "target_state": "|+x>",
    "parameter_sweep": {
        "T1_values": np.logspace(-5, -4, 10),  # 10µs to 100µs
        "T2_values": np.logspace(-5, -4, 10),
        "readout_errors": [0.001, 0.01, 0.05, 0.1]
    },
    "n_runs_per_point": 20
}
```

**Analysis**:
1. Create heatmap: Fidelity(T1, T2)
2. Identify operating regimes (high/low fidelity)
3. Determine critical T1/T2 thresholds
4. Compare to theoretical noise bounds

## Statistical Analysis Best Practices

### Sample Size Determination

For comparing two strategies with target effect size δ:
```
n ≥ 2(Z_α + Z_β)²σ² / δ²
```

Where:
- Z_α = 1.96 (95% confidence)
- Z_β = 0.84 (80% power)
- σ = estimated standard deviation
- δ = minimum detectable effect

**Rule of thumb**: n ≥ 30 for t-test validity, n ≥ 100 for tight confidence intervals.

### Significance Testing

Always report:
1. **Mean and std**: μ ± σ
2. **Confidence intervals**: [μ - 1.96σ/√n, μ + 1.96σ/√n]
3. **Effect size**: Cohen's d = (μ₁ - μ₂) / σ_pooled
4. **Statistical test**: t-test, Wilcoxon, or ANOVA as appropriate
5. **p-value**: with Bonferroni correction for multiple comparisons

```python
from scipy import stats

def analyze_experiment_results(strategy_a, strategy_b):
    """Statistical analysis of two strategies."""
    
    # Descriptive statistics
    results = {
        "mean_a": np.mean(strategy_a),
        "std_a": np.std(strategy_a, ddof=1),
        "mean_b": np.mean(strategy_b),
        "std_b": np.std(strategy_b, ddof=1)
    }
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((results["std_a"]**2 + results["std_b"]**2) / 2)
    results["cohens_d"] = (results["mean_b"] - results["mean_a"]) / pooled_std
    
    # Hypothesis test
    t_stat, p_value = stats.ttest_ind(strategy_b, strategy_a)
    results["t_statistic"] = t_stat
    results["p_value"] = p_value
    results["significant"] = p_value < 0.05
    
    # Confidence intervals
    n_a, n_b = len(strategy_a), len(strategy_b)
    results["ci_a"] = stats.t.interval(
        0.95, n_a-1, loc=results["mean_a"], 
        scale=results["std_a"]/np.sqrt(n_a)
    )
    results["ci_b"] = stats.t.interval(
        0.95, n_b-1, loc=results["mean_b"],
        scale=results["std_b"]/np.sqrt(n_b)
    )
    
    return results
```

### Handling Outliers

```python
def detect_outliers(data, method="iqr"):
    """Detect and optionally remove outliers."""
    
    if method == "iqr":
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
    
    elif method == "zscore":
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > 3
    
    return outliers, data[~outliers]
```

**Policy**: Report outliers but include them in analysis unless they indicate simulation errors.

## Visualization Guidelines

### Required Plots

1. **Histograms**: Distribution of final fidelity/metric
2. **Box plots**: Compare strategies side-by-side
3. **Trajectory plots**: Fidelity vs time for representative runs
4. **Heatmaps**: Performance across parameter space
5. **Convergence plots**: Mean ± std vs measurement step

### Plot Specifications

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_experiment_results(results, save_path="results.png"):
    """Create comprehensive visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (0,0): Histogram comparison
    ax = axes[0, 0]
    ax.hist(results["fixed"], bins=30, alpha=0.5, label="Fixed", density=True)
    ax.hist(results["adaptive"], bins=30, alpha=0.5, label="Adaptive", density=True)
    ax.set_xlabel("Final Fidelity")
    ax.set_ylabel("Density")
    ax.set_title("Distribution Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (0,1): Box plot
    ax = axes[0, 1]
    ax.boxplot([results["fixed"], results["adaptive"]], 
                labels=["Fixed", "Adaptive"])
    ax.set_ylabel("Fidelity")
    ax.set_title("Statistical Summary")
    ax.grid(True, alpha=0.3, axis='y')
    
    # (1,0): Trajectory examples
    ax = axes[1, 0]
    for i in range(min(5, len(results["trajectories_fixed"]))):
        ax.plot(results["trajectories_fixed"][i], 'b-', alpha=0.3)
        ax.plot(results["trajectories_adaptive"][i], 'r-', alpha=0.3)
    ax.set_xlabel("Measurement Step")
    ax.set_ylabel("Fidelity")
    ax.set_title("Example Trajectories")
    ax.grid(True, alpha=0.3)
    
    # (1,1): Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    Fixed Strategy:
      Mean: {results['mean_fixed']:.4f}
      Std:  {results['std_fixed']:.4f}
    
    Adaptive Strategy:
      Mean: {results['mean_adaptive']:.4f}
      Std:  {results['std_adaptive']:.4f}
    
    Improvement: ΔF = {results['delta_F']:.4f}
    Statistical: p = {results['p_value']:.2e}
    Effect size: d = {results['cohens_d']:.2f}
    """
    ax.text(0.1, 0.5, summary_text, fontsize=11, 
            family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

## Reproducibility Checklist

Ensure every experiment includes:

- [ ] **Config file**: JSON with all parameters
- [ ] **Random seeds**: Fixed and documented
- [ ] **Versions**: UMM version, Python version, dependency versions
- [ ] **Hardware specs**: If using actual quantum hardware
- [ ] **Noise model**: Explicit T1, T2, error rates
- [ ] **Number of runs**: Statistical power calculation
- [ ] **Success criteria**: Pre-specified before running
- [ ] **Code availability**: Scripts included with results
- [ ] **Data storage**: Raw outcomes saved for reanalysis
- [ ] **Timestamp**: When experiment was run

### Reproducibility Template

```python
def create_experiment_manifest(config, results, code_hash):
    """Generate manifest for reproducibility."""
    
    import datetime
    import hashlib
    import platform
    
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "umm_version": "0.1.0",  # From __version__
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "config": config,
        "results_summary": {
            k: v for k, v in results.items() 
            if not isinstance(v, (list, np.ndarray))
        },
        "code_hash": code_hash,
        "random_seed": config.get("seed"),
        "n_runs": config["n_runs"]
    }
    
    return manifest
```

## Performance Optimization

### Parallelization

For large n_runs, parallelize across trials:

```python
from multiprocessing import Pool
from functools import partial

def run_single_trial(trial_id, config):
    """Run single trial with unique seed."""
    config_copy = config.copy()
    config_copy["seed"] = config["seed"] + trial_id
    simulator = UMMSimulator(**config_copy)
    # ... run trial ...
    return result

def run_parallel_experiment(config, n_workers=4):
    """Run experiment in parallel."""
    
    with Pool(n_workers) as pool:
        results = pool.map(
            partial(run_single_trial, config=config),
            range(config["n_runs"])
        )
    
    return results
```

### Memory Management

For long experiments with trajectory storage:

```python
import h5py

def save_trajectories_hdf5(trajectories, filename):
    """Save large trajectory data efficiently."""
    
    with h5py.File(filename, 'w') as f:
        for i, traj in enumerate(trajectories):
            f.create_dataset(f"trial_{i}", data=traj, compression="gzip")
```

## Common Experiment Issues

### Issue 1: Non-Convergence
**Symptom**: Fidelity oscillates wildly, no clear trend
**Diagnosis**: Check if T_budget << T2 (not enough time)
**Solution**: Increase budget or reduce measurement strength

### Issue 2: Statistical Insignificance
**Symptom**: p-value > 0.05 despite visible difference
**Diagnosis**: Too few runs, high variance
**Solution**: Increase n_runs or reduce noise

### Issue 3: Biased Comparison
**Symptom**: Adaptive wins but uses different noise model
**Diagnosis**: Baselines not using same simulation setup
**Solution**: Ensure identical simulator instance for all strategies

### Issue 4: Overfitting to Noise Model
**Symptom**: Adaptive fails when noise parameters change slightly
**Diagnosis**: Policy trained on narrow parameter range
**Solution**: Use domain randomization during training

### Issue 5: Numerical Instability
**Symptom**: NaN or Inf in density matrices
**Diagnosis**: Accumulation of numerical errors over long sequences
**Solution**: Renormalize density matrix every N steps, use higher precision

## Paper Experiment Reproduction

### Section 6.2: Bloch Steering

Run with config from `configs/state_prep_example.json`:
```bash
python experiments/bloch_steering.py --config configs/state_prep_example.json
```

**Expected results**:
- Fixed: F = 0.93 ± 0.04
- Adaptive: F = 0.96 ± 0.03
- ΔF ≥ 0.02, p < 0.01

### Section 6.3: Quantum Zeno

Run with config from `configs/zeno_example.json`:
```bash
python experiments/zeno_stabilization.py --config configs/zeno_example.json
```

**Expected results**:
- Fixed rate: P_survive(50µs) ≈ 0.75
- Adaptive: P_survive(50µs) ≈ 0.89
- Reduction in effective Γ_φ by factor of 2-3

### Section 6.4: Adaptive Metrology

Run with config from `configs/metrology_example.json`:
```bash
python experiments/adaptive_metrology.py --config configs/metrology_example.json
```

**Expected results**:
- Separable: σ²(φ) ∝ 1/N (SQL)
- Entangled: σ²(φ) ∝ 1/N² (Heisenberg)
- Crossover at N ≈ 20 probes

### Section 7: Superconducting Demo

Run with hardware parameters from `configs/superconducting_params.yaml`:
```bash
python experiments/superconducting_demo.py --config configs/superconducting_params.yaml
```

**Expected results**:
- State prep fidelity > 0.99 (vs 0.95 for fixed)
- Protocol duration < 5µs
- Back-action cost < 3.0

## Integration Testing

Test UMM components together:

```python
def test_end_to_end_pipeline():
    """Test complete UMM pipeline."""
    
    # 1. Parse objective
    parser = ObjectiveParser()
    objective = {
        "task": "state_prep",
        "target_state": "|+x>",
        "constraints": {"budget_time_us": 10.0}
    }
    reward_fn, constraints = parser.compile(objective)
    
    # 2. Create simulator
    simulator = UMMSimulator(n_qubits=1, T1=50e-6, T2=50e-6)
    
    # 3. Build IR program
    from umm.core import IRBuilder
    program = IRBuilder().reset() \
        .weak_measure(np.array([1,0,0]), 0.3) \
        .wait(1e-6).build()
    
    # 4. Execute
    result = simulator.execute_program(program)
    
    # 5. Validate
    assert result.final_state is not None
    assert result.purity > 0.0
    assert len(result.history) > 0
```

## Summary

Successful UMM experiments require:
1. **Clear objectives**: Pre-specify success criteria
2. **Statistical rigor**: Adequate sample size and significance testing
3. **Reproducibility**: Complete configuration and seed management
4. **Validation**: Compare against baselines and theoretical bounds
5. **Visualization**: Clear plots that tell the story

Run experiments systematically and document thoroughly for maximum scientific impact.
