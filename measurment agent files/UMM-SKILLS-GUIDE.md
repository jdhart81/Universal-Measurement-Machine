# Universal Measurement Machine (UMM) Skills Guide

## Overview

This guide provides a comprehensive workflow for using the Universal Measurement Machine skills. The UMM framework consists of four complementary skills that cover the complete lifecycle of quantum measurement design, from task specification to optimization.

## Skill Ecosystem

### 1. **UMM Task Design** (`umm-task-design-SKILL.md`)
**Purpose**: Formulate quantum measurement objectives with proper constraints and cost functions.

**Use when**:
- Starting a new measurement task
- Defining objectives for policy training
- Specifying constraints and trade-offs
- Designing DSL or JSON specifications

**Key outputs**: Valid objective specifications, reward functions, constraint sets

### 2. **UMM Experiment** (`umm-experiment-SKILL.md`)
**Purpose**: Design, run, and reproduce quantum measurement experiments with statistical rigor.

**Use when**:
- Validating algorithm correctness
- Benchmarking adaptive vs baseline strategies
- Running parameter sweeps
- Reproducing paper results
- Conducting ablation studies

**Key outputs**: Experimental data, statistical analyses, visualizations, reproducibility manifests

### 3. **UMM Policy Training** (`umm-policy-training-SKILL.md`)
**Purpose**: Train adaptive measurement policies using reinforcement learning.

**Use when**:
- Developing learned policies for measurement tasks
- Training neural networks (Transformer, LSTM, GNN)
- Implementing RL algorithms (PPO, SAC, PG)
- Applying domain randomization for robustness
- Debugging training issues

**Key outputs**: Trained policies, training logs, evaluation metrics, checkpoints

### 4. **UMM Analysis** (`umm-analysis-SKILL.md`)
**Purpose**: Analyze results, interpret policy behavior, and optimize measurement protocols.

**Use when**:
- Understanding convergence dynamics
- Quantifying back-action and information gain
- Interpreting learned strategies
- Testing noise sensitivity
- Optimizing hyperparameters
- Profiling performance

**Key outputs**: Analysis reports, visualizations, optimization recommendations, diagnostic insights

## Complete UMM Workflow

### Phase 1: Task Specification (Task Design Skill)

```python
# 1. Define objective using Task Design skill
objective = {
    "task": "state_prep",
    "target_state": "|+x>",
    "constraints": {
        "budget_time_us": 10.0,
        "max_strength": 0.8
    },
    "cost_weights": {
        "backaction": 0.5,
        "time": 0.2
    }
}

# 2. Validate against invariants
assert objective["cost_weights"]["backaction"] >= 0
assert objective["constraints"]["budget_time_us"] > 0
# ... other validation checks from Task Design skill
```

**Deliverable**: Validated objective specification ready for compilation

### Phase 2: Baseline Establishment (Experiment Skill)

```python
# 3. Run baseline experiments
from umm.core import UMMSimulator
from umm.intent import ObjectiveParser

parser = ObjectiveParser()
reward_fn, constraints = parser.compile(objective)

# Fixed-strategy baseline
baseline_results = run_fixed_baseline(
    simulator=UMMSimulator(n_qubits=1, T1=50e-6, T2=50e-6),
    target_state=ket_plus(),
    n_runs=100,
    n_steps=32
)

# 4. Establish performance targets
target_fidelity = baseline_results["mean_fidelity"] + 0.02  # 2% improvement
```

**Deliverable**: Baseline performance metrics, target thresholds

### Phase 3: Policy Development (Policy Training Skill)

```python
# 5. Select architecture (from Policy Training skill)
from umm.policy import TransformerPolicy

policy = TransformerPolicy(
    d_model=128,
    n_heads=4,
    n_layers=3,
    max_seq_len=32
)

# 6. Configure training
training_config = {
    "algorithm": "ppo",
    "n_episodes": 50000,
    "learning_rate": 3e-4,
    "domain_randomization": {
        "T1": (30e-6, 100e-6),
        "T2": (30e-6, 100e-6)
    }
}

# 7. Train policy
trained_policy = train_umm_policy(
    objective_spec=objective,
    architecture="transformer",
    **training_config
)

# 8. Evaluate
eval_results = evaluate_policy(
    policy=trained_policy,
    objective_spec=objective,
    n_trials=100
)
```

**Deliverable**: Trained policy, evaluation metrics, training curves

### Phase 4: Experimental Validation (Experiment Skill)

```python
# 9. Run comparative experiments
from experiments.bloch_steering import run_bloch_steering_experiment

experiment_config = {
    "n_runs": 100,
    "n_steps": 32,
    "T1": 50e-6,
    "T2": 50e-6,
    "target_delta_F": 0.02
}

results = run_bloch_steering_experiment(experiment_config)

# 10. Statistical testing
assert results["delta_F"] >= 0.02, "Did not meet improvement threshold"
assert results["p_value"] < 0.01, "Not statistically significant"

# 11. Generate reproducibility manifest
manifest = create_experiment_manifest(
    config=experiment_config,
    results=results,
    code_hash=hash_codebase()
)
```

**Deliverable**: Validated results, statistical significance, reproducibility documentation

### Phase 5: Analysis and Interpretation (Analysis Skill)

```python
# 12. Trajectory analysis
fidelity_metrics = analyze_fidelity_convergence(
    trajectories=results["fidelity_trajectories"],
    target_fidelity=0.95
)

print(f"Convergence time: {fidelity_metrics['mean_convergence_time']:.1f} steps")
print(f"Final stability: {fidelity_metrics['final_stability']:.4f}")

# 13. Back-action analysis
backaction_metrics = analyze_backaction(
    histories=results["histories"],
    target_state=ket_plus()
)

print(f"Mean back-action: {backaction_metrics['mean_backaction']:.2f}")
print(f"Efficiency: {backaction_metrics['mean_efficiency']:.4f}")

# 14. Policy behavior analysis
policy_metrics = analyze_policy_behavior(
    policy=trained_policy,
    test_states=generate_test_states(100),
    n_samples=1000
)

print(f"Axis coherence: {policy_metrics['axis_coherence']:.3f}")
print(f"Strength-distance correlation: {policy_metrics['strength_distance_correlation']:.3f}")

# 15. Generate visualizations
plot_fidelity_analysis(
    trajectories=results["fidelity_trajectories"],
    save_path="analysis/fidelity_analysis.png"
)
```

**Deliverable**: Comprehensive analysis report, visualizations, insights

### Phase 6: Optimization (Analysis Skill)

```python
# 16. Noise sensitivity analysis
sensitivity_results = analyze_noise_sensitivity(
    policy=trained_policy,
    objective_spec=objective,
    parameter_ranges={
        "T1": (30e-6, 100e-6),
        "T2": (30e-6, 100e-6),
        "readout_error": (0.001, 0.05)
    },
    n_samples_per_param=20
)

# 17. Identify operating regimes
optimal_region = find_optimal_operating_region(sensitivity_results)
print(f"Optimal T1 range: {optimal_region['T1']}")
print(f"Optimal T2 range: {optimal_region['T2']}")

# 18. Hyperparameter optimization (if needed)
from skopt import gp_minimize

optimized_params = optimize_measurement_protocol(objective)
print(f"Optimized parameters: {optimized_params}")
```

**Deliverable**: Optimized protocols, operating regime identification, performance bounds

## Skill Integration Patterns

### Pattern 1: Rapid Prototyping

For quick validation of ideas:

```
1. Task Design → Define simple objective
2. Experiment → Run with parametric policy (no training)
3. Analysis → Check if concept works
4. Decision: Train full policy or iterate on design
```

### Pattern 2: Research Publication

For rigorous scientific results:

```
1. Task Design → Formalize objective with theoretical justification
2. Policy Training → Train with domain randomization
3. Experiment → Run comprehensive benchmarks (n≥100)
4. Analysis → Full statistical analysis + visualizations
5. Experiment → Reproduce results with new seeds
6. Output: Paper-ready figures and data
```

### Pattern 3: Hardware Deployment

For real quantum systems:

```
1. Task Design → Define with hardware constraints
2. Experiment → Validate in simulation with hardware noise
3. Policy Training → Train with realistic noise models
4. Analysis → Sensitivity analysis for calibration drift
5. Optimization → Fine-tune for hardware parameters
6. Deployment: Export compiled pulse sequences
```

### Pattern 4: Comparative Study

For benchmarking algorithms:

```
1. Task Design → Single objective for all algorithms
2. Policy Training → Train multiple architectures
3. Experiment → Run all on identical simulator instances
4. Analysis → Statistical comparison with effect sizes
5. Analysis → Interpret why certain approaches win
```

## Common Cross-Skill Scenarios

### Scenario 1: Policy Not Converging

**Symptoms**: Training loss plateaus, poor evaluation performance

**Workflow**:
1. **Task Design**: Verify reward function is informative
2. **Policy Training**: Check state representation, add reward shaping
3. **Analysis**: Profile to find bottlenecks
4. **Experiment**: Run ablation on architectural choices

### Scenario 2: Results Not Reproducible

**Symptoms**: Different runs give wildly different results

**Workflow**:
1. **Experiment**: Add random seed control
2. **Analysis**: Check for numerical instabilities
3. **Task Design**: Verify constraints are enforced
4. **Policy Training**: Add gradient clipping

### Scenario 3: Policy Overfits to Noise Model

**Symptoms**: Works in training, fails in validation

**Workflow**:
1. **Policy Training**: Increase domain randomization range
2. **Experiment**: Test on broader parameter sweeps
3. **Analysis**: Identify which parameters cause failure
4. **Optimization**: Retrain with refined robustness objectives

### Scenario 4: Baseline Stronger Than Expected

**Symptoms**: Adaptive policy doesn't beat simple baseline

**Workflow**:
1. **Task Design**: Check if objective favors baseline by construction
2. **Analysis**: Identify where baseline succeeds (operating regime)
3. **Task Design**: Reformulate to emphasize adaptivity advantages
4. **Policy Training**: Retrain on harder scenarios

## Skill Selection Decision Tree

```
START
  │
  ├─ Need to formulate task? → Use TASK DESIGN
  │
  ├─ Need to run experiments? → Use EXPERIMENT
  │   │
  │   ├─ Statistical benchmark? → Experiment (Pattern 1-3)
  │   ├─ Reproduce paper? → Experiment (Section on Reproduction)
  │   └─ Parameter sweep? → Experiment (Noise Robustness)
  │
  ├─ Need to train policy? → Use POLICY TRAINING
  │   │
  │   ├─ Choose architecture? → Policy Training (Architectures)
  │   ├─ Select algorithm? → Policy Training (Training Algorithms)
  │   ├─ Debugging training? → Policy Training (Debugging)
  │   └─ Domain randomization? → Policy Training (Robustness)
  │
  └─ Need to analyze results? → Use ANALYSIS
      │
      ├─ Understand convergence? → Analysis (Fidelity Analysis)
      ├─ Quantify back-action? → Analysis (Back-action Analysis)
      ├─ Interpret policy? → Analysis (Policy Behavior)
      ├─ Test robustness? → Analysis (Noise Sensitivity)
      └─ Optimize? → Analysis (Optimization Strategies)
```

## Best Practices for Multi-Skill Projects

### 1. Documentation Standards

Every UMM project should include:
- **`objective.json`**: Task specification (from Task Design)
- **`config.json`**: Experiment configuration (from Experiment)
- **`training_log.json`**: Training metrics (from Policy Training)
- **`analysis_report.md`**: Results and insights (from Analysis)
- **`manifest.json`**: Reproducibility metadata (from Experiment)

### 2. Directory Structure

```
umm-project/
├── objectives/          # Task specifications
│   ├── state_prep.json
│   └── zeno_stab.json
├── policies/            # Trained policies
│   ├── transformer_state_prep.pt
│   └── lstm_zeno.pt
├── experiments/         # Experiment scripts
│   ├── bloch_steering.py
│   └── configs/
├── results/            # Experimental data
│   ├── fidelity_trajectories.npy
│   ├── histories.pkl
│   └── manifest.json
├── analysis/           # Analysis outputs
│   ├── plots/
│   ├── reports/
│   └── sensitivity/
└── docs/              # Documentation
    ├── architecture.md
    └── experiments.md
```

### 3. Version Control

Track which skill versions were used:

```json
{
  "skills_used": {
    "task_design": "v1.0",
    "experiment": "v1.0",
    "policy_training": "v1.0",
    "analysis": "v1.0"
  },
  "umm_version": "0.1.0",
  "date": "2025-10-26"
}
```

### 4. Iterative Refinement

```
Iteration 1: Task Design → Experiment (baseline) → Analysis (identify gaps)
Iteration 2: Task Design (refine) → Policy Training → Experiment → Analysis
Iteration 3: Optimization → Final Experiment → Comprehensive Analysis
```

## Quick Reference

### When to Read Each Skill

| Question | Read |
|----------|------|
| How do I specify my task? | Task Design |
| What constraints should I use? | Task Design (Constraint Specification) |
| How do I run experiments? | Experiment |
| How many runs do I need? | Experiment (Statistical Analysis) |
| Which policy architecture? | Policy Training (Architectures) |
| Which RL algorithm? | Policy Training (Training Algorithms) |
| Training not working? | Policy Training (Debugging) |
| How do I analyze convergence? | Analysis (Fidelity Analysis) |
| How much back-action? | Analysis (Back-action Analysis) |
| Is my policy robust? | Analysis (Noise Sensitivity) |
| How do I optimize? | Analysis (Optimization Strategies) |

### Skill Dependencies

```
Task Design
    ↓
    ├─→ Experiment (needs objective spec)
    │       ↓
    │       └─→ Analysis (needs results)
    │
    └─→ Policy Training (needs objective spec)
            ↓
            ├─→ Experiment (validate trained policy)
            │       ↓
            │       └─→ Analysis (comprehensive evaluation)
            │
            └─→ Analysis (interpret policy behavior)
```

## Advanced Integration

### Custom Workflow Builder

```python
class UMMWorkflow:
    """Orchestrate multi-skill UMM workflow."""
    
    def __init__(self, objective_spec):
        self.objective = objective_spec
        self.results = {}
    
    def design_task(self):
        """Phase 1: Task Design"""
        from umm.intent import ObjectiveParser
        parser = ObjectiveParser()
        self.reward_fn, self.constraints = parser.compile(self.objective)
        print("✓ Task designed and validated")
    
    def run_baseline(self, n_runs=100):
        """Phase 2: Baseline Experiment"""
        baseline_results = run_experiments(
            policy="fixed",
            objective=self.objective,
            n_runs=n_runs
        )
        self.results["baseline"] = baseline_results
        print(f"✓ Baseline: F = {baseline_results['mean_fidelity']:.4f}")
    
    def train_policy(self, architecture="transformer", n_episodes=50000):
        """Phase 3: Policy Training"""
        policy = train_umm_policy(
            objective_spec=self.objective,
            architecture=architecture,
            n_episodes=n_episodes
        )
        self.policy = policy
        print("✓ Policy trained")
    
    def validate_policy(self, n_runs=100):
        """Phase 4: Validation Experiment"""
        validation_results = run_experiments(
            policy=self.policy,
            objective=self.objective,
            n_runs=n_runs
        )
        self.results["adaptive"] = validation_results
        print(f"✓ Adaptive: F = {validation_results['mean_fidelity']:.4f}")
    
    def analyze_results(self):
        """Phase 5: Comprehensive Analysis"""
        analysis = {
            "fidelity": analyze_fidelity_convergence(
                self.results["adaptive"]["trajectories"]
            ),
            "backaction": analyze_backaction(
                self.results["adaptive"]["histories"],
                self.objective["target_state"]
            ),
            "comparison": statistical_comparison(
                self.results["baseline"],
                self.results["adaptive"]
            )
        }
        self.analysis = analysis
        print("✓ Analysis complete")
    
    def optimize(self):
        """Phase 6: Optimization"""
        optimized = optimize_measurement_protocol(self.objective)
        self.optimized_params = optimized
        print("✓ Protocol optimized")
    
    def run_full_pipeline(self):
        """Execute complete workflow"""
        self.design_task()
        self.run_baseline()
        self.train_policy()
        self.validate_policy()
        self.analyze_results()
        self.optimize()
        print("\n✓ Complete UMM workflow finished")
        return self.generate_report()
```

### Usage

```python
# Define objective
objective = {
    "task": "state_prep",
    "target_state": "|+x>",
    "constraints": {"budget_time_us": 10.0},
    "cost_weights": {"backaction": 0.5, "time": 0.2}
}

# Run complete workflow
workflow = UMMWorkflow(objective)
report = workflow.run_full_pipeline()

# Access results
print(f"Improvement: ΔF = {report['delta_F']:.4f}")
print(f"Statistical significance: p = {report['p_value']:.2e}")
```

## Summary

The UMM skill ecosystem provides:
1. **Structured workflow**: From task design to optimization
2. **Modular skills**: Use independently or together
3. **Cross-skill patterns**: Common scenarios and solutions
4. **Integration tools**: Workflow orchestration and automation
5. **Best practices**: Documentation, versioning, reproducibility

Master each skill individually, then integrate them for comprehensive quantum measurement system design and optimization.

---

**Next Steps**:
1. Read the **Task Design** skill to formulate your objective
2. Use the **Experiment** skill to establish baselines
3. Apply the **Policy Training** skill to develop adaptive strategies
4. Leverage the **Analysis** skill to interpret and optimize
5. Iterate using cross-skill workflows for best results
