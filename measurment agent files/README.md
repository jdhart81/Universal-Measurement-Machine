# UMM Skills Package

**AI-Powered Quantum Measurement Design**

A comprehensive skill set for the Universal Measurement Machine (UMM) framework, enabling systematic design, training, experimentation, and optimization of adaptive quantum measurement protocols.

## Overview

This package contains five specialized skills that cover the complete lifecycle of quantum measurement system development:

1. **Task Design** - Formulate measurement objectives with proper constraints
2. **Experiment** - Design and run statistically rigorous experiments  
3. **Policy Training** - Train adaptive policies using reinforcement learning
4. **Analysis** - Interpret results and optimize protocols
5. **Skills Guide** - Master workflow integrating all skills

## Quick Start

### Basic Usage Pattern

```python
# 1. Define objective (Task Design skill)
objective = {
    "task": "state_prep",
    "target_state": "|+x>",
    "constraints": {"budget_time_us": 10.0},
    "cost_weights": {"backaction": 0.5, "time": 0.2}
}

# 2. Run experiment (Experiment skill)
from umm.core import UMMSimulator
from umm.intent import ObjectiveParser

parser = ObjectiveParser()
reward_fn, constraints = parser.compile(objective)
simulator = UMMSimulator(n_qubits=1, T1=50e-6, T2=50e-6)

# 3. Train policy (Policy Training skill)
from umm.policy import TransformerPolicy
policy = train_umm_policy(
    objective_spec=objective,
    architecture="transformer",
    n_episodes=50000
)

# 4. Analyze results (Analysis skill)
results = evaluate_policy(policy, objective, n_trials=100)
analysis = analyze_fidelity_convergence(results["trajectories"])
```

## Skill Descriptions

### 1. UMM Task Design Skill (`umm-task-design-SKILL.md`)

**When to use**: Starting a new measurement task, defining objectives, specifying constraints.

**Key features**:
- Objective formulation patterns for state prep, Zeno stabilization, entanglement concentration, adaptive metrology
- Constraint specification (time budgets, measurement rates, LOCC restrictions)
- Cost function design (back-action, time, probe count)
- Multi-objective optimization
- Validation checklists and common pitfalls

**Example objectives**:
```python
# State preparation
{"task": "state_prep", "target_state": "|+x>", ...}

# Quantum Zeno stabilization  
{"task": "zeno", "target_subspace": "|+x>", ...}

# Adaptive metrology
{"task": "phase_est", "parameter": "phi", ...}
```

### 2. UMM Experiment Skill (`umm-experiment-SKILL.md`)

**When to use**: Running experiments, benchmarking algorithms, reproducing results.

**Key features**:
- Experiment configuration standards
- Statistical analysis (sample size, significance testing, effect sizes)
- Visualization guidelines (histograms, trajectories, heatmaps)
- Reproducibility checklists (random seeds, manifests, versioning)
- Performance optimization (parallelization, memory management)
- Paper result reproduction

**Experiment patterns**:
- Single-run validation
- Statistical benchmarking  
- Parameter sweeps
- Ablation studies
- Comparative analysis

### 3. UMM Policy Training Skill (`umm-policy-training-SKILL.md`)

**When to use**: Developing learned policies, training neural networks, applying RL.

**Key features**:
- Policy architectures (Transformer, LSTM, GNN, parametric)
- Training algorithms (PPO, REINFORCE, SAC, model-based)
- Domain randomization for robustness
- State representation strategies
- Reward shaping and curriculum learning
- Safety constraint enforcement
- Debugging training issues

**Supported architectures**:
```python
# Transformer (best for long sequences)
policy = TransformerPolicy(d_model=128, n_heads=4)

# LSTM (good balance)
policy = LSTMPolicy(hidden_dim=128, n_layers=2)

# GNN (multi-qubit systems)
policy = GNNPolicy(n_qubits=2, hidden_dim=64)
```

### 4. UMM Analysis Skill (`umm-analysis-SKILL.md`)

**When to use**: Interpreting results, understanding policy behavior, optimizing protocols.

**Key features**:
- Fidelity trajectory analysis (convergence metrics, stability)
- Back-action quantification (efficiency, strength distribution)
- Information gain analysis (entropy reduction, Fisher information)
- Policy interpretability (measurement axes, adaptation patterns)
- Noise sensitivity analysis (robustness testing)
- Optimization strategies (hyperparameter tuning, multi-objective, GRAPE)

**Analysis workflows**:
```python
# Convergence analysis
metrics = analyze_fidelity_convergence(trajectories)
# Returns: convergence_time, stability, monotonicity, etc.

# Back-action analysis  
ba_metrics = analyze_backaction(histories, target_state)
# Returns: mean_backaction, efficiency, strength patterns

# Noise sensitivity
sensitivity = analyze_noise_sensitivity(policy, param_ranges)
# Returns: performance across T1/T2/error ranges
```

### 5. UMM Skills Guide (`UMM-SKILLS-GUIDE.md`)

**When to use**: Understanding how skills work together, planning projects.

**Key features**:
- Complete workflow from task design to optimization
- Skill integration patterns (rapid prototyping, research publication, hardware deployment)
- Cross-skill scenarios (debugging, reproducibility, overfitting)
- Decision trees for skill selection
- Best practices for multi-skill projects
- Custom workflow orchestration

## File Specifications

| Skill | Lines | Size | Key Sections |
|-------|-------|------|--------------|
| Task Design | ~600 | 12KB | Task Types, Cost Functions, Constraints, Validation |
| Experiment | ~900 | 18KB | Patterns, Statistics, Visualization, Reproducibility |
| Policy Training | ~1200 | 24KB | Architectures, Algorithms, Training, Debugging |
| Analysis | ~1300 | 26KB | Fidelity, Back-action, Information, Optimization |
| Skills Guide | ~900 | 18KB | Workflow, Integration, Patterns, Quick Reference |

## Skill Invariants

All skills follow **spec invariance** principles:

### Task Design Invariants
- ✓ Measurement strengths m ∈ [0, 1]
- ✓ Time budgets positive and finite
- ✓ Cost weights non-negative
- ✓ Target states normalized
- ✓ Measurement axes unit vectors

### Experiment Invariants
- ✓ n_runs ≥ 30 for statistics
- ✓ Noise parameters physical
- ✓ Random seeds set
- ✓ Metrics well-defined
- ✓ Baselines use identical setup

### Policy Training Invariants
- ✓ State space well-defined
- ✓ Action space feasible
- ✓ Reward function bounded
- ✓ Episode termination clear
- ✓ Training matches deployment
- ✓ Domain randomization covers variations
- ✓ Safety via projection

### Analysis Invariants
- ✓ Complete episodes only
- ✓ Consistent metrics
- ✓ Multiple comparison correction
- ✓ Physical units and bounds
- ✓ Appropriate visualization scales
- ✓ Finite-sample corrections
- ✓ Consistent information bases

## Integration Workflows

### Workflow 1: Rapid Prototyping

```
Task Design (simple objective)
    ↓
Experiment (parametric baseline)
    ↓
Analysis (check feasibility)
    ↓
Decision: Train or iterate
```

**Timeline**: 1-2 days

### Workflow 2: Research Publication

```
Task Design (formal objective)
    ↓
Policy Training (with domain randomization)
    ↓
Experiment (comprehensive benchmarks, n≥100)
    ↓
Analysis (statistics + visualizations)
    ↓
Experiment (reproduce with new seeds)
    ↓
Paper-ready results
```

**Timeline**: 2-4 weeks

### Workflow 3: Hardware Deployment

```
Task Design (hardware constraints)
    ↓
Experiment (validate in simulation)
    ↓
Policy Training (realistic noise models)
    ↓
Analysis (sensitivity to calibration drift)
    ↓
Optimization (fine-tune parameters)
    ↓
Export pulse sequences
```

**Timeline**: 4-8 weeks

## Common Use Cases

### Use Case 1: State Preparation Optimization

**Goal**: Prepare |+x⟩ with F > 0.95 under dephasing.

**Skills used**: Task Design → Policy Training → Experiment → Analysis

**Expected outcome**: Adaptive policy achieving 2-5% higher fidelity than fixed baseline.

### Use Case 2: Quantum Zeno Stabilization

**Goal**: Suppress dephasing in target subspace.

**Skills used**: Task Design → Policy Training (LSTM) → Experiment → Analysis

**Expected outcome**: Learned measurement schedule extending coherence time by 2-3×.

### Use Case 3: Adaptive Phase Estimation

**Goal**: Estimate unknown phase with Heisenberg scaling.

**Skills used**: Task Design → Policy Training (Transformer) → Experiment → Analysis

**Expected outcome**: QFI approaching N² scaling vs N for non-adaptive.

### Use Case 4: Noise Robustness Study

**Goal**: Characterize performance across T1/T2 parameter space.

**Skills used**: Task Design → Experiment (sweeps) → Analysis (sensitivity)

**Expected outcome**: Heatmap identifying operating regimes and failure modes.

## Directory Structure

Recommended project organization:

```
umm-project/
├── skills/                     # This package
│   ├── umm-task-design-SKILL.md
│   ├── umm-experiment-SKILL.md
│   ├── umm-policy-training-SKILL.md
│   ├── umm-analysis-SKILL.md
│   └── UMM-SKILLS-GUIDE.md
├── objectives/                 # Task specifications
│   └── my_task.json
├── policies/                   # Trained models
│   └── transformer_state_prep.pt
├── experiments/               # Experiment scripts
│   └── my_experiment.py
├── results/                   # Data
│   ├── trajectories/
│   ├── manifests/
│   └── figures/
└── analysis/                  # Reports
    └── comprehensive_analysis.md
```

## Performance Expectations

Based on reference implementation benchmarks:

| Task | Architecture | Episodes | Final Performance | Training Time |
|------|-------------|----------|-------------------|---------------|
| State Prep (1Q) | Transformer | 20k | F > 0.96 | ~4 hours |
| State Prep (1Q) | LSTM | 30k | F > 0.95 | ~5 hours |
| Zeno (1Q) | LSTM | 50k | P_survive > 0.90 | ~8 hours |
| Metrology (1Q) | Transformer | 100k | Near Heisenberg | ~15 hours |
| LOCC (2Q) | GNN | 200k | C > 0.8 | ~30 hours |

*Times on GPU (NVIDIA A100). CPU training 5-10× slower.*

## Skill Synergies

Skills are designed to work together:

1. **Task Design + Experiment**: Valid objectives → reproducible experiments
2. **Task Design + Policy Training**: Clear objectives → efficient learning
3. **Policy Training + Analysis**: Trained policies → interpretable behavior
4. **Experiment + Analysis**: Statistical data → physical insights
5. **All skills**: Complete pipeline from design to deployment

## Best Practices

### Documentation
- Use skill-specific validation checklists
- Create manifest for each experiment
- Version control objectives and configs
- Document all design decisions

### Iteration
- Start with simple objectives (Task Design)
- Establish baselines (Experiment)
- Train incrementally (Policy Training)
- Analyze continuously (Analysis)

### Quality
- Follow invariant specifications
- Apply statistical rigor (n ≥ 30)
- Visualize comprehensively
- Test robustness systematically

### Collaboration
- Share objectives in JSON format
- Include reproducibility manifests
- Provide trained policy checkpoints
- Document analysis methods

## Troubleshooting

### Skill-Specific Issues

**Task Design**: Objective doesn't compile
- ✓ Check all invariants satisfied
- ✓ Validate JSON schema
- ✓ Test with simple examples first

**Experiment**: Results not significant
- ✓ Increase n_runs (≥ 100)
- ✓ Check random seed setting
- ✓ Verify baseline is fair comparison

**Policy Training**: Training not converging
- ✓ Verify reward function informative
- ✓ Check state representation
- ✓ Add reward shaping
- ✓ Reduce learning rate

**Analysis**: Unexpected behavior
- ✓ Validate input data complete
- ✓ Check for outliers
- ✓ Verify metric definitions
- ✓ Test on simple cases first

### Cross-Skill Issues

**Policy overfits to training noise**
- Solution: Increase domain randomization (Policy Training)
- Validate: Run sensitivity analysis (Analysis)
- Document: Record parameter ranges (Experiment)

**Results not reproducible**
- Solution: Set random seeds (Experiment)
- Check: Numerical stability (Analysis)
- Verify: Constraint enforcement (Task Design)

## Citation

If you use these skills in your research:

```bibtex
@software{umm_skills2025,
  title={UMM Skills: AI-Powered Quantum Measurement Design},
  author={Hart, Justin},
  year={2025},
  publisher={Viridis LLC},
  url={https://github.com/viridis-llc/umm-project}
}
```

## Related Documentation

- [UMM Architecture](../umm-project/docs/architecture.md)
- [API Reference](../umm-project/docs/api_reference.md)
- [Experimental Guide](../umm-project/docs/experimental_guide.md)
- [UMM-IR Specification](../umm-project/docs/umm_ir_spec.md)

## Support

For questions or issues:
- Email: justin@viridis.llc
- GitHub: [viridis-llc/umm-project](https://github.com/viridis-llc/umm-project)
- Website: https://measurement-machine.ai

## Version History

- **v1.0** (2025-10-26): Initial release
  - Task Design skill
  - Experiment skill
  - Policy Training skill
  - Analysis skill
  - Skills Guide

## License

MIT License - See [LICENSE](../LICENSE) for details.

---

**Ready to design the future of quantum measurement?** Start with the [UMM Skills Guide](UMM-SKILLS-GUIDE.md) to understand the complete workflow, then dive into individual skills as needed.
