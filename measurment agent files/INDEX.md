# UMM Skills Package - Master Index

**Version**: 1.0  
**Date**: October 26, 2025  
**Author**: Justin Hart, Viridis LLC  
**Total Size**: ~121KB (7 documents)

## Package Contents

### Core Skills (4 documents, ~80KB)

1. **umm-task-design-SKILL.md** (12KB, ~600 lines)
   - Task formulation for state prep, Zeno, entanglement, metrology
   - Constraint specification and cost function design
   - Validation checklists and common pitfalls
   - Multi-objective optimization patterns

2. **umm-experiment-SKILL.md** (18KB, ~900 lines)
   - Experimental design and statistical analysis
   - Reproducibility standards and manifests
   - Visualization guidelines
   - Performance benchmarking patterns

3. **umm-policy-training-SKILL.md** (24KB, ~1200 lines)
   - Neural network architectures (Transformer, LSTM, GNN)
   - RL algorithms (PPO, REINFORCE, SAC)
   - Domain randomization and robustness
   - Training debugging and optimization

4. **umm-analysis-SKILL.md** (26KB, ~1300 lines)
   - Fidelity trajectory and convergence analysis
   - Back-action quantification
   - Information gain and Fisher information
   - Policy interpretability and optimization

### Integration Documents (3 documents, ~41KB)

5. **UMM-SKILLS-GUIDE.md** (18KB, ~900 lines)
   - Complete workflow from design to deployment
   - Integration patterns (prototyping, publication, hardware)
   - Cross-skill problem solving
   - Custom workflow orchestration

6. **README.md** (14KB, ~700 lines)
   - Package overview and quick start
   - Performance expectations and benchmarks
   - Troubleshooting guide
   - Best practices and citations

7. **VISUAL-WORKFLOW.md** (10KB, ~500 lines)
   - Mermaid diagrams of skill relationships
   - Decision trees for skill selection
   - Workflow patterns at different complexity levels
   - Feature matrices and navigation guides

## Quick Reference Table

| Need | Skill | Section | Time |
|------|-------|---------|------|
| Define task | Task Design | Task Types | 30 min |
| Set constraints | Task Design | Constraints | 15 min |
| Design costs | Task Design | Cost Functions | 20 min |
| Run baseline | Experiment | Pattern 1 | 2 hours |
| Statistical test | Experiment | Statistics | 1 hour |
| Reproduce paper | Experiment | Reproduction | Variable |
| Choose architecture | Policy Training | Architectures | 30 min |
| Train policy | Policy Training | Algorithms | 4-30 hours |
| Debug training | Policy Training | Debugging | 1-4 hours |
| Analyze convergence | Analysis | Fidelity Analysis | 30 min |
| Measure back-action | Analysis | Back-action | 20 min |
| Test robustness | Analysis | Noise Sensitivity | 2 hours |
| Optimize protocol | Analysis | Optimization | 4-8 hours |
| Understand workflow | Skills Guide | Complete Workflow | 1 hour |
| Visual overview | Visual Workflow | All diagrams | 15 min |

## Skill Specifications

### Invariants Summary

**Task Design**:
- m ∈ [0,1], T_budget > 0, λ ≥ 0, ||ψ|| = 1, ||n|| = 1

**Experiment**:
- n_runs ≥ 30, seeds fixed, metrics consistent, noise physical

**Policy Training**:
- State/action well-defined, safety via projection, training = deployment

**Analysis**:
- Complete episodes, consistent metrics, proper statistics

### Key Algorithms

**Task Design**:
- ObjectiveParser.compile()
- Constraint validation
- Cost function generation

**Experiment**:
- run_bloch_steering_experiment()
- analyze_experiment_results()
- create_experiment_manifest()

**Policy Training**:
- train_policy_gradient() - REINFORCE
- train_ppo() - Proximal Policy Optimization
- domain_randomized_training()

**Analysis**:
- analyze_fidelity_convergence()
- analyze_backaction()
- analyze_noise_sensitivity()
- optimize_measurement_protocol()

## Workflow Recipes

### Recipe 1: Quick Validation (1-2 days)
```python
# 1. Simple objective (Task Design)
objective = {"task": "state_prep", "target_state": "|+x>"}

# 2. Baseline experiment (Experiment)  
baseline = run_fixed_baseline(objective, n_runs=30)

# 3. Quick analysis (Analysis)
metrics = analyze_fidelity_convergence(baseline.trajectories)

# Decision: proceed or iterate
```

### Recipe 2: Full Research Pipeline (2-4 weeks)
```python
# 1. Formal objective (Task Design)
objective = formalize_objective_with_theory(...)

# 2. Train policy (Policy Training)
policy = train_umm_policy(objective, n_episodes=50000)

# 3. Comprehensive experiments (Experiment)
results = run_comparative_experiment(policy, n_runs=100)

# 4. Statistical analysis (Analysis)
report = generate_comprehensive_analysis(results)

# 5. Reproduce (Experiment)
validate = reproduce_with_new_seeds(policy, n_trials=100)
```

### Recipe 3: Hardware Deployment (4-8 weeks)
```python
# 1. Hardware-constrained objective (Task Design)
objective = define_with_hardware_limits(T1, T2, gates)

# 2. Validate in simulation (Experiment)
sim_results = validate_with_hardware_noise(objective)

# 3. Train with realistic model (Policy Training)
policy = train_with_noise_model(objective, hardware_params)

# 4. Sensitivity analysis (Analysis)
sensitivity = test_calibration_drift(policy, param_ranges)

# 5. Optimize and export (Analysis)
pulses = optimize_and_compile_for_hardware(policy)
```

## Performance Benchmarks

| Task | Method | Expected Performance | Training Time |
|------|--------|---------------------|---------------|
| State Prep 1Q | Transformer | F > 0.96 | 4h (GPU) |
| State Prep 1Q | LSTM | F > 0.95 | 5h (GPU) |
| Zeno 1Q | LSTM | P > 0.90 | 8h (GPU) |
| Metrology 1Q | Transformer | Near HL | 15h (GPU) |
| LOCC 2Q | GNN | C > 0.8 | 30h (GPU) |

*GPU: NVIDIA A100. CPU times 5-10× longer.*

## Usage Patterns

### For Researchers
1. Read: Task Design, Experiment, Analysis
2. Use: Full research pipeline recipe
3. Focus: Statistical rigor, reproducibility
4. Output: Paper-ready results

### For Engineers
1. Read: Skills Guide, Policy Training
2. Use: Hardware deployment recipe
3. Focus: Robustness, optimization
4. Output: Deployment-ready protocols

### For Students
1. Read: README, Visual Workflow, Task Design
2. Use: Quick validation recipe
3. Focus: Understanding fundamentals
4. Output: Working prototypes

### For Collaborators
1. Share: objective.json, config.json, manifest.json
2. Use: Standardized formats from Experiment skill
3. Focus: Reproducibility
4. Output: Comparable results

## Integration with UMM Codebase

Skills map directly to code modules:

```
Skills                    UMM Code
-----------              -----------
Task Design       ↔      umm.intent.parser
Experiment        ↔      experiments/*.py
Policy Training   ↔      umm.policy.*
Analysis          ↔      Custom analysis scripts
```

## File Relationships

```
README.md ─────────────┐
                       ├──→ Quick Start
UMM-SKILLS-GUIDE.md ───┘    Complete Workflow

VISUAL-WORKFLOW.md ────→    Visual Overview

umm-task-design-SKILL.md ──→ Objective Formulation
          ↓
umm-experiment-SKILL.md ────→ Experimental Validation
          ↓
umm-policy-training-SKILL.md → Policy Development
          ↓
umm-analysis-SKILL.md ──────→ Results Interpretation
          ↓
        Optimization
```

## Key Concepts

### Spec Invariance
All skills enforce invariants before execution:
- Physical constraints (m ∈ [0,1], T > 0)
- Statistical requirements (n ≥ 30)
- Safety guarantees (projection, not learning)

### Domain Randomization
Train with parameter variations:
- T1: (30µs, 100µs)
- T2: (30µs, 100µs)
- Errors: (0.001, 0.05)

### Multi-Objective Optimization
Balance competing objectives:
- Maximize fidelity
- Minimize back-action
- Minimize time
- Maximize robustness

### Reproducibility
Every experiment includes:
- Random seed
- Noise parameters
- Configuration
- Manifest
- Code hash

## Common Scenarios

### Scenario: Training Fails
**Path**: Policy Training (Debugging) → Task Design (Verify Reward) → Analysis (Profile)

### Scenario: Results Vary
**Path**: Experiment (Seeds) → Analysis (Check Stability) → Task Design (Constraints)

### Scenario: Need Optimization
**Path**: Analysis (Sensitivity) → Analysis (Optimization) → Experiment (Validate)

### Scenario: Deploy to Hardware
**Path**: Task Design (HW Constraints) → Policy Training (Realistic Noise) → Analysis (Calibration) → Export

## Citation Format

```bibtex
@software{umm_skills2025,
  title={UMM Skills: AI-Powered Quantum Measurement Design},
  author={Hart, Justin},
  year={2025},
  version={1.0},
  publisher={Viridis LLC},
  url={https://github.com/viridis-llc/umm-project}
}
```

## Next Steps

1. **New users**: Start with [README.md](README.md)
2. **Visual learners**: Check [VISUAL-WORKFLOW.md](VISUAL-WORKFLOW.md)
3. **Full pipeline**: Read [UMM-SKILLS-GUIDE.md](UMM-SKILLS-GUIDE.md)
4. **Specific task**: Jump to relevant skill document

## Support Resources

- **Documentation**: All 7 skill documents
- **Examples**: Embedded in each skill
- **Code**: UMM reference implementation
- **Contact**: justin@viridis.llc

## Version Notes

**v1.0 (2025-10-26)**:
- Initial release
- 4 core skills + 3 integration docs
- ~121KB total documentation
- Complete workflow coverage
- Cross-skill problem solving
- Visual workflow diagrams

---

**The UMM Skills Package provides everything needed to design, train, validate, and deploy adaptive quantum measurement protocols.** Start with the README for an overview, then dive into specific skills as needed for your project phase.
