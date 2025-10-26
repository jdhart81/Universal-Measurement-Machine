# UMM Task Design Skill

## Purpose
This skill provides best practices for formulating quantum measurement objectives using the Universal Measurement Machine (UMM) framework. Use this skill when users want to design adaptive measurement tasks, define objectives, or create custom measurement strategies.

## Core Principles

### 1. Objective Formulation
Every UMM task must specify:
- **Task type**: state_prep, entanglement_max, phase_est, or zeno
- **Target/Goal**: The desired quantum state or property to achieve
- **Constraints**: Physical limitations (time budget, measurement rate, LOCC)
- **Cost weights**: Trade-offs between competing objectives (backaction, time, probe count)

### 2. Spec Invariants for UMM Tasks

Before implementing any UMM objective, restate these invariants:
- **INVARIANT 1**: All measurement strengths m ∈ [0, 1]
- **INVARIANT 2**: Waiting times Δt ≥ 100ns (hardware constraint)
- **INVARIANT 3**: Total time budget must be positive and finite
- **INVARIANT 4**: Cost weights must be non-negative and sum to reasonable values
- **INVARIANT 5**: Target states must be normalized (||ψ|| = 1)
- **INVARIANT 6**: Measurement axes must be unit vectors (||n|| = 1)

## Task Types and Best Practices

### State Preparation (state_prep)

**Objective**: Prepare target quantum state |ψ⟩ from unknown mixed state with minimal back-action.

**DSL Format**:
```
STATE_PREP |target> WITH_COST WEIGHTED λ_BA*BACKACTION + λ_time*TIME BUDGET TIME T_max
```

**JSON Format**:
```json
{
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
```

**Best Practices**:
1. Start with weak measurements (m ≈ 0.3) for unknown initial states
2. Use higher weights on back-action (λ_BA ≈ 0.5) for high-fidelity preparation
3. For noisy systems (short T1/T2), increase time weight to favor faster sequences
4. Target states should be eigenstates of measurable operators when possible
5. Budget time should be 5-10× T1 for convergence

**Common Targets**:
- `|+x>`: (|0⟩ + |1⟩)/√2 - Requires X-axis measurements
- `|-x>`: (|0⟩ - |1⟩)/√2 - Requires X-axis measurements  
- `|+y>`: (|0⟩ + i|1⟩)/√2 - Requires Y-axis measurements
- `|0>`: Ground state - Simple Z-axis measurements
- `|1>`: Excited state - Z-axis with feedback

### Quantum Zeno Stabilization (zeno)

**Objective**: Stabilize quantum state in target subspace via frequent measurements.

**DSL Format**:
```
ZENO_STABILIZE |ψ> WITH_SCHEDULE ADAPTIVE BUDGET TIME T_total
```

**JSON Format**:
```json
{
  "task": "zeno",
  "target_subspace": "|+x>",
  "constraints": {
    "budget_time_us": 50.0,
    "min_interval_ns": 100
  },
  "cost_weights": {
    "survival_prob": 1.0,
    "measurement_count": 0.01
  }
}
```

**Best Practices**:
1. Start with projective measurements (m = 1.0) for strong Zeno effect
2. Adapt measurement rate based on decoherence: Γ_meas ≈ 10 × Γ_decohere
3. Use weak measurements (m < 0.5) near end of protocol for gentle monitoring
4. Monitor survival probability - if dropping below 0.8, increase measurement rate
5. Budget time should be comparable to or shorter than T2 for effectiveness

**Scheduling Strategies**:
- **Fixed rate**: Constant Δt between measurements - simple but suboptimal
- **Exponential**: Δt_n = Δt_0 × exp(n/N) - dense early, sparse late
- **Adaptive**: Policy learns optimal schedule - requires training

### Entanglement Concentration (entanglement_max)

**Objective**: Maximize entanglement monotone (concurrence, negativity) via LOCC.

**DSL Format**:
```
ENTANGLEMENT_MAX concurrence UNDER LOCC BUDGET ROUNDS R_max
```

**JSON Format**:
```json
{
  "task": "entanglement_max",
  "monotone": "concurrence",
  "constraints": {
    "locc": true,
    "budget_rounds": 10,
    "two_qubit_gates": false
  },
  "cost_weights": {
    "entanglement": 1.0,
    "success_prob": 0.3
  }
}
```

**Best Practices**:
1. Restrict to LOCC operations only (no two-qubit gates)
2. Measure in Bell basis when possible
3. Post-select on favorable outcomes to concentrate entanglement
4. Track success probability - typical protocols achieve 0.5-0.8
5. Use concurrence for two-qubit systems, negativity for multi-partite
6. Budget 5-20 rounds depending on initial entanglement

**Monotone Selection**:
- **Concurrence**: Two-qubit systems, range [0,1], convex roof construction
- **Negativity**: Multi-partite systems, easier to compute, range [0, (d-1)/d]

### Adaptive Metrology (phase_est)

**Objective**: Estimate unknown phase φ with optimal quantum Fisher information.

**DSL Format**:
```
PHASE_EST φ WITH_CRITERION QFI_MAX BUDGET PROBES N_max
```

**JSON Format**:
```json
{
  "task": "phase_est",
  "parameter": "phi",
  "constraints": {
    "budget_probes": 100,
    "parallel_probes": 1
  },
  "cost_weights": {
    "qfi": 1.0,
    "probe_count": 0.01
  }
}
```

**Best Practices**:
1. Use weak measurements to avoid resetting quantum state
2. Adapt measurement basis based on Bayesian posterior
3. QFI scales as N² for entangled probes vs N for separable
4. Start with broad measurement basis, narrow as you localize φ
5. Budget probes should be 10-100× desired precision level
6. Use parallel probes when available (multiplicative speedup)

**Adaptive Strategies**:
- **Bayesian**: Update posterior after each measurement, measure in direction of maximum information gain
- **Covariant**: Rotate measurement basis by estimated phase after each round
- **Two-stage**: Coarse estimation followed by fine-scale adaptive

## Cost Function Design

### Back-action Cost
Quantifies measurement disturbance:
```
C_BA = Σ_t m_t
```
where m_t ∈ [0,1] is measurement strength at time t.

**Typical weights**: λ_BA ∈ [0.3, 0.7]
- High (0.7): Fragile quantum states, require gentle monitoring
- Low (0.3): Robust states or when speed is critical

### Time Cost
Penalizes long protocols:
```
C_time = T_total / T_budget
```

**Typical weights**: λ_time ∈ [0.1, 0.3]
- High (0.3): Noisy systems where decoherence dominates
- Low (0.1): High-coherence systems where precision matters more

### Probe Cost
Counts number of measurements:
```
C_probe = N_measurements / N_budget
```

**Typical weights**: λ_probe ∈ [0.01, 0.1]
- High (0.1): Limited measurement resources
- Low (0.01): Measurement is cheap, focus on quality

## Constraint Specification

### Time Constraints
```json
{
  "budget_time_us": 10.0,        // Total allowed time
  "min_wait_ns": 100,            // Hardware minimum idle time
  "max_sequence_length": 32      // Maximum number of steps
}
```

### Measurement Constraints
```json
{
  "max_strength": 0.8,           // Maximum measurement strength
  "max_rate_MHz": 10.0,          // Maximum measurement rate
  "allowed_bases": ["X", "Y"]    // Restrict measurement axes
}
```

### Resource Constraints
```json
{
  "budget_probes": 100,          // Maximum measurements
  "budget_ancilla": 2,           // Available ancilla qubits
  "locc": true                   // Restrict to LOCC operations
}
```

## Advanced Patterns

### Multi-Objective Optimization
When optimizing multiple competing objectives:

```json
{
  "task": "state_prep",
  "target_state": "|+x>",
  "objectives": [
    {"name": "fidelity", "weight": 1.0, "type": "maximize"},
    {"name": "backaction", "weight": 0.5, "type": "minimize"},
    {"name": "time", "weight": 0.2, "type": "minimize"},
    {"name": "purity", "weight": 0.3, "type": "maximize"}
  ],
  "constraints": {
    "budget_time_us": 10.0,
    "min_fidelity": 0.95
  }
}
```

### Conditional Objectives
Objectives that adapt based on intermediate results:

```json
{
  "task": "state_prep",
  "target_state": "|+x>",
  "conditional_strategy": {
    "if_fidelity_above": 0.9,
    "then_switch_to": {
      "task": "zeno",
      "target_subspace": "|+x>"
    }
  }
}
```

### Robust Objectives
Account for parameter uncertainty:

```json
{
  "task": "state_prep",
  "target_state": "|+x>",
  "robustness": {
    "T1_range": [30e-6, 100e-6],
    "T2_range": [30e-6, 100e-6],
    "optimize_for": "worst_case"  // or "average_case"
  }
}
```

## Validation Checklist

Before running any UMM task, verify:

- [ ] Task type is supported (state_prep, zeno, entanglement_max, phase_est)
- [ ] Target state/objective is well-defined and normalized
- [ ] All measurement strengths in [0, 1]
- [ ] Time budget is positive and > 100ns
- [ ] Cost weights are non-negative
- [ ] Constraints are physically realizable
- [ ] Measurement axes are unit vectors
- [ ] For LOCC tasks: no two-qubit gates specified
- [ ] For multi-objective: weights sum to reasonable total (typically 1-3)

## Common Pitfalls and Solutions

### Pitfall 1: Unrealistic Time Budgets
**Problem**: Setting time budget << T2 gives no time for convergence
**Solution**: Use T_budget ≥ 5×T2 for state prep, ≥ 10×T2 for metrology

### Pitfall 2: Conflicting Constraints
**Problem**: High-fidelity target with minimal back-action and short time
**Solution**: Prioritize objectives - can't optimize all three simultaneously

### Pitfall 3: Wrong Measurement Basis
**Problem**: Measuring Z-axis to prepare X-eigenstate
**Solution**: Align measurement axis with target state's dominant component

### Pitfall 4: Over-constrained LOCC
**Problem**: Requiring entanglement concentration without allowing post-selection
**Solution**: Enable success probability cost and allow probabilistic protocols

### Pitfall 5: Ignoring Noise
**Problem**: Using same parameters for T1=50µs and T1=5µs systems
**Solution**: Scale measurement rate and time budget with decoherence rates

## Examples

### Example 1: High-Fidelity State Prep
```python
objective = {
    "task": "state_prep",
    "target_state": "|+x>",
    "constraints": {
        "budget_time_us": 10.0,
        "max_strength": 0.5  # Gentle measurements only
    },
    "cost_weights": {
        "backaction": 0.7,  # High - prioritize gentleness
        "time": 0.1         # Low - precision over speed
    }
}
```

### Example 2: Fast State Prep (Noisy Environment)
```python
objective = {
    "task": "state_prep",
    "target_state": "|+x>",
    "constraints": {
        "budget_time_us": 3.0,  # Short - before decoherence
        "max_strength": 0.9     # Strong measurements allowed
    },
    "cost_weights": {
        "backaction": 0.2,  # Low - speed is priority
        "time": 0.5         # High - must finish quickly
    }
}
```

### Example 3: Adaptive Phase Estimation
```python
objective = {
    "task": "phase_est",
    "parameter": "phi",
    "constraints": {
        "budget_probes": 50,
        "adaptive_basis": true
    },
    "cost_weights": {
        "qfi": 1.0,
        "probe_count": 0.02
    },
    "target_precision": 0.01  # Estimate φ to within 0.01 radians
}
```

## Integration with UMM Components

### With Intent Parser
The parser validates objectives and compiles them to reward functions:
```python
from umm.intent import ObjectiveParser

parser = ObjectiveParser()
reward_fn, constraints = parser.compile(objective)
```

### With Simulator
The simulator executes measurement sequences:
```python
from umm.core import UMMSimulator

simulator = UMMSimulator(n_qubits=1, T1=50e-6, T2=50e-6)
result = simulator.run_adaptive(
    policy=policy,
    reward_fn=reward_fn,
    constraints=constraints,
    max_steps=32
)
```

### With Policy Training
Policies learn from objectives:
```python
from umm.policy.training import train_policy

policy = train_policy(
    objective_spec=objective,
    n_episodes=50000,
    domain_randomization={"T1": (30e-6, 100e-6)}
)
```

## Summary

When designing UMM tasks:
1. **Start simple**: Use template objectives before customizing
2. **Validate invariants**: Check all constraints before running
3. **Scale with noise**: Adjust budgets and weights for T1/T2
4. **Trade-offs explicit**: Make cost weights reflect true priorities
5. **Iterate**: Run quick tests before expensive policy training

The UMM framework succeeds when objectives are precisely specified with realistic constraints.
