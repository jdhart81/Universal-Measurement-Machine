# Universal Measurement Machine (UMM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**AI-Programmed Adaptive Quantum Measurement**

> Operational wavefunction collapse as user-extended action

## Overview

The Universal Measurement Machine (UMM) is a framework for compiling user objectives into adaptive quantum measurement sequences. By treating measurement design (basis, strength, timing, adaptivity) as a programmable control variable, UMM enables:

- **Adaptive state preparation** with minimal back-action
- **Quantum Zeno stabilization** with learned schedules
- **LOCC entanglement concentration**
- **Adaptive quantum metrology** for phase estimation
- **Measurement-based quantum control**

This repository contains the reference implementation accompanying the paper:
> Justin Hart, "AI as a Universal Measurement Machine: Operational Wavefunction Collapse as User-Extended Action", 2025

## Key Features

- 🎯 **Intent Parser**: Compile natural objectives (DSL or JSON) into reward functions
- 🔧 **UMM-IR**: Auditable intermediate representation for measurement sequences
- 🤖 **Adaptive Policies**: Transformer, LSTM, and GNN architectures with RL training
- 🛡️ **Safety**: Convex projection onto feasible action sets with tamper-evident logs
- 📊 **Calibration**: Classical shadows and canary POVMs for drift tracking
- ⚡ **Simulator**: Fast quantum state evolution with realistic noise models

## Quick Start

### Installation

```bash
git clone https://github.com/viridis-llc/umm-project.git
cd umm-project
pip install -e .
```

### Basic Example

```python
from umm.intent import ObjectiveParser
from umm.core import UMMSimulator
from umm.policy import TransformerPolicy

# Define objective
objective = {
    "task": "state_prep",
    "target_state": "|+x>",
    "constraints": {"budget_time_us": 10.0},
    "cost_weights": {"backaction": 0.5, "time": 0.2}
}

# Parse and compile
parser = ObjectiveParser()
reward_fn, constraints = parser.compile(objective)

# Initialize simulator and policy
simulator = UMMSimulator(n_qubits=1, T1=50e-6, T2=50e-6)
policy = TransformerPolicy.load_pretrained("state_prep_qubit")

# Run adaptive measurement sequence
result = simulator.run_adaptive(
    policy=policy,
    reward_fn=reward_fn,
    constraints=constraints,
    max_steps=32
)

print(f"Final fidelity: {result.fidelity:.4f}")
print(f"Back-action cost: {result.backaction:.4f}")
```

## Reproducing Paper Results

All experiments from the paper can be reproduced:

```bash
# Bloch steering (state preparation)
python experiments/bloch_steering.py --config configs/state_prep_example.json

# Quantum Zeno stabilization
python experiments/zeno_stabilization.py --config configs/zeno_example.json

# Adaptive metrology
python experiments/adaptive_metrology.py --config configs/metrology_example.json

# Superconducting qubit demo (Section 7)
python experiments/superconducting_demo.py --config configs/superconducting_params.yaml
```

## UMM-IR Examples

The intermediate representation makes measurement sequences auditable:

```
RESET(rho_init)
FOR t in 1..T:
  WEAK_MEAS(axis=[1,0,0], strength=0.3)
  k_t = MEASURE_ANCILLA({Pi_+, Pi_-}, thresh=0.5)
  IF k_t == - THEN APPLY_U(R_x(0.1))
  WAIT(1e-6)
  SAFETY_ENFORCE(C)
```

## Architecture

```
User Objective (DSL/JSON)
    ↓
Intent Parser Γ
    ↓
(Reward R, Constraints C)
    ↓
Policy π_θ (Transformer/LSTM/GNN)
    ↓
UMM-IR Actions
    ↓
Safety Projection onto C
    ↓
Compiler → Device Gates
    ↓
Quantum Instrument {M_k}
    ↓
Outcome k, State Update
```

## Training Your Own Policies

```python
from umm.policy import TransformerPolicy
from umm.policy.training import train_policy

# Define custom objective
objective_spec = {...}

# Train with domain randomization
policy = train_policy(
    objective_spec=objective_spec,
    policy_class=TransformerPolicy,
    n_episodes=50000,
    warm_start_from="projective_baseline",
    domain_randomization={"T1": (30e-6, 100e-6), "T2": (30e-6, 100e-6)}
)

# Save trained policy
policy.save("my_custom_policy.pt")
```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Experimental Guide](docs/experimental_guide.md)
- [UMM-IR Specification](docs/umm_ir_spec.md)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hart2025umm,
  title={AI as a Universal Measurement Machine: Operational Wavefunction Collapse as User-Extended Action},
  author={Hart, Justin},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

- Justin Hart - Viridis LLC
- Email: justin@viridis.llc
- Website: https://measurement-machine.ai

## Acknowledgments

This work was developed with assistance from Claude (Anthropic) for drafting and code organization.
