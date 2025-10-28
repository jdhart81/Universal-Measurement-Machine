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

## Project Status

This project is under active development. See [WIREFRAME.md](WIREFRAME.md) for the complete project architecture and roadmap.

**Phase 1 (Foundation) - Completed:**
- Core quantum state evolution
- Basic quantum instruments (POVMs, weak measurements)
- UMM-IR intermediate representation
- Quantum simulator with realistic noise models
- Intent parser for objective compilation

**Phase 2 (Policy Framework) - In Progress:**
- Policy architectures (Transformer, LSTM, GNN)
- RL training pipeline
- Pre-trained models for common tasks

**Future Phases:**
- Safety projection and audit logging
- Calibration with classical shadows and canary POVMs
- Hardware integration with Qiskit/Cirq
- Advanced multi-qubit policies

## Key Features

**Currently Available:**
- 🎯 **Intent Parser**: Compile natural objectives (DSL or JSON) into reward functions
- 🔧 **UMM-IR**: Auditable intermediate representation for measurement sequences
- ⚡ **Simulator**: Fast quantum state evolution with realistic noise models
- 🔬 **Quantum Instruments**: Weak measurements, projective POVMs, adaptive protocols

**In Development:**
- 🤖 **Adaptive Policies**: Transformer, LSTM, and GNN architectures with RL training
- 🛡️ **Safety**: Convex projection onto feasible action sets with tamper-evident logs
- 📊 **Calibration**: Classical shadows and canary POVMs for drift tracking

## Quick Start

### Installation

```bash
git clone https://github.com/jdhart81/Universal-Measurement-Machine.git
cd Universal-Measurement-Machine/umm-project
pip install -e .
```

### Basic Example

```python
from umm.intent import ObjectiveParser
from umm.core import UMMSimulator
# from umm.policy import TransformerPolicy  # Coming soon!

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

# Initialize simulator
simulator = UMMSimulator(n_qubits=1, T1=50e-6, T2=50e-6)

# Current: Manual measurement sequence design
# Future: policy = TransformerPolicy.load_pretrained("state_prep_qubit")
# Future: result = simulator.run_adaptive(policy, reward_fn, constraints)

# Run basic simulation with defined measurement protocol
result = simulator.simulate_protocol(reward_fn=reward_fn, constraints=constraints)

print(f"Final fidelity: {result.fidelity:.4f}")
print(f"Back-action cost: {result.backaction:.4f}")
```

## Repository Structure

```
Universal-Measurement-Machine/
├── README.md                 # This file
├── WIREFRAME.md             # Complete project architecture
├── LICENSE                  # MIT License
├── measurment agent files/  # AI agent skills for UMM system
│   ├── UMM-SKILLS-GUIDE.md
│   ├── umm-analysis-SKILL.md
│   ├── umm-task-design-SKILL.md
│   ├── umm-policy-training-SKILL.md
│   └── umm-experiment-SKILL.md
└── umm-project/            # Main implementation
    ├── umm/                # Core package
    │   ├── core/          # Quantum state, instruments, simulator
    │   ├── intent/        # Objective parser
    │   ├── policy/        # RL policies (in development)
    │   ├── safety/        # Safety mechanisms (planned)
    │   └── calibration/   # Drift tracking (planned)
    ├── experiments/       # Reproducible experiments
    ├── configs/          # Configuration files
    ├── docs/             # Documentation
    └── tests/            # Test suite
```

## Experiments

**Currently Available:**
```bash
cd Universal-Measurement-Machine/umm-project

# Bloch steering (state preparation) - basic implementation
python experiments/bloch_steering.py --config configs/state_prep_example.json
```

**Coming Soon:**
- Quantum Zeno stabilization experiments
- Adaptive metrology protocols
- Entanglement concentration demos
- Superconducting qubit integration

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

## Training Your Own Policies (Coming Soon)

The policy training framework is under development. The planned API will be:

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

See [WIREFRAME.md](WIREFRAME.md#phase-2-policy-framework-weeks-5-8) for the complete policy architecture roadmap.

## Documentation

**Available:**
- [Project Wireframe](WIREFRAME.md) - Complete architecture and roadmap
- [Architecture Overview](umm-project/docs/architecture.md)
- [Measurement Agent Skills Guide](measurment%20agent%20files/UMM-SKILLS-GUIDE.md)

**Planned:**
- API Reference
- Experimental Guide
- UMM-IR Specification

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

We welcome contributions! Please see [CONTRIBUTING.md](umm-project/CONTRIBUTING.md) for guidelines.

For questions or discussions:
- Open an issue on GitHub
- See the [Project Wireframe](WIREFRAME.md) for architecture details
- Check the [Measurement Agent Skills](measurment%20agent%20files/) for AI-assisted development workflows

## Contact

- Justin Hart - Viridis LLC
- Email: justin@viridis.llc
- Website: https://measurement-machine.ai

## Acknowledgments

This work was developed with assistance from Claude (Anthropic) for drafting and code organization.
