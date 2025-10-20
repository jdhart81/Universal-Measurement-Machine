# UMM Architecture

## Overview

The Universal Measurement Machine (UMM) consists of several layered components:

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                         │
│              (DSL, JSON, Python API)                     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  Intent Parser (Γ)                       │
│         Compile Objective → (Reward, Constraints)        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  Policy π_θ                              │
│    (Transformer, LSTM, GNN)                              │
│    History/Belief → Action                               │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              Safety Projection                           │
│         Project onto Constraint Set C                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  UMM-IR                                  │
│    (Auditable Intermediate Representation)               │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  Compiler                                │
│         IR → Device-Level Gates/Pulses                   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              Quantum Hardware / Simulator                │
│    System + Ancilla → Instrument {M_k} → Outcome         │
└─────────────────────────────────────────────────────────┘
```

## Core Modules

### `umm.core`

**quantum_state.py**: Density matrix evolution with noise models (T1, T2, gate errors)

**instruments.py**: POVM effects and quantum instruments with Kraus operators

**ir.py**: UMM-IR instruction set and program representation

**simulator.py**: Orchestrates state evolution, instrument execution, and history tracking

### `umm.intent`

**parser.py**: Compiles DSL/JSON objectives into reward functions and constraint sets

### `umm.policy`

Policy architectures for adaptive measurement selection (to be expanded)

### `umm.safety`

Safety projection onto convex constraint sets and tamper-evident audit logs

### `umm.calibration`

Calibration drift tracking with canary POVMs and classical shadows

## Data Flow

1. **User Input**: Objective specification (DSL or JSON)
2. **Parsing**: Γ compiles objective → (reward_fn, constraints)
3. **Policy Selection**: π_θ(history) → raw_action
4. **Safety**: Project raw_action onto constraint set C
5. **Compilation**: IR instruction → device-level implementation
6. **Execution**: Apply quantum instrument, observe outcome
7. **Update**: State update ρ → ρ', history ← (action, outcome)
8. **Repeat**: Steps 3-7 for adaptive sequence

## Key Design Principles

1. **Auditability**: Every measurement is explicitly represented in UMM-IR
2. **Safety**: Convex constraint projection ensures physical realizability
3. **Flexibility**: Policy-agnostic framework supports any RL algorithm
4. **Composability**: Modular architecture enables easy extension

## Extension Points

- **New Policies**: Implement policy interface in `umm.policy`
- **New Instruments**: Add factory functions in `umm.core.instruments`
- **New Objectives**: Extend `ObjectiveParser` with new task types
- **New Platforms**: Add compiler backends for different hardware
