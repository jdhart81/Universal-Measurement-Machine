# UMM Skills Ecosystem - Visual Workflow

This document provides visual representations of how the UMM skills work together.

## Complete Workflow Diagram

```mermaid
graph TD
    A[User Objective] --> B[Task Design Skill]
    B --> C{Valid Objective?}
    C -->|No| B
    C -->|Yes| D[Compiled Objective]
    
    D --> E[Experiment Skill:<br/>Baseline]
    E --> F[Baseline Metrics]
    
    D --> G[Policy Training Skill]
    G --> H{Converged?}
    H -->|No| I[Adjust Training]
    I --> G
    H -->|Yes| J[Trained Policy]
    
    J --> K[Experiment Skill:<br/>Validation]
    F --> K
    K --> L[Experimental Data]
    
    L --> M[Analysis Skill]
    M --> N{Performance<br/>Acceptable?}
    N -->|No| O{Root Cause?}
    O -->|Task Design| B
    O -->|Training| G
    O -->|Experiment| K
    N -->|Yes| P[Analysis Skill:<br/>Optimization]
    
    P --> Q[Optimized Protocol]
    Q --> R[Deployment]
    
    style B fill:#e1f5ff
    style E fill:#ffe1f5
    style G fill:#e1ffe1
    style K fill:#ffe1f5
    style M fill:#fff5e1
    style P fill:#fff5e1
```

## Skill Dependencies

```mermaid
graph LR
    TD[Task Design<br/>Skill] --> EX[Experiment<br/>Skill]
    TD --> PT[Policy Training<br/>Skill]
    PT --> EX
    EX --> AN[Analysis<br/>Skill]
    PT --> AN
    AN --> OPT[Optimization]
    OPT -.->|Iterate| TD
    
    style TD fill:#e1f5ff
    style EX fill:#ffe1f5
    style PT fill:#e1ffe1
    style AN fill:#fff5e1
```

## Decision Tree for Skill Selection

```mermaid
graph TD
    START[What do you need?] --> Q1{Define<br/>task?}
    Q1 -->|Yes| TASK[Task Design<br/>Skill]
    
    Q1 -->|No| Q2{Run<br/>experiments?}
    Q2 -->|Yes| Q3{What type?}
    Q3 -->|Benchmark| EXP1[Experiment Skill:<br/>Statistical Analysis]
    Q3 -->|Parameter Sweep| EXP2[Experiment Skill:<br/>Noise Robustness]
    Q3 -->|Reproduce Paper| EXP3[Experiment Skill:<br/>Reproduction Guide]
    
    Q2 -->|No| Q4{Train<br/>policy?}
    Q4 -->|Yes| Q5{Which aspect?}
    Q5 -->|Architecture| PT1[Policy Training:<br/>Architectures]
    Q5 -->|Algorithm| PT2[Policy Training:<br/>Training Algorithms]
    Q5 -->|Debugging| PT3[Policy Training:<br/>Debugging]
    Q5 -->|Robustness| PT4[Policy Training:<br/>Domain Randomization]
    
    Q4 -->|No| Q6{Analyze<br/>results?}
    Q6 -->|Yes| Q7{What aspect?}
    Q7 -->|Convergence| AN1[Analysis:<br/>Fidelity Analysis]
    Q7 -->|Back-action| AN2[Analysis:<br/>Back-action Analysis]
    Q7 -->|Interpretation| AN3[Analysis:<br/>Policy Behavior]
    Q7 -->|Robustness| AN4[Analysis:<br/>Noise Sensitivity]
    Q7 -->|Optimize| AN5[Analysis:<br/>Optimization]
    
    Q6 -->|No| GUIDE[Read Skills Guide]
    
    style TASK fill:#e1f5ff
    style EXP1 fill:#ffe1f5
    style EXP2 fill:#ffe1f5
    style EXP3 fill:#ffe1f5
    style PT1 fill:#e1ffe1
    style PT2 fill:#e1ffe1
    style PT3 fill:#e1ffe1
    style PT4 fill:#e1ffe1
    style AN1 fill:#fff5e1
    style AN2 fill:#fff5e1
    style AN3 fill:#fff5e1
    style AN4 fill:#fff5e1
    style AN5 fill:#fff5e1
```

## Rapid Prototyping Pattern

```mermaid
sequenceDiagram
    participant U as User
    participant TD as Task Design
    participant EX as Experiment
    participant AN as Analysis
    
    U->>TD: Define simple objective
    TD->>TD: Validate invariants
    TD->>EX: Pass objective
    EX->>EX: Run with baseline policy
    EX->>AN: Results
    AN->>AN: Quick analysis
    AN->>U: Feasibility assessment
    
    alt Looks promising
        U->>U: Proceed to full training
    else Not promising
        U->>TD: Iterate on design
    end
```

## Research Publication Pattern

```mermaid
sequenceDiagram
    participant TD as Task Design
    participant PT as Policy Training
    participant EX as Experiment
    participant AN as Analysis
    
    TD->>TD: Formalize objective
    TD->>PT: Objective + constraints
    PT->>PT: Train with domain randomization
    PT->>EX: Trained policy
    EX->>EX: Comprehensive benchmarks (n≥100)
    EX->>AN: Statistical data
    AN->>AN: Full analysis + viz
    AN->>EX: Validation request
    EX->>EX: Reproduce with new seeds
    EX->>AN: Confirmation data
    AN->>AN: Generate paper figures
```

## Hardware Deployment Pattern

```mermaid
sequenceDiagram
    participant TD as Task Design
    participant EX as Experiment
    participant PT as Policy Training
    participant AN as Analysis
    participant HW as Hardware
    
    TD->>TD: Include hardware constraints
    TD->>EX: Test in simulation
    EX->>EX: Validate with hardware noise
    EX->>PT: Noise-validated objective
    PT->>PT: Train with realistic model
    PT->>AN: Trained policy
    AN->>AN: Sensitivity analysis
    AN->>AN: Calibration drift study
    AN->>AN: Fine-tune parameters
    AN->>HW: Export pulse sequences
    HW->>HW: Deploy and monitor
```

## Iterative Refinement Cycle

```mermaid
graph LR
    A[Iteration 1] --> B[Task Design]
    B --> C[Experiment: Baseline]
    C --> D[Analysis: Gaps]
    
    D --> E[Iteration 2]
    E --> F[Task Design: Refine]
    F --> G[Policy Training]
    G --> H[Experiment: Validation]
    H --> I[Analysis: Performance]
    
    I --> J[Iteration 3]
    J --> K[Optimization]
    K --> L[Final Experiment]
    L --> M[Comprehensive Analysis]
    
    M --> N[Deployment]
    
    style B fill:#e1f5ff
    style C fill:#ffe1f5
    style D fill:#fff5e1
    style F fill:#e1f5ff
    style G fill:#e1ffe1
    style H fill:#ffe1f5
    style I fill:#fff5e1
    style K fill:#fff5e1
    style L fill:#ffe1f5
    style M fill:#fff5e1
```

## Cross-Skill Problem Solving

```mermaid
graph TD
    PROB[Problem Detected] --> TYPE{Problem Type?}
    
    TYPE -->|Training not converging| PT_CHECK{Check what?}
    PT_CHECK -->|Reward function| TD[Task Design:<br/>Verify reward informative]
    PT_CHECK -->|State encoding| PT[Policy Training:<br/>Improve representation]
    PT_CHECK -->|Architecture| AN[Analysis:<br/>Profile bottlenecks]
    
    TYPE -->|Results not reproducible| REPRO{Check what?}
    REPRO -->|Random seeds| EX1[Experiment:<br/>Add seed control]
    REPRO -->|Numerical stability| AN1[Analysis:<br/>Check for NaN/Inf]
    REPRO -->|Constraints| TD1[Task Design:<br/>Verify enforcement]
    
    TYPE -->|Overfitting to noise| OVER{Solution?}
    OVER -->|Increase randomization| PT1[Policy Training:<br/>Broader parameter ranges]
    OVER -->|Test robustness| AN2[Analysis:<br/>Sensitivity analysis]
    OVER -->|Reformulate objective| TD2[Task Design:<br/>Add robustness terms]
    
    TYPE -->|Baseline too strong| BASE{Why?}
    BASE -->|Objective favors it| TD3[Task Design:<br/>Emphasize adaptivity]
    BASE -->|Limited operating regime| AN3[Analysis:<br/>Find where baseline fails]
    BASE -->|Need harder scenarios| PT2[Policy Training:<br/>Curriculum learning]
    
    style TD fill:#e1f5ff
    style PT fill:#e1ffe1
    style AN fill:#fff5e1
    style EX1 fill:#ffe1f5
    style TD1 fill:#e1f5ff
    style AN1 fill:#fff5e1
    style PT1 fill:#e1ffe1
    style AN2 fill:#fff5e1
    style TD2 fill:#e1f5ff
    style TD3 fill:#e1f5ff
    style AN3 fill:#fff5e1
    style PT2 fill:#e1ffe1
```

## Skill Feature Matrix

```mermaid
graph LR
    subgraph "Task Design Features"
        TD1[Objective Formulation]
        TD2[Constraint Specification]
        TD3[Cost Function Design]
        TD4[Validation Checklists]
    end
    
    subgraph "Experiment Features"
        EX1[Statistical Analysis]
        EX2[Visualization]
        EX3[Reproducibility]
        EX4[Benchmarking]
    end
    
    subgraph "Policy Training Features"
        PT1[Architectures<br/>Transformer/LSTM/GNN]
        PT2[RL Algorithms<br/>PPO/SAC/PG]
        PT3[Domain Randomization]
        PT4[Debugging Tools]
    end
    
    subgraph "Analysis Features"
        AN1[Convergence Analysis]
        AN2[Back-action Analysis]
        AN3[Policy Interpretation]
        AN4[Optimization]
    end
    
    TD1 --> EX1
    TD2 --> PT3
    TD3 --> AN2
    
    PT1 --> EX4
    PT2 --> AN1
    
    EX1 --> AN1
    EX2 --> AN3
    
    style TD1 fill:#e1f5ff
    style TD2 fill:#e1f5ff
    style TD3 fill:#e1f5ff
    style TD4 fill:#e1f5ff
    style EX1 fill:#ffe1f5
    style EX2 fill:#ffe1f5
    style EX3 fill:#ffe1f5
    style EX4 fill:#ffe1f5
    style PT1 fill:#e1ffe1
    style PT2 fill:#e1ffe1
    style PT3 fill:#e1ffe1
    style PT4 fill:#e1ffe1
    style AN1 fill:#fff5e1
    style AN2 fill:#fff5e1
    style AN3 fill:#fff5e1
    style AN4 fill:#fff5e1
```

## Legend

- 🔵 **Task Design** (Blue): Objective formulation and specification
- 🔴 **Experiment** (Pink): Running experiments and validation  
- 🟢 **Policy Training** (Green): Training adaptive measurement policies
- 🟡 **Analysis** (Yellow): Interpreting results and optimization

## Quick Navigation

Based on your current phase:

| Phase | Primary Skill | Supporting Skills |
|-------|--------------|-------------------|
| **Starting new project** | Task Design | Skills Guide |
| **Establishing baselines** | Experiment | Task Design |
| **Developing policies** | Policy Training | Task Design, Experiment |
| **Validating results** | Experiment | Analysis |
| **Understanding behavior** | Analysis | Experiment |
| **Optimizing protocols** | Analysis | All skills |
| **Debugging issues** | Analysis | All skills |

## Workflow Complexity Levels

### Level 1: Simple (1-2 days)
```
Task Design → Experiment (baseline) → Quick Analysis
```

### Level 2: Standard (1-2 weeks)  
```
Task Design → Policy Training → Experiment → Analysis
```

### Level 3: Comprehensive (2-4 weeks)
```
Task Design → Baseline Experiment → Policy Training → 
Validation Experiment → Full Analysis → Optimization → 
Reproduction
```

### Level 4: Publication (1-2 months)
```
Task Design (with theory) → Multiple Policy Variants → 
Extensive Experiments → Statistical Analysis → 
Ablation Studies → Noise Characterization → 
Optimization → Multiple Reproductions → Paper Writing
```

---

**Ready to start?** Pick your workflow level and follow the corresponding skill sequence. The Skills Guide provides detailed integration patterns for each level.
