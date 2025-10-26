# UMM Analysis and Optimization Skill

## Purpose
This skill provides best practices for analyzing UMM experiment results, optimizing measurement sequences, and extracting physical insights from adaptive measurement data. Use this when interpreting policy behavior, debugging performance issues, or optimizing measurement protocols.

## Core Analysis Tasks

### 1. Fidelity Analysis
Understand convergence to target states.

### 2. Back-action Quantification
Measure measurement-induced disturbance.

### 3. Information Gain Analysis
Track how information is extracted over time.

### 4. Policy Interpretability
Understand what strategies policies learn.

### 5. Noise Sensitivity
Characterize robustness to parameter variations.

## Spec Invariants for UMM Analysis

Before analyzing any results, verify:
- **INVARIANT 1**: All data is from completed episodes (no truncated trajectories)
- **INVARIANT 2**: Metrics are computed consistently across comparisons
- **INVARIANT 3**: Statistical tests account for multiple comparisons (Bonferroni)
- **INVARIANT 4**: Physical quantities have correct units and bounds
- **INVARIANT 5**: Visualization scales are appropriate for quantum metrics
- **INVARIANT 6**: Confidence intervals include finite-sample corrections
- **INVARIANT 7**: Information-theoretic quantities use consistent bases (nats vs bits)

## Fidelity Trajectory Analysis

### Convergence Metrics

```python
def analyze_fidelity_convergence(trajectories, target_fidelity=0.95):
    """Analyze how fidelity evolves over measurement sequence."""
    
    metrics = {}
    
    # Convergence time (steps to reach target)
    convergence_times = []
    for traj in trajectories:
        times = np.where(traj >= target_fidelity)[0]
        if len(times) > 0:
            convergence_times.append(times[0])
        else:
            convergence_times.append(len(traj))  # Didn't converge
    
    metrics["mean_convergence_time"] = np.mean(convergence_times)
    metrics["convergence_probability"] = np.mean([t < len(traj) for t in convergence_times])
    
    # Convergence rate (exponential fit)
    # F(t) ≈ F_∞ - (F_∞ - F_0) exp(-t/τ)
    def exponential_model(t, F_inf, tau):
        return F_inf * (1 - np.exp(-t / tau))
    
    avg_trajectory = np.mean(trajectories, axis=0)
    from scipy.optimize import curve_fit
    try:
        params, _ = curve_fit(
            exponential_model,
            np.arange(len(avg_trajectory)),
            avg_trajectory,
            p0=[0.95, 10]
        )
        metrics["F_infinity"] = params[0]
        metrics["tau_convergence"] = params[1]
    except:
        metrics["F_infinity"] = avg_trajectory[-1]
        metrics["tau_convergence"] = None
    
    # Stability (variance in final 25% of trajectory)
    final_segment = [traj[int(0.75*len(traj)):] for traj in trajectories]
    metrics["final_stability"] = np.mean([np.std(seg) for seg in final_segment])
    
    # Monotonicity (fraction of steps with positive ΔF)
    monotonic_steps = []
    for traj in trajectories:
        diffs = np.diff(traj)
        monotonic_steps.append(np.mean(diffs >= 0))
    metrics["monotonicity"] = np.mean(monotonic_steps)
    
    return metrics
```

### Visualization

```python
def plot_fidelity_analysis(trajectories, save_path="fidelity_analysis.png"):
    """Comprehensive fidelity visualization."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # (0,0): Individual trajectories
    ax = axes[0, 0]
    for traj in trajectories[:20]:  # Show subset
        ax.plot(traj, alpha=0.3, color='blue')
    ax.plot(np.mean(trajectories, axis=0), 'r-', linewidth=2, label='Mean')
    ax.fill_between(
        range(len(trajectories[0])),
        np.percentile(trajectories, 25, axis=0),
        np.percentile(trajectories, 75, axis=0),
        alpha=0.3, color='red', label='IQR'
    )
    ax.set_xlabel('Measurement Step')
    ax.set_ylabel('Fidelity')
    ax.set_title('Trajectory Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (0,1): Convergence time distribution
    ax = axes[0, 1]
    convergence_times = []
    for traj in trajectories:
        times = np.where(traj >= 0.95)[0]
        if len(times) > 0:
            convergence_times.append(times[0])
    ax.hist(convergence_times, bins=20, density=True, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Steps to F > 0.95')
    ax.set_ylabel('Density')
    ax.set_title('Convergence Time Distribution')
    ax.axvline(np.mean(convergence_times), color='red', linestyle='--', 
               label=f'Mean = {np.mean(convergence_times):.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (0,2): Final fidelity distribution
    ax = axes[0, 2]
    final_fidelities = [traj[-1] for traj in trajectories]
    ax.hist(final_fidelities, bins=30, density=True, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Final Fidelity')
    ax.set_ylabel('Density')
    ax.set_title(f'Final F: {np.mean(final_fidelities):.4f} ± {np.std(final_fidelities):.4f}')
    ax.axvline(0.95, color='red', linestyle='--', label='Target')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (1,0): Convergence rate analysis
    ax = axes[1, 0]
    avg_traj = np.mean(trajectories, axis=0)
    ax.semilogy(1 - avg_traj, 'b-', label='1 - F(t)')
    
    # Fit exponential
    t = np.arange(len(avg_traj))
    log_infidelity = np.log(np.maximum(1 - avg_traj, 1e-6))
    valid = ~np.isinf(log_infidelity)
    if np.sum(valid) > 5:
        slope, intercept = np.polyfit(t[valid], log_infidelity[valid], 1)
        ax.plot(t, np.exp(slope * t + intercept), 'r--', 
                label=f'Fit: τ = {-1/slope:.2f}')
    
    ax.set_xlabel('Measurement Step')
    ax.set_ylabel('Infidelity (log scale)')
    ax.set_title('Exponential Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (1,1): Step-wise improvement
    ax = axes[1, 1]
    improvements = []
    for traj in trajectories:
        diffs = np.diff(traj)
        improvements.extend(diffs)
    ax.hist(improvements, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('ΔF per step')
    ax.set_ylabel('Density')
    ax.set_title(f'Step Improvements: {np.mean(improvements):.5f} ± {np.std(improvements):.5f}')
    ax.grid(True, alpha=0.3)
    
    # (1,2): Purity evolution
    ax = axes[1, 2]
    ax.text(0.5, 0.5, 'Reserved for\nPurity Analysis', 
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

## Back-action Analysis

Quantify measurement-induced disturbance:

```python
def analyze_backaction(histories, target_state):
    """Analyze measurement back-action."""
    
    metrics = {}
    
    # Total back-action cost
    backactions = []
    for history in histories:
        total_ba = sum(
            instr.strength for instr, _ in history
            if isinstance(instr, WeakMeasurementInstruction)
        )
        backactions.append(total_ba)
    
    metrics["mean_backaction"] = np.mean(backactions)
    metrics["std_backaction"] = np.std(backactions)
    
    # Back-action efficiency: ΔF / C_BA
    efficiencies = []
    for history, final_fidelity in zip(histories, final_fidelities):
        ba = sum(instr.strength for instr, _ in history 
                 if isinstance(instr, WeakMeasurementInstruction))
        if ba > 0:
            efficiency = final_fidelity / ba
            efficiencies.append(efficiency)
    
    metrics["mean_efficiency"] = np.mean(efficiencies)
    
    # Measurement strength distribution
    all_strengths = []
    for history in histories:
        strengths = [instr.strength for instr, _ in history
                     if isinstance(instr, WeakMeasurementInstruction)]
        all_strengths.extend(strengths)
    
    metrics["mean_strength"] = np.mean(all_strengths)
    metrics["std_strength"] = np.std(all_strengths)
    metrics["strength_range"] = [np.min(all_strengths), np.max(all_strengths)]
    
    # Adaptive strength pattern
    avg_strength_trajectory = []
    max_len = max(len(h) for h in histories)
    for t in range(max_len):
        strengths_at_t = [
            h[t][0].strength for h in histories if t < len(h)
            and isinstance(h[t][0], WeakMeasurementInstruction)
        ]
        if strengths_at_t:
            avg_strength_trajectory.append(np.mean(strengths_at_t))
    
    metrics["strength_trajectory"] = avg_strength_trajectory
    
    # Correlation between strength and outcome
    strength_outcome_corr = []
    for history in histories:
        strengths = []
        outcomes = []
        for instr, outcome in history:
            if isinstance(instr, WeakMeasurementInstruction):
                strengths.append(instr.strength)
                outcomes.append(outcome)
        
        if len(strengths) > 2:
            corr = np.corrcoef(strengths, outcomes)[0, 1]
            if not np.isnan(corr):
                strength_outcome_corr.append(corr)
    
    metrics["strength_outcome_correlation"] = np.mean(strength_outcome_corr)
    
    return metrics
```

## Information Gain Analysis

Track how information is extracted:

```python
def analyze_information_gain(histories, states):
    """Analyze information gain per measurement."""
    
    # Mutual information I(K:Ψ) = H(K) - H(K|Ψ)
    # where K is measurement outcome, Ψ is quantum state
    
    metrics = {}
    
    # Outcome entropy (how variable are measurements)
    all_outcomes = []
    for history in histories:
        outcomes = [k for _, k in history if k is not None]
        all_outcomes.extend(outcomes)
    
    # Binary outcomes: H(K) = -p log p - (1-p) log(1-p)
    if all_outcomes:
        p_plus = np.mean(all_outcomes)  # Assume outcome ∈ {0,1}
        if 0 < p_plus < 1:
            outcome_entropy = -(p_plus * np.log2(p_plus) + 
                               (1-p_plus) * np.log2(1-p_plus))
        else:
            outcome_entropy = 0
        
        metrics["outcome_entropy"] = outcome_entropy
    
    # State uncertainty reduction (von Neumann entropy)
    initial_entropies = []
    final_entropies = []
    
    for state_trajectory in states:
        # Initial state entropy
        S_init = compute_von_neumann_entropy(state_trajectory[0].rho)
        initial_entropies.append(S_init)
        
        # Final state entropy
        S_final = compute_von_neumann_entropy(state_trajectory[-1].rho)
        final_entropies.append(S_final)
    
    metrics["mean_entropy_reduction"] = (np.mean(initial_entropies) - 
                                         np.mean(final_entropies))
    
    # Information gain per measurement
    info_per_meas = metrics["mean_entropy_reduction"] / len(histories[0])
    metrics["info_per_measurement"] = info_per_meas
    
    # Fisher information (for parameter estimation)
    # F_Q = 4 Var(H) for measurements of generator H
    fisher_info = compute_quantum_fisher_information(states)
    metrics["quantum_fisher_info"] = fisher_info
    
    return metrics

def compute_von_neumann_entropy(rho):
    """Compute S(ρ) = -Tr(ρ log ρ)."""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]  # Remove numerical zeros
    return -np.sum(eigvals * np.log2(eigvals))

def compute_quantum_fisher_information(state, parameter_generator):
    """
    Compute quantum Fisher information.
    
    F_Q = 4 Var_ρ(H) = 4[⟨H²⟩ - ⟨H⟩²]
    
    where H is the generator of parameter evolution.
    """
    
    rho = state.rho
    H = parameter_generator
    
    # Expectation values
    H_avg = np.real(np.trace(rho @ H))
    H2_avg = np.real(np.trace(rho @ H @ H))
    
    # Variance
    var_H = H2_avg - H_avg**2
    
    # Fisher information
    F_Q = 4 * var_H
    
    return F_Q
```

## Policy Behavior Analysis

Understand what strategies are learned:

```python
def analyze_policy_behavior(policy, test_states, n_samples=1000):
    """Analyze learned policy behavior."""
    
    metrics = {}
    
    # Action distribution across states
    actions = []
    for state in test_states:
        action = policy(state, [])
        actions.append({
            "axis": action.axis,
            "strength": action.strength
        })
    
    # Preferred measurement axes
    axes = np.array([a["axis"] for a in actions])
    
    # Check if policy has preferred directions
    mean_axis = np.mean(axes, axis=0)
    axis_alignment = np.dot(axes, mean_axis / np.linalg.norm(mean_axis))
    metrics["axis_coherence"] = np.mean(axis_alignment)
    
    # Visualize in Bloch sphere
    if axes.shape[1] == 3:
        plot_axes_on_bloch_sphere(axes, "policy_measurement_axes.png")
    
    # Strength adaptation pattern
    strengths = np.array([a["strength"] for a in actions])
    metrics["mean_strength"] = np.mean(strengths)
    metrics["std_strength"] = np.std(strengths)
    
    # Correlation: strength vs distance to target
    if hasattr(test_states[0], 'to_bloch_vector'):
        distances = []
        for state in test_states:
            bloch = state.to_bloch_vector()
            target = np.array([1, 0, 0])  # Assume |+x⟩
            distance = np.linalg.norm(bloch - target)
            distances.append(distance)
        
        corr = np.corrcoef(distances, strengths)[0, 1]
        metrics["strength_distance_correlation"] = corr
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(distances, strengths, alpha=0.5)
        plt.xlabel('Distance to Target')
        plt.ylabel('Measurement Strength')
        plt.title(f'Policy Adaptation (corr = {corr:.3f})')
        plt.grid(True, alpha=0.3)
        plt.savefig('policy_strength_adaptation.png', dpi=150)
        plt.close()
    
    # Decision boundary analysis (if applicable)
    # For different state regions, what actions does policy take?
    
    return metrics

def plot_axes_on_bloch_sphere(axes, save_path):
    """Visualize measurement axes on Bloch sphere."""
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Bloch sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='gray')
    
    # Draw measurement axes
    for axis in axes[:100]:  # Subsample
        ax.quiver(0, 0, 0, axis[0], axis[1], axis[2],
                 color='blue', alpha=0.3, arrow_length_ratio=0.1)
    
    # Mean axis
    mean_axis = np.mean(axes, axis=0)
    mean_axis = mean_axis / np.linalg.norm(mean_axis)
    ax.quiver(0, 0, 0, mean_axis[0], mean_axis[1], mean_axis[2],
             color='red', linewidth=3, arrow_length_ratio=0.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Policy Measurement Axes Distribution')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

## Noise Sensitivity Analysis

Characterize robustness:

```python
def analyze_noise_sensitivity(
    policy,
    objective_spec,
    parameter_ranges,
    n_samples_per_param=20
):
    """Systematic noise sensitivity analysis."""
    
    parser = ObjectiveParser()
    reward_fn, constraints = parser.compile(objective_spec)
    
    results = {}
    
    # Sweep T1
    if "T1" in parameter_ranges:
        T1_values = np.linspace(*parameter_ranges["T1"], 10)
        fidelities_vs_T1 = []
        
        for T1 in T1_values:
            fids = []
            for _ in range(n_samples_per_param):
                simulator = UMMSimulator(n_qubits=1, T1=T1, T2=50e-6)
                result = simulator.run_adaptive(policy, reward_fn, constraints, max_steps=32)
                fids.append(result.fidelity)
            
            fidelities_vs_T1.append({
                "T1": T1,
                "mean": np.mean(fids),
                "std": np.std(fids)
            })
        
        results["T1_sweep"] = fidelities_vs_T1
    
    # Sweep T2
    if "T2" in parameter_ranges:
        T2_values = np.linspace(*parameter_ranges["T2"], 10)
        fidelities_vs_T2 = []
        
        for T2 in T2_values:
            fids = []
            for _ in range(n_samples_per_param):
                simulator = UMMSimulator(n_qubits=1, T1=50e-6, T2=T2)
                result = simulator.run_adaptive(policy, reward_fn, constraints, max_steps=32)
                fids.append(result.fidelity)
            
            fidelities_vs_T2.append({
                "T2": T2,
                "mean": np.mean(fids),
                "std": np.std(fids)
            })
        
        results["T2_sweep"] = fidelities_vs_T2
    
    # 2D heatmap: T1 vs T2
    if "T1" in parameter_ranges and "T2" in parameter_ranges:
        T1_grid = np.linspace(*parameter_ranges["T1"], 8)
        T2_grid = np.linspace(*parameter_ranges["T2"], 8)
        
        fidelity_grid = np.zeros((len(T1_grid), len(T2_grid)))
        
        for i, T1 in enumerate(T1_grid):
            for j, T2 in enumerate(T2_grid):
                fids = []
                for _ in range(n_samples_per_param):
                    simulator = UMMSimulator(n_qubits=1, T1=T1, T2=T2)
                    result = simulator.run_adaptive(policy, reward_fn, constraints, max_steps=32)
                    fids.append(result.fidelity)
                
                fidelity_grid[i, j] = np.mean(fids)
        
        results["T1_T2_heatmap"] = {
            "T1_grid": T1_grid,
            "T2_grid": T2_grid,
            "fidelity_grid": fidelity_grid
        }
    
    # Visualize
    plot_noise_sensitivity(results, "noise_sensitivity.png")
    
    return results

def plot_noise_sensitivity(results, save_path):
    """Visualize noise sensitivity analysis."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # T1 sweep
    if "T1_sweep" in results:
        ax = axes[0]
        data = results["T1_sweep"]
        T1_vals = [d["T1"] for d in data]
        means = [d["mean"] for d in data]
        stds = [d["std"] for d in data]
        
        ax.plot(T1_vals, means, 'b-', linewidth=2)
        ax.fill_between(T1_vals, 
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.3)
        ax.set_xlabel('T1 (seconds)')
        ax.set_ylabel('Fidelity')
        ax.set_title('Sensitivity to T1')
        ax.grid(True, alpha=0.3)
    
    # T2 sweep
    if "T2_sweep" in results:
        ax = axes[1]
        data = results["T2_sweep"]
        T2_vals = [d["T2"] for d in data]
        means = [d["mean"] for d in data]
        stds = [d["std"] for d in data]
        
        ax.plot(T2_vals, means, 'r-', linewidth=2)
        ax.fill_between(T2_vals,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.3)
        ax.set_xlabel('T2 (seconds)')
        ax.set_ylabel('Fidelity')
        ax.set_title('Sensitivity to T2')
        ax.grid(True, alpha=0.3)
    
    # Heatmap
    if "T1_T2_heatmap" in results:
        ax = axes[2]
        data = results["T1_T2_heatmap"]
        
        im = ax.imshow(data["fidelity_grid"], 
                       extent=[data["T2_grid"][0], data["T2_grid"][-1],
                              data["T1_grid"][0], data["T1_grid"][-1]],
                       origin='lower', aspect='auto', cmap='viridis')
        ax.set_xlabel('T2 (seconds)')
        ax.set_ylabel('T1 (seconds)')
        ax.set_title('Fidelity: T1 vs T2')
        plt.colorbar(im, ax=ax, label='Fidelity')
        
        # Contours
        ax.contour(data["T2_grid"], data["T1_grid"], data["fidelity_grid"],
                  levels=[0.9, 0.95, 0.99], colors='white', linewidths=1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

## Optimization Strategies

### 1. Hyperparameter Optimization

```python
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

def optimize_measurement_protocol(objective_spec):
    """Bayesian optimization of measurement protocol."""
    
    # Define search space
    space = [
        Real(0.1, 0.9, name='strength'),
        Integer(8, 64, name='n_steps'),
        Real(0.0, 1.0, name='backaction_weight'),
        Categorical(['X', 'Y', 'Z'], name='measurement_axis')
    ]
    
    def objective(params):
        """Objective function to minimize (negative fidelity)."""
        strength, n_steps, ba_weight, axis = params
        
        # Run simulation
        simulator = UMMSimulator(n_qubits=1, T1=50e-6, T2=50e-6)
        # ... configure with params ...
        result = run_experiment(simulator, params)
        
        # Return negative fidelity (minimization)
        return -result.fidelity
    
    # Optimize
    result = gp_minimize(
        objective,
        space,
        n_calls=100,
        random_state=42
    )
    
    return result
```

### 2. Multi-Objective Optimization

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

class UMMMultiObjective(Problem):
    """Multi-objective optimization for UMM protocols."""
    
    def __init__(self):
        super().__init__(
            n_var=3,  # strength, n_steps, weight
            n_obj=3,  # maximize fidelity, minimize backaction, minimize time
            n_constr=0,
            xl=np.array([0.1, 8, 0.0]),
            xu=np.array([0.9, 64, 1.0])
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objectives."""
        
        # Run simulations
        fidelities = []
        backactions = []
        times = []
        
        for params in x:
            result = run_umm_experiment(params)
            fidelities.append(result.fidelity)
            backactions.append(result.backaction)
            times.append(result.wall_time)
        
        # Return objectives (minimization convention)
        out["F"] = np.column_stack([
            -np.array(fidelities),  # Maximize fidelity
            np.array(backactions),   # Minimize backaction
            np.array(times)          # Minimize time
        ])

def run_pareto_optimization():
    """Find Pareto-optimal measurement protocols."""
    
    problem = UMMMultiObjective()
    algorithm = NSGA2(pop_size=100)
    
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 200),
        verbose=True
    )
    
    # Extract Pareto front
    pareto_front = res.F
    pareto_solutions = res.X
    
    return pareto_front, pareto_solutions
```

### 3. Gradient-Based Pulse Optimization

For hardware implementation:

```python
def optimize_pulse_sequence(target_unitary, n_time_steps=100):
    """Optimize control pulses via GRAPE."""
    
    # Initialize random pulse sequence
    pulses = torch.randn(n_time_steps, requires_grad=True)
    optimizer = torch.optim.Adam([pulses], lr=0.01)
    
    for iteration in range(1000):
        # Forward propagation
        U = compute_propagator(pulses)
        
        # Fidelity with target
        fidelity = torch.abs(torch.trace(U.conj().T @ target_unitary))**2 / 4
        loss = 1 - fidelity
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Project onto constraints
        with torch.no_grad():
            pulses.clamp_(-1, 1)  # Amplitude limits
        
        if iteration % 100 == 0:
            print(f"Iter {iteration}: Fidelity = {fidelity.item():.6f}")
    
    return pulses.detach().numpy()
```

## Diagnostic Tools

### Performance Profiler

```python
def profile_umm_performance(policy, objective_spec, n_trials=100):
    """Comprehensive performance profiling."""
    
    import time
    
    profile = {
        "timing": {},
        "memory": {},
        "efficiency": {}
    }
    
    # Time each component
    start = time.time()
    parser = ObjectiveParser()
    reward_fn, constraints = parser.compile(objective_spec)
    profile["timing"]["parsing"] = time.time() - start
    
    start = time.time()
    simulator = UMMSimulator(n_qubits=1, T1=50e-6, T2=50e-6)
    profile["timing"]["simulator_init"] = time.time() - start
    
    # Run trials and time policy calls
    policy_times = []
    simulation_times = []
    
    for trial in range(n_trials):
        simulator.reset()
        
        for step in range(32):
            # Time policy
            start = time.time()
            action = policy(simulator.history, simulator.state)
            policy_times.append(time.time() - start)
            
            # Time simulation
            start = time.time()
            simulator._execute_instruction(action)
            simulation_times.append(time.time() - start)
    
    profile["timing"]["mean_policy_call"] = np.mean(policy_times)
    profile["timing"]["mean_simulation_step"] = np.mean(simulation_times)
    
    # Memory usage
    import psutil
    process = psutil.Process()
    profile["memory"]["rss_mb"] = process.memory_info().rss / 1024 / 1024
    
    # Efficiency metrics
    profile["efficiency"]["steps_per_second"] = 1.0 / profile["timing"]["mean_simulation_step"]
    
    return profile
```

## Summary

Effective UMM analysis requires:
1. **Trajectory analysis**: Understand convergence dynamics
2. **Back-action quantification**: Measure measurement cost
3. **Information tracking**: Quantify learning per measurement
4. **Policy interpretation**: Understand learned strategies
5. **Robustness testing**: Validate across parameter variations
6. **Optimization**: Systematically improve protocols

Analyze systematically, visualize comprehensively, optimize iteratively for maximum insight and performance.
