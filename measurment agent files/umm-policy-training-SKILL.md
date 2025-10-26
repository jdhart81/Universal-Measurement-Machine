# UMM Policy Training Skill

## Purpose
This skill provides best practices for training adaptive measurement policies using reinforcement learning (RL) with the Universal Measurement Machine (UMM) framework. Use this when developing learned policies for state preparation, Zeno stabilization, metrology, or custom measurement tasks.

## Core Concept

A **policy** π_θ maps measurement history to measurement actions:
```
π_θ: (h_1, ..., h_t) → a_t+1
```

Where:
- **History** h_t = (s_t, m_t, k_t): state belief, measurement, outcome
- **Action** a_t+1 = (n, m): measurement axis and strength
- **Parameters** θ: neural network weights or parametric model

## Spec Invariants for Policy Training

Before training any policy, verify:
- **INVARIANT 1**: State space is well-defined (Hilbert space dimension, belief representation)
- **INVARIANT 2**: Action space is feasible (m ∈ [0,1], ||n|| = 1)
- **INVARIANT 3**: Reward function is bounded and differentiable (if using policy gradients)
- **INVARIANT 4**: Episode termination is well-defined (max steps or convergence criterion)
- **INVARIANT 5**: Training environment matches deployment environment (same noise model)
- **INVARIANT 6**: Domain randomization covers expected parameter variations
- **INVARIANT 7**: Safety constraints are enforced via projection, not learned

## Policy Architectures

### 1. Transformer Policy (Best for Long Sequences)

**Architecture**:
```python
class TransformerPolicy(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=3):
        super().__init__()
        self.embedding = nn.Linear(state_dim + action_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads),
            num_layers=n_layers
        )
        self.action_head = nn.Linear(d_model, action_dim)
        
    def forward(self, history_sequence):
        # history_sequence: (seq_len, batch, features)
        x = self.embedding(history_sequence)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        action = self.action_head(x[-1])  # Last timestep
        
        # Enforce constraints
        axis = action[:3]
        axis = axis / torch.norm(axis)
        strength = torch.sigmoid(action[3])
        
        return torch.cat([axis, strength.unsqueeze(-1)])
```

**When to use**:
- Long measurement sequences (> 16 steps)
- Need to capture long-range dependencies
- Attention over past outcomes important

**Hyperparameters**:
```python
config = {
    "d_model": 128,           # Embedding dimension
    "n_heads": 4,             # Attention heads
    "n_layers": 3,            # Transformer blocks
    "max_seq_len": 32,        # Maximum episode length
    "dropout": 0.1,
    "learning_rate": 3e-4,
    "batch_size": 256
}
```

### 2. LSTM Policy (Good Balance)

**Architecture**:
```python
class LSTMPolicy(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=state_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, history_sequence, hidden=None):
        x, hidden = self.lstm(history_sequence, hidden)
        action = self.action_head(x[:, -1, :])
        
        # Enforce constraints
        axis = action[:, :3]
        axis = axis / torch.norm(axis, dim=1, keepdim=True)
        strength = torch.sigmoid(action[:, 3])
        
        return torch.cat([axis, strength.unsqueeze(-1)], dim=1), hidden
```

**When to use**:
- Medium-length sequences (8-32 steps)
- Sequential decision making
- Need stateful hidden representation

**Hyperparameters**:
```python
config = {
    "hidden_dim": 128,
    "n_layers": 2,
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "batch_size": 128
}
```

### 3. GNN Policy (For Multi-Qubit Systems)

**Architecture**:
```python
class GNNPolicy(nn.Module):
    def __init__(self, n_qubits, hidden_dim=64):
        super().__init__()
        self.n_qubits = n_qubits
        
        # Graph neural network for qubit interactions
        self.gnn = GCNConv(node_features, hidden_dim)
        self.action_head = nn.Linear(hidden_dim * n_qubits, action_dim)
        
    def forward(self, qubit_states, edge_index):
        # qubit_states: (n_qubits, features)
        # edge_index: (2, n_edges) connectivity
        
        x = F.relu(self.gnn(qubit_states, edge_index))
        x = x.view(-1)  # Flatten
        action = self.action_head(x)
        
        # Select which qubit to measure + how
        qubit_idx = torch.argmax(action[:self.n_qubits])
        axis = action[self.n_qubits:self.n_qubits+3]
        axis = axis / torch.norm(axis)
        strength = torch.sigmoid(action[-1])
        
        return qubit_idx, axis, strength
```

**When to use**:
- Multi-qubit systems (> 2 qubits)
- Spatially structured systems (lattices)
- Need to reason about qubit connectivity

### 4. Simple Parametric Policy (Fast Baseline)

For interpretable baselines:

```python
class ParametricPolicy:
    """
    Parametric policy with simple rules.
    
    a_t = (n_target, m_0 * f(distance))
    
    where f is a learned function of distance to target.
    """
    
    def __init__(self, target_axis, strength_fn):
        self.target_axis = target_axis / np.linalg.norm(target_axis)
        self.strength_fn = strength_fn  # e.g., lambda d: 0.3 + 0.5*d
    
    def __call__(self, history, state):
        bloch = state.to_bloch_vector()
        distance = np.linalg.norm(bloch - self.target_axis)
        strength = self.strength_fn(distance)
        
        return WeakMeasurementInstruction(
            axis=self.target_axis,
            strength=np.clip(strength, 0, 1)
        )
```

## Training Algorithms

### Algorithm 1: Policy Gradient (REINFORCE)

**When to use**: Simple tasks, differentiable reward, episodic setting.

**Implementation**:
```python
def train_policy_gradient(
    policy,
    objective_spec,
    n_episodes=10000,
    learning_rate=1e-3,
    gamma=0.99
):
    """Train policy with REINFORCE algorithm."""
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    parser = ObjectiveParser()
    reward_fn, constraints = parser.compile(objective_spec)
    
    for episode in range(n_episodes):
        # Rollout episode
        simulator = UMMSimulator(**noise_params)
        simulator.reset()
        
        log_probs = []
        rewards = []
        
        for t in range(max_steps):
            # Sample action from policy
            state_repr = encode_state(simulator.state, simulator.history)
            action_dist = policy(state_repr)
            action = action_dist.sample()
            log_probs.append(action_dist.log_prob(action))
            
            # Execute action
            umm_action = decode_action(action)
            simulator._execute_instruction(umm_action)
            
            # Compute reward
            reward = reward_fn(simulator.state, simulator.history)
            rewards.append(reward)
        
        # Compute returns with discount
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # Normalize returns (reduces variance)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient update
        loss = 0
        for log_prob, R in zip(log_probs, returns):
            loss += -log_prob * R
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if episode % 100 == 0:
            avg_reward = np.mean(rewards)
            print(f"Episode {episode}: Avg Reward = {avg_reward:.4f}")
    
    return policy
```

**Hyperparameters**:
- Learning rate: 1e-3 to 1e-4
- Discount γ: 0.95 to 0.99
- Episodes: 10,000 to 100,000
- Baseline: Use value function to reduce variance

### Algorithm 2: Proximal Policy Optimization (PPO)

**When to use**: Complex tasks, need sample efficiency, continuous action spaces.

**Implementation**:
```python
def train_ppo(
    policy,
    value_network,
    objective_spec,
    n_episodes=50000,
    n_epochs=10,
    clip_epsilon=0.2,
    learning_rate=3e-4
):
    """Train policy with PPO."""
    
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    value_optimizer = torch.optim.Adam(value_network.parameters(), lr=learning_rate)
    
    parser = ObjectiveParser()
    reward_fn, constraints = parser.compile(objective_spec)
    
    for episode in range(n_episodes):
        # Collect trajectories
        trajectories = []
        for _ in range(n_trajectories_per_update):
            trajectory = rollout_episode(policy, reward_fn, constraints)
            trajectories.append(trajectory)
        
        # Compute advantages
        advantages = compute_gae(trajectories, value_network, gamma=0.99, lambda_=0.95)
        
        # PPO update
        for epoch in range(n_epochs):
            for batch in make_batches(trajectories, advantages):
                # Policy update
                states, actions, old_log_probs, advs, returns = batch
                
                # New policy distribution
                new_dist = policy(states)
                new_log_probs = new_dist.log_prob(actions)
                
                # Importance sampling ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advs
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value function update
                value_pred = value_network(states)
                value_loss = F.mse_loss(value_pred, returns)
                
                # Optimize
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()
    
    return policy
```

**Hyperparameters**:
- Clip ε: 0.1 to 0.3
- Learning rate: 3e-4
- PPO epochs: 10
- Trajectories per update: 32
- GAE λ: 0.95

### Algorithm 3: Soft Actor-Critic (SAC)

**When to use**: Continuous control, off-policy learning, maximum entropy RL.

**Key features**:
- Off-policy: Sample efficient
- Maximum entropy: Encourages exploration
- Twin Q-networks: Reduces overestimation

**Hyperparameters**:
- Learning rate: 3e-4 (actor), 3e-4 (critic)
- Temperature α: 0.2 (auto-tune recommended)
- Replay buffer: 1M transitions
- Batch size: 256

### Algorithm 4: Model-Based RL (Optional)

For tasks where simulator is expensive:

```python
def train_world_model(trajectories):
    """Learn dynamics model: s_t+1 = f(s_t, a_t)."""
    
    model = DynamicsNetwork()
    
    for epoch in range(n_epochs):
        for s_t, a_t, s_next in trajectories:
            pred_next = model(s_t, a_t)
            loss = F.mse_loss(pred_next, s_next)
            # ... optimize ...
    
    return model

def plan_with_model(model, policy, initial_state):
    """Use learned model for planning."""
    
    # Model Predictive Control
    best_action = None
    best_value = -float('inf')
    
    for _ in range(n_samples):
        action = policy.sample()
        # Rollout in model
        state = initial_state
        total_reward = 0
        for t in range(horizon):
            state = model(state, action)
            total_reward += reward_fn(state)
        
        if total_reward > best_value:
            best_value = total_reward
            best_action = action
    
    return best_action
```

## Domain Randomization

Critical for robustness:

```python
def domain_randomized_training(
    policy,
    objective_spec,
    param_ranges
):
    """Train with randomized environment parameters."""
    
    for episode in range(n_episodes):
        # Sample noise parameters
        T1 = np.random.uniform(*param_ranges["T1"])
        T2 = np.random.uniform(*param_ranges["T2"])
        readout_error = np.random.uniform(*param_ranges["readout_error"])
        
        # Train on this instance
        simulator = UMMSimulator(
            n_qubits=1,
            T1=T1,
            T2=T2,
            readout_error=readout_error
        )
        
        # ... run episode and update policy ...
```

**Recommended ranges**:
```python
param_ranges = {
    "T1": (30e-6, 100e-6),      # ±50% around nominal
    "T2": (30e-6, 100e-6),
    "readout_error": (0.001, 0.05),
    "gate_error": (0.0001, 0.01)
}
```

## State Representation

Critical design choice: how to encode history for policy input.

### Option 1: Bloch Vector + Statistics

```python
def encode_state_bloch(state, history):
    """Encode as Bloch vector + measurement statistics."""
    
    bloch = state.to_bloch_vector()  # [x, y, z]
    
    # Recent measurement statistics
    recent_outcomes = [k for _, k in history[-5:]]
    avg_outcome = np.mean(recent_outcomes) if recent_outcomes else 0.5
    
    # Recent measurement strengths
    recent_strengths = [
        instr.strength for instr, _ in history[-5:]
        if isinstance(instr, WeakMeasurementInstruction)
    ]
    avg_strength = np.mean(recent_strengths) if recent_strengths else 0.0
    
    # Fidelity trend (if target known)
    # fidelities = [compute_fidelity(s) for s in state_history]
    # fidelity_slope = linear_fit(fidelities)
    
    return np.concatenate([
        bloch,                      # [x, y, z]
        [avg_outcome],              # Recent measurement bias
        [avg_strength],             # Recent measurement strength
        [len(history)],             # Time step
        # [fidelity_slope]          # Progress indicator
    ])
```

### Option 2: Density Matrix (Flattened)

```python
def encode_state_density(state, history):
    """Flatten density matrix for input."""
    
    # Real and imaginary parts
    rho_real = np.real(state.rho).flatten()
    rho_imag = np.imag(state.rho).flatten()
    
    # Measurement history (last 5)
    history_vector = np.zeros(5 * 4)  # 5 measurements × 4 features
    for i, (instr, outcome) in enumerate(history[-5:]):
        if isinstance(instr, WeakMeasurementInstruction):
            history_vector[i*4:(i+1)*4] = [
                *instr.axis,
                instr.strength
            ]
    
    return np.concatenate([rho_real, rho_imag, history_vector])
```

### Option 3: Belief State (Bayesian)

```python
class BeliefStateRepresentation:
    """Maintain belief over quantum states."""
    
    def __init__(self, n_particles=1000):
        self.particles = self.initialize_particles(n_particles)
        self.weights = np.ones(n_particles) / n_particles
    
    def update(self, action, outcome):
        """Bayesian update given measurement."""
        
        for i, particle in enumerate(self.particles):
            # Likelihood of outcome given particle
            prob = self.measurement_probability(particle, action, outcome)
            self.weights[i] *= prob
        
        # Renormalize
        self.weights /= self.weights.sum()
        
        # Resample if effective sample size low
        if 1.0 / np.sum(self.weights**2) < len(self.particles) / 2:
            self.resample()
    
    def get_state_estimate(self):
        """Return weighted average state."""
        return np.average(self.particles, weights=self.weights, axis=0)
```

## Reward Shaping

Critical for learning efficiency.

### Sparse vs Dense Rewards

**Sparse** (only at end):
```python
def sparse_reward(state, history, target_state):
    if len(history) < max_steps:
        return 0  # No intermediate reward
    else:
        return state.fidelity(target_state)  # Only at end
```

**Dense** (every step):
```python
def dense_reward(state, history, target_state):
    # Fidelity progress
    fidelity = state.fidelity(target_state)
    
    # Penalties
    backaction = history[-1][0].strength if history else 0
    
    return fidelity - 0.1 * backaction  # Balance objectives
```

**Recommendation**: Start with dense rewards for faster learning, then anneal to sparse for better final performance.

### Curriculum Learning

Gradually increase task difficulty:

```python
def curriculum_training(policy, objective_spec):
    """Train with increasing difficulty."""
    
    # Stage 1: Short sequences, low noise
    train(policy, T1=100e-6, T2=100e-6, max_steps=8)
    
    # Stage 2: Medium sequences, medium noise
    train(policy, T1=50e-6, T2=50e-6, max_steps=16)
    
    # Stage 3: Long sequences, high noise
    train(policy, T1=30e-6, T2=30e-6, max_steps=32)
```

## Safety Constraints

Enforce via projection, not learning:

```python
def project_action_to_constraints(action, constraints):
    """Hard constraint enforcement."""
    
    axis, strength = action[:3], action[3]
    
    # Normalize axis
    axis = axis / np.linalg.norm(axis)
    
    # Clip strength
    max_strength = constraints.get("max_strength", 1.0)
    strength = np.clip(strength, 0, max_strength)
    
    # Check measurement rate
    if len(history) > 0:
        last_time = history[-1].wall_time
        min_interval = constraints.get("min_wait", 100e-9)
        if time.time() - last_time < min_interval:
            # Force wait instead
            return WaitInstruction(min_interval)
    
    return WeakMeasurementInstruction(axis=axis, strength=strength)
```

## Training Workflow

### Complete Training Pipeline

```python
def train_umm_policy(
    objective_spec,
    architecture="transformer",
    algorithm="ppo",
    n_episodes=50000,
    save_path="trained_policy.pt"
):
    """Complete training pipeline."""
    
    # 1. Parse objective
    parser = ObjectiveParser()
    reward_fn, constraints = parser.compile(objective_spec)
    
    # 2. Initialize policy
    if architecture == "transformer":
        policy = TransformerPolicy(d_model=128, n_heads=4)
    elif architecture == "lstm":
        policy = LSTMPolicy(hidden_dim=128)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # 3. Initialize trainer
    if algorithm == "ppo":
        trainer = PPOTrainer(policy, reward_fn, constraints)
    elif algorithm == "pg":
        trainer = PolicyGradientTrainer(policy, reward_fn, constraints)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # 4. Domain randomization setup
    param_ranges = {
        "T1": (30e-6, 100e-6),
        "T2": (30e-6, 100e-6),
        "readout_error": (0.001, 0.05)
    }
    
    # 5. Training loop
    for episode in range(n_episodes):
        # Sample environment parameters
        env_params = sample_params(param_ranges)
        
        # Run episode
        simulator = UMMSimulator(**env_params)
        trajectory = trainer.collect_trajectory(simulator)
        
        # Update policy
        metrics = trainer.update(trajectory)
        
        # Logging
        if episode % 100 == 0:
            print(f"Episode {episode}: {metrics}")
        
        # Checkpointing
        if episode % 1000 == 0:
            torch.save(policy.state_dict(), f"{save_path}.ep{episode}")
    
    # 6. Save final policy
    torch.save(policy.state_dict(), save_path)
    
    return policy
```

### Evaluation Protocol

```python
def evaluate_policy(policy, objective_spec, n_trials=100):
    """Evaluate trained policy."""
    
    parser = ObjectiveParser()
    reward_fn, constraints = parser.compile(objective_spec)
    
    results = {
        "fidelities": [],
        "backactions": [],
        "wall_times": []
    }
    
    for trial in range(n_trials):
        simulator = UMMSimulator(n_qubits=1, T1=50e-6, T2=50e-6, seed=trial)
        
        # Run episode
        result = simulator.run_adaptive(
            policy=policy,
            reward_fn=reward_fn,
            constraints=constraints,
            max_steps=32
        )
        
        results["fidelities"].append(result.fidelity)
        results["backactions"].append(result.backaction)
        results["wall_times"].append(result.wall_time)
    
    # Summary statistics
    summary = {
        "mean_fidelity": np.mean(results["fidelities"]),
        "std_fidelity": np.std(results["fidelities"]),
        "mean_backaction": np.mean(results["backactions"]),
        "mean_time": np.mean(results["wall_times"])
    }
    
    return summary
```

## Debugging Training

### Issue 1: Policy Not Learning

**Symptoms**: Reward plateaus early, no improvement

**Diagnostics**:
```python
# Check gradient flow
for name, param in policy.named_parameters():
    print(f"{name}: grad_norm = {param.grad.norm().item()}")

# Check state representation
print(f"State vector: {state_repr}")
print(f"Range: [{state_repr.min()}, {state_repr.max()}]")

# Check reward scale
print(f"Rewards: mean={np.mean(rewards)}, std={np.std(rewards)}")
```

**Solutions**:
- Normalize state representation
- Scale rewards to [-1, 1]
- Reduce learning rate
- Add reward shaping

### Issue 2: Policy Collapse

**Symptoms**: Policy outputs same action regardless of state

**Diagnostics**:
```python
# Check action diversity
actions = [policy(state) for state in test_states]
print(f"Action std: {np.std(actions, axis=0)}")

# Check entropy
action_dist = policy(state)
entropy = action_dist.entropy()
print(f"Policy entropy: {entropy}")
```

**Solutions**:
- Add entropy bonus to reward
- Increase exploration (ε-greedy, noise)
- Check if reward function is informative

### Issue 3: Unstable Training

**Symptoms**: Reward variance explodes, NaN losses

**Diagnostics**:
```python
# Check for exploding gradients
grad_norms = [p.grad.norm() for p in policy.parameters() if p.grad is not None]
print(f"Max gradient norm: {max(grad_norms)}")

# Check value estimates
print(f"Value range: [{values.min()}, {values.max()}]")
```

**Solutions**:
- Clip gradients (norm ≤ 1.0)
- Normalize advantages
- Reduce learning rate
- Add gradient penalty

## Pretrained Baselines

Include warm-start policies:

```python
def load_pretrained_policy(task_type):
    """Load pretrained baseline policy."""
    
    baselines = {
        "state_prep_qubit": "models/state_prep_transformer.pt",
        "zeno_stabilization": "models/zeno_lstm.pt",
        "adaptive_metrology": "models/metrology_transformer.pt"
    }
    
    if task_type not in baselines:
        raise ValueError(f"No baseline for {task_type}")
    
    policy = TransformerPolicy.load(baselines[task_type])
    return policy
```

## Performance Benchmarks

Expected performance after training:

| Task | Architecture | Episodes | Final Performance |
|------|-------------|----------|-------------------|
| State Prep (1Q) | Transformer | 20k | F > 0.96 |
| State Prep (1Q) | LSTM | 30k | F > 0.95 |
| Zeno (1Q) | LSTM | 50k | P_survive > 0.90 |
| Metrology (1Q) | Transformer | 100k | QFI near Heisenberg |
| LOCC (2Q) | GNN | 200k | C > 0.8 (from 0.5) |

## Summary

Successful UMM policy training requires:
1. **Right architecture**: Match network to task complexity
2. **Proper state encoding**: Capture relevant history compactly
3. **Reward shaping**: Dense rewards for learning, sparse for optimality
4. **Domain randomization**: Robustness to parameter variations
5. **Hard constraints**: Safety via projection, not learning
6. **Systematic debugging**: Monitor gradients, entropy, action diversity

Train policies iteratively: start simple, add complexity gradually, validate continuously.
