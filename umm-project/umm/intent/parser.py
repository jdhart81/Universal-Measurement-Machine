"""
Intent Parser: Compile user objectives into reward functions and constraints.

Supports:
- DSL parsing (BNF grammar)
- JSON schema validation
- Reward function generation
- Constraint set compilation
"""

import json
import numpy as np
from typing import Tuple, Dict, Any, Callable
from dataclasses import dataclass
import re


@dataclass
class ParsedObjective:
    """Parsed user objective."""
    task: str
    target_state: Any = None
    monotone: str = None
    constraints: Dict[str, Any] = None
    cost_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}
        if self.cost_weights is None:
            self.cost_weights = {"backaction": 0.5, "time": 0.2}


class ObjectiveParser:
    """
    Parse user objectives from DSL or JSON.
    
    DSL Grammar (simplified):
        objective := STATE_PREP <state> [WITH_COST <cost>] [BUDGET <resource> <value>]
                   | ENTANGLEMENT_MAX <monotone> [UNDER LOCC]
                   | PHASE_EST <param> [BUDGET PROBES <N>]
    
    JSON Schema: See docs/json_schema.json
    """
    
    def __init__(self):
        self.supported_tasks = ["state_prep", "entanglement_max", "phase_est", "zeno"]
        self.supported_states = {
            "|+x>": np.array([1, 1]) / np.sqrt(2),
            "|-x>": np.array([1, -1]) / np.sqrt(2),
            "|+y>": np.array([1, 1j]) / np.sqrt(2),
            "|0>": np.array([1, 0]),
            "|1>": np.array([0, 1]),
        }
    
    def parse_dsl(self, dsl_string: str) -> ParsedObjective:
        """
        Parse DSL string into ParsedObjective.
        
        Example:
            "STATE_PREP |+x> WITH_COST WEIGHTED 0.5*BACKACTION + 0.2*TIME BUDGET TIME 10µs"
        
        Args:
            dsl_string: DSL command string
        
        Returns:
            ParsedObjective
        """
        dsl_string = dsl_string.strip()
        
        # Extract task
        if dsl_string.startswith("STATE_PREP"):
            task = "state_prep"
            # Extract target state
            state_match = re.search(r'\|[^>]+\>', dsl_string)
            if state_match:
                state_str = state_match.group(0)
                target_state = self.supported_states.get(state_str)
            else:
                target_state = self.supported_states["|+x>"]
            
            # Extract cost weights
            cost_weights = {"backaction": 0.5, "time": 0.2}
            if "WEIGHTED" in dsl_string:
                weight_pattern = r'(\d+\.?\d*)\s*\*\s*(\w+)'
                matches = re.findall(weight_pattern, dsl_string)
                for weight, resource in matches:
                    cost_weights[resource.lower()] = float(weight)
            
            # Extract constraints
            constraints = {}
            if "BUDGET TIME" in dsl_string:
                time_match = re.search(r'BUDGET TIME (\d+\.?\d*)([µu]?s)', dsl_string)
                if time_match:
                    time_val = float(time_match.group(1))
                    unit = time_match.group(2)
                    if unit in ['µs', 'us']:
                        time_val *= 1e-6
                    constraints["budget_time_us"] = time_val * 1e6
            
            return ParsedObjective(
                task=task,
                target_state=target_state,
                constraints=constraints,
                cost_weights=cost_weights
            )
        
        elif dsl_string.startswith("ENTANGLEMENT_MAX"):
            task = "entanglement_max"
            monotone = "concurrence"  # default
            if "concurrence" in dsl_string.lower():
                monotone = "concurrence"
            elif "negativity" in dsl_string.lower():
                monotone = "negativity"
            
            constraints = {}
            if "UNDER LOCC" in dsl_string:
                constraints["locc"] = True
            
            return ParsedObjective(
                task=task,
                monotone=monotone,
                constraints=constraints
            )
        
        elif dsl_string.startswith("PHASE_EST"):
            task = "phase_est"
            constraints = {}
            # Extract probe budget
            probe_match = re.search(r'BUDGET PROBES (\d+)', dsl_string)
            if probe_match:
                constraints["budget_probes"] = int(probe_match.group(1))
            
            return ParsedObjective(task=task, constraints=constraints)
        
        else:
            raise ValueError(f"Unknown DSL command: {dsl_string}")
    
    def parse_json(self, json_data: Union[str, dict]) -> ParsedObjective:
        """
        Parse JSON specification into ParsedObjective.
        
        Args:
            json_data: JSON string or dictionary
        
        Returns:
            ParsedObjective
        """
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        task = data.get("task")
        if task not in self.supported_tasks:
            raise ValueError(f"Unsupported task: {task}")
        
        target_state = None
        if "target_state" in data:
            state_str = data["target_state"]
            target_state = self.supported_states.get(state_str)
        
        monotone = data.get("monotone", "concurrence")
        constraints = data.get("constraints", {})
        cost_weights = data.get("cost_weights", {"backaction": 0.5, "time": 0.2})
        
        return ParsedObjective(
            task=task,
            target_state=target_state,
            monotone=monotone,
            constraints=constraints,
            cost_weights=cost_weights
        )
    
    def compile(self, objective: Union[str, dict, ParsedObjective]) -> Tuple[Callable, Dict]:
        """
        Compile objective into (reward_function, constraints).
        
        Args:
            objective: DSL string, JSON dict, or ParsedObjective
        
        Returns:
            (reward_fn, constraints_dict)
        """
        # Parse if needed
        if isinstance(objective, str):
            if objective.strip().startswith("{"):
                parsed = self.parse_json(objective)
            else:
                parsed = self.parse_dsl(objective)
        elif isinstance(objective, dict):
            parsed = self.parse_json(objective)
        else:
            parsed = objective
        
        # Compile reward function
        if parsed.task == "state_prep":
            reward_fn = self._compile_state_prep_reward(parsed)
        elif parsed.task == "entanglement_max":
            reward_fn = self._compile_entanglement_reward(parsed)
        elif parsed.task == "phase_est":
            reward_fn = self._compile_phase_est_reward(parsed)
        elif parsed.task == "zeno":
            reward_fn = self._compile_zeno_reward(parsed)
        else:
            raise ValueError(f"Unknown task: {parsed.task}")
        
        # Compile constraints
        constraints = self._compile_constraints(parsed)
        
        return reward_fn, constraints
    
    def _compile_state_prep_reward(self, objective: ParsedObjective) -> Callable:
        """
        Compile state preparation reward.
        
        R = F(ρ_T, |ψ_target>) - λ_BA * C_backaction - λ_time * T_wall
        """
        target_state = objective.target_state
        weights = objective.cost_weights
        
        def reward_fn(state, history):
            from .quantum_state import QuantumState
            
            if isinstance(state, QuantumState):
                fidelity = state.fidelity(target_state)
            else:
                # Assume state is already fidelity
                fidelity = state
            
            # Compute costs
            backaction_cost = sum(
                instr.strength if hasattr(instr, 'strength') else 0
                for instr, _ in history
            )
            
            time_cost = sum(
                instr.duration if hasattr(instr, 'duration') else 1e-6
                for instr, _ in history
            )
            
            reward = fidelity - weights.get("backaction", 0) * backaction_cost - \
                     weights.get("time", 0) * time_cost * 1e6  # Convert to µs scale
            
            return reward
        
        return reward_fn
    
    def _compile_entanglement_reward(self, objective: ParsedObjective) -> Callable:
        """Compile entanglement maximization reward."""
        def reward_fn(state, history):
            # Placeholder: compute entanglement monotone
            # Real implementation needs concurrence/negativity calculation
            return 0.0
        return reward_fn
    
    def _compile_phase_est_reward(self, objective: ParsedObjective) -> Callable:
        """Compile phase estimation reward (quantum Fisher information)."""
        def reward_fn(state, history):
            # Placeholder: compute QFI
            return 0.0
        return reward_fn
    
    def _compile_zeno_reward(self, objective: ParsedObjective) -> Callable:
        """Compile Zeno stabilization reward."""
        def reward_fn(state, history):
            # Reward is survival probability in target subspace
            # Placeholder
            return 0.0
        return reward_fn
    
    def _compile_constraints(self, objective: ParsedObjective) -> Dict:
        """Compile safety constraint set C."""
        constraints = {
            "max_strength": 1.0,
            "min_wait": 0.1e-6,  # 100ns minimum
            "max_rate": 10e6,    # 10MHz maximum measurement rate
        }
        
        # Add objective-specific constraints
        if "budget_time_us" in objective.constraints:
            constraints["budget_time_us"] = objective.constraints["budget_time_us"]
        
        if "locc" in objective.constraints:
            constraints["locc_only"] = True
        
        return constraints


# Example usage
def example_intent_parsing():
    """Demonstrate intent parsing."""
    parser = ObjectiveParser()
    
    # DSL example
    dsl = "STATE_PREP |+x> WITH_COST WEIGHTED 0.5*BACKACTION + 0.2*TIME BUDGET TIME 10µs"
    reward_fn, constraints = parser.compile(dsl)
    print(f"Parsed DSL: {constraints}")
    
    # JSON example
    json_obj = {
        "task": "state_prep",
        "target_state": "|+x>",
        "constraints": {"budget_time_us": 10.0},
        "cost_weights": {"backaction": 0.5, "time": 0.2}
    }
    reward_fn, constraints = parser.compile(json_obj)
    print(f"Parsed JSON: {constraints}")
