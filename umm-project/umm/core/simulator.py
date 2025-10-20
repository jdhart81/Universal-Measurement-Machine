"""
UMM Simulator: Executes UMM-IR programs on quantum states.

Orchestrates:
- State evolution
- Instrument implementation
- Noise models
- Measurement outcomes
- History tracking
"""

import numpy as np
from typing import Optional, Callable, List, Tuple, Dict, Any
from dataclasses import dataclass, field

from .quantum_state import QuantumState, NoiseParams, ket_0
from .instruments import (
    QuantumInstrument,
    weak_measurement_via_dilation,
    projective_measurement
)
from .ir import (
    IRProgram,
    IRInstruction,
    InstructionType,
    ResetInstruction,
    WeakMeasurementInstruction,
    WaitInstruction,
    ConditionalInstruction,
    ApplyUnitaryInstruction
)


@dataclass
class SimulationResult:
    """Results from UMM simulation run."""
    final_state: QuantumState
    history: List[Tuple[IRInstruction, int]]  # (instruction, outcome)
    outcomes: List[int]
    fidelity: float
    purity: float
    backaction: float
    wall_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class UMMSimulator:
    """
    Universal Measurement Machine simulator.
    
    Executes UMM-IR programs with realistic noise models and tracks
    measurement history for policy learning.
    """
    
    def __init__(
        self,
        n_qubits: int = 1,
        T1: float = 50e-6,
        T2: float = 50e-6,
        readout_error: float = 0.01,
        gate_error: float = 0.001,
        seed: Optional[int] = None
    ):
        """
        Initialize simulator.
        
        Args:
            n_qubits: Number of system qubits
            T1: Amplitude damping time
            T2: Dephasing time
            readout_error: SPAM error probability
            gate_error: Single-qubit gate error rate
            seed: Random seed for reproducibility
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        
        self.noise_params = NoiseParams(
            T1=T1,
            T2=T2,
            readout_error=readout_error,
            gate_error=gate_error
        )
        
        self.rng = np.random.default_rng(seed)
        
        # State
        self.state: Optional[QuantumState] = None
        
        # History tracking
        self.history: List[Tuple[IRInstruction, Optional[int]]] = []
        self.wall_time: float = 0.0
    
    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset simulator to initial state."""
        if initial_state is None:
            # Default to |0...0>
            psi_0 = np.zeros(self.dim, dtype=complex)
            psi_0[0] = 1.0
            rho_0 = np.outer(psi_0, psi_0.conj())
        else:
            if initial_state.ndim == 1:
                # State vector
                rho_0 = np.outer(initial_state, initial_state.conj())
            else:
                # Density matrix
                rho_0 = initial_state
        
        self.state = QuantumState(self.dim, rho_0)
        self.history = []
        self.wall_time = 0.0
    
    def execute_program(
        self,
        program: IRProgram,
        target_state: Optional[np.ndarray] = None
    ) -> SimulationResult:
        """
        Execute a complete IR program.
        
        Args:
            program: UMM-IR program to execute
            target_state: Optional target state for fidelity calculation
        
        Returns:
            SimulationResult with final state and metrics
        """
        # Validate and reset
        program.validate()
        self.reset()
        
        # Execute instructions sequentially
        for instruction in program.instructions:
            self._execute_instruction(instruction)
        
        # Compute final metrics
        fidelity = 0.0
        if target_state is not None:
            fidelity = self.state.fidelity(target_state)
        
        purity = self.state.purity()
        
        # Estimate back-action cost (sum of measurement strengths)
        backaction = sum(
            instr.strength if isinstance(instr, WeakMeasurementInstruction) else 0.0
            for instr, _ in self.history
        )
        
        # Extract outcomes
        outcomes = [
            outcome for _, outcome in self.history
            if outcome is not None
        ]
        
        return SimulationResult(
            final_state=self.state,
            history=self.history,
            outcomes=outcomes,
            fidelity=fidelity,
            purity=purity,
            backaction=backaction,
            wall_time=self.wall_time
        )
    
    def _execute_instruction(self, instruction: IRInstruction):
        """Execute a single IR instruction."""
        
        if isinstance(instruction, ResetInstruction):
            self.reset(instruction.initial_state)
            self.history.append((instruction, None))
        
        elif isinstance(instruction, WeakMeasurementInstruction):
            # Compile weak measurement via ancilla dilation
            outcome, post_state = weak_measurement_via_dilation(
                system_state=self.state,
                axis=instruction.axis,
                strength=instruction.strength,
                noise_params=self.noise_params,
                rng=self.rng
            )
            self.state = post_state
            self.history.append((instruction, outcome))
            
            # Timing estimate: 1µs per weak measurement
            self.wall_time += 1e-6
        
        elif isinstance(instruction, WaitInstruction):
            # Apply noise evolution
            self.state.evolve_noise(instruction.duration, self.noise_params)
            self.wall_time += instruction.duration
            self.history.append((instruction, None))
        
        elif isinstance(instruction, ApplyUnitaryInstruction):
            # Apply unitary gate
            if instruction.unitary is not None:
                self.state.apply_unitary(instruction.unitary, self.noise_params)
            elif instruction.gate_name:
                U = self._get_gate(instruction.gate_name, instruction.parameters)
                self.state.apply_unitary(U, self.noise_params)
            
            self.history.append((instruction, None))
            self.wall_time += 40e-9  # Typical single-qubit gate time
        
        elif isinstance(instruction, ConditionalInstruction):
            # Execute conditional based on last outcome
            if len(self.history) > 0 and self.history[-1][1] is not None:
                last_outcome = self.history[-1][1]
                if last_outcome == instruction.condition_outcome:
                    for sub_instr in instruction.if_block:
                        self._execute_instruction(sub_instr)
                else:
                    for sub_instr in instruction.else_block:
                        self._execute_instruction(sub_instr)
            
            self.history.append((instruction, None))
        
        else:
            # Unimplemented instruction
            self.history.append((instruction, None))
    
    def _get_gate(self, gate_name: str, parameters: dict) -> np.ndarray:
        """Get unitary matrix for named gate."""
        from .quantum_state import rotation_x, rotation_y, rotation_z, pauli_x, pauli_y, pauli_z
        
        if gate_name == "R_x":
            return rotation_x(parameters.get("theta", 0.0))
        elif gate_name == "R_y":
            return rotation_y(parameters.get("theta", 0.0))
        elif gate_name == "R_z":
            return rotation_z(parameters.get("theta", 0.0))
        elif gate_name == "X":
            return pauli_x()
        elif gate_name == "Y":
            return pauli_y()
        elif gate_name == "Z":
            return pauli_z()
        else:
            raise ValueError(f"Unknown gate: {gate_name}")
    
    def run_adaptive(
        self,
        policy: Callable,
        reward_fn: Callable,
        constraints: dict,
        max_steps: int = 32,
        target_state: Optional[np.ndarray] = None
    ) -> SimulationResult:
        """
        Run adaptive measurement sequence with a learned policy.
        
        Args:
            policy: Policy function (history) -> action
            reward_fn: Reward function (state, history) -> float
            constraints: Safety constraints
            max_steps: Maximum number of adaptive steps
            target_state: Optional target for fidelity computation
        
        Returns:
            SimulationResult
        """
        self.reset()
        
        for t in range(max_steps):
            # Policy selects action
            action = policy(self.history, self.state)
            
            # TODO: Apply safety projection onto constraints
            
            # Execute action (assume it's a WeakMeasurementInstruction)
            self._execute_instruction(action)
        
        # Compute reward
        reward = reward_fn(self.state, self.history)
        
        # Metrics
        fidelity = 0.0
        if target_state is not None:
            fidelity = self.state.fidelity(target_state)
        
        outcomes = [outcome for _, outcome in self.history if outcome is not None]
        
        return SimulationResult(
            final_state=self.state,
            history=self.history,
            outcomes=outcomes,
            fidelity=fidelity,
            purity=self.state.purity(),
            backaction=sum(
                instr.strength if isinstance(instr, WeakMeasurementInstruction) else 0
                for instr, _ in self.history
            ),
            wall_time=self.wall_time,
            metadata={"reward": reward}
        )
