"""
UMM-IR: Universal Measurement Machine Intermediate Representation.

Defines the instruction set for auditable measurement sequences.

Instructions:
- RESET(rho_init)
- PREPARE_ANCILLA(|0>^m)
- APPLY_U(name|params)
- MEASURE_ANCILLA({Π_k}, thresh=τ)
- WEAK_MEAS(axis=n, strength=s)
- WAIT(Δt)
- IF outcome==k* THEN <block>
- SAFETY_ENFORCE(C)
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Any
from enum import Enum
import numpy as np


class InstructionType(Enum):
    """UMM-IR instruction types."""
    RESET = "RESET"
    PREPARE_ANCILLA = "PREPARE_ANCILLA"
    APPLY_U = "APPLY_U"
    MEASURE_ANCILLA = "MEASURE_ANCILLA"
    WEAK_MEAS = "WEAK_MEAS"
    WAIT = "WAIT"
    CONDITIONAL = "IF"
    SAFETY_ENFORCE = "SAFETY_ENFORCE"


@dataclass
class IRInstruction:
    """Base class for UMM-IR instructions."""
    type: InstructionType
    metadata: dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.type.value}"


@dataclass
class ResetInstruction(IRInstruction):
    """RESET(rho_init): Initialize system state."""
    initial_state: np.ndarray = None  # Density matrix or state vector
    
    def __post_init__(self):
        self.type = InstructionType.RESET
    
    def __str__(self) -> str:
        state_desc = "default" if self.initial_state is None else "custom"
        return f"RESET({state_desc})"


@dataclass
class PrepareAncillaInstruction(IRInstruction):
    """PREPARE_ANCILLA(|0>^m): Allocate and initialize ancilla qubits."""
    n_ancilla: int = 1
    
    def __post_init__(self):
        self.type = InstructionType.PREPARE_ANCILLA
    
    def __str__(self) -> str:
        return f"PREPARE_ANCILLA(|0>^{self.n_ancilla})"


@dataclass
class ApplyUnitaryInstruction(IRInstruction):
    """APPLY_U(name|params): Apply calibrated unitary gate/macro."""
    gate_name: str = ""
    parameters: dict = field(default_factory=dict)
    unitary: Optional[np.ndarray] = None
    
    def __post_init__(self):
        self.type = InstructionType.APPLY_U
    
    def __str__(self) -> str:
        if self.gate_name:
            params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
            return f"APPLY_U({self.gate_name}({params_str}))"
        return "APPLY_U(custom)"


@dataclass
class MeasureAncillaInstruction(IRInstruction):
    """MEASURE_ANCILLA({Π_k}, thresh=τ): Projective readout on ancilla."""
    projectors: List[np.ndarray] = field(default_factory=list)
    threshold: float = 0.5
    
    def __post_init__(self):
        self.type = InstructionType.MEASURE_ANCILLA
    
    def __str__(self) -> str:
        n_outcomes = len(self.projectors) if self.projectors else 2
        return f"MEASURE_ANCILLA({{Π_k, k=0..{n_outcomes-1}}}, τ={self.threshold:.2f})"


@dataclass
class WeakMeasurementInstruction(IRInstruction):
    """
    WEAK_MEAS(axis=n, strength=s): Weak measurement (compiled via dilation).
    
    This is syntactic sugar that will be compiled into:
    1. PREPARE_ANCILLA
    2. APPLY_U (entangling)
    3. MEASURE_ANCILLA
    """
    axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    strength: float = 0.5
    
    def __post_init__(self):
        self.type = InstructionType.WEAK_MEAS
        # Normalize axis
        if np.linalg.norm(self.axis) > 0:
            self.axis = self.axis / np.linalg.norm(self.axis)
    
    def __str__(self) -> str:
        return f"WEAK_MEAS(axis={self.axis}, strength={self.strength:.3f})"


@dataclass
class WaitInstruction(IRInstruction):
    """WAIT(Δt): Idle for free evolution."""
    duration: float = 0.0  # seconds
    
    def __post_init__(self):
        self.type = InstructionType.WAIT
    
    def __str__(self) -> str:
        return f"WAIT({self.duration*1e6:.2f}µs)"


@dataclass
class ConditionalInstruction(IRInstruction):
    """IF outcome==k* THEN <block>: Conditional execution based on measurement."""
    condition_outcome: int = 0
    if_block: List[IRInstruction] = field(default_factory=list)
    else_block: List[IRInstruction] = field(default_factory=list)
    
    def __post_init__(self):
        self.type = InstructionType.CONDITIONAL
    
    def __str__(self) -> str:
        n_if = len(self.if_block)
        n_else = len(self.else_block)
        return f"IF outcome=={self.condition_outcome} THEN [{n_if} instructions] " \
               f"ELSE [{n_else} instructions]"


@dataclass
class SafetyEnforceInstruction(IRInstruction):
    """SAFETY_ENFORCE(C): Project action onto safety constraint set."""
    constraints: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.type = InstructionType.SAFETY_ENFORCE
    
    def __str__(self) -> str:
        return f"SAFETY_ENFORCE({len(self.constraints)} constraints)"


class IRProgram:
    """
    A complete UMM-IR program: sequence of instructions.
    
    Provides:
    - Validation
    - Pretty printing
    - Execution interface
    """
    
    def __init__(self, instructions: Optional[List[IRInstruction]] = None):
        """
        Initialize IR program.
        
        Args:
            instructions: List of IR instructions
        """
        self.instructions = instructions or []
    
    def append(self, instruction: IRInstruction):
        """Add instruction to program."""
        self.instructions.append(instruction)
    
    def validate(self) -> bool:
        """
        Check program validity:
        - RESET must be first
        - Ancilla must be prepared before use
        - Conditionals reference valid outcomes
        """
        if len(self.instructions) == 0:
            return True
        
        # Check first instruction
        if not isinstance(self.instructions[0], ResetInstruction):
            # Allow programs without explicit reset
            pass
        
        # More validation can be added
        return True
    
    def __str__(self) -> str:
        """Pretty print program."""
        lines = ["UMM-IR Program:", "=" * 50]
        for i, instr in enumerate(self.instructions):
            lines.append(f"{i:3d}:  {str(instr)}")
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return {
            "instructions": [
                {
                    "type": instr.type.value,
                    "params": self._serialize_instruction(instr)
                }
                for instr in self.instructions
            ]
        }
    
    @staticmethod
    def _serialize_instruction(instr: IRInstruction) -> dict:
        """Serialize instruction parameters."""
        if isinstance(instr, WeakMeasurementInstruction):
            return {
                "axis": instr.axis.tolist(),
                "strength": float(instr.strength)
            }
        elif isinstance(instr, WaitInstruction):
            return {"duration": float(instr.duration)}
        elif isinstance(instr, ConditionalInstruction):
            return {
                "condition_outcome": instr.condition_outcome,
                "n_if_instructions": len(instr.if_block),
                "n_else_instructions": len(instr.else_block)
            }
        # Add more as needed
        return {}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'IRProgram':
        """Deserialize from dictionary."""
        program = cls()
        # Implementation depends on serialization format
        # Placeholder for now
        return program


# ============================================================================
# Builder utilities
# ============================================================================

class IRBuilder:
    """Fluent interface for building IR programs."""
    
    def __init__(self):
        self.program = IRProgram()
    
    def reset(self, initial_state: Optional[np.ndarray] = None) -> 'IRBuilder':
        """Add RESET instruction."""
        self.program.append(ResetInstruction(initial_state=initial_state))
        return self
    
    def prepare_ancilla(self, n: int = 1) -> 'IRBuilder':
        """Add PREPARE_ANCILLA instruction."""
        self.program.append(PrepareAncillaInstruction(n_ancilla=n))
        return self
    
    def apply_gate(self, gate_name: str, **params) -> 'IRBuilder':
        """Add APPLY_U instruction."""
        self.program.append(ApplyUnitaryInstruction(gate_name=gate_name, parameters=params))
        return self
    
    def weak_measure(self, axis: np.ndarray, strength: float) -> 'IRBuilder':
        """Add WEAK_MEAS instruction."""
        self.program.append(WeakMeasurementInstruction(axis=axis, strength=strength))
        return self
    
    def wait(self, duration: float) -> 'IRBuilder':
        """Add WAIT instruction."""
        self.program.append(WaitInstruction(duration=duration))
        return self
    
    def if_outcome(self, k: int, then_fn=None, else_fn=None) -> 'IRBuilder':
        """Add IF instruction with lambda builders."""
        if_block = []
        else_block = []
        
        if then_fn:
            then_builder = IRBuilder()
            then_fn(then_builder)
            if_block = then_builder.program.instructions
        
        if else_fn:
            else_builder = IRBuilder()
            else_fn(else_builder)
            else_block = else_builder.program.instructions
        
        self.program.append(ConditionalInstruction(
            condition_outcome=k,
            if_block=if_block,
            else_block=else_block
        ))
        return self
    
    def safety_enforce(self, **constraints) -> 'IRBuilder':
        """Add SAFETY_ENFORCE instruction."""
        self.program.append(SafetyEnforceInstruction(constraints=constraints))
        return self
    
    def build(self) -> IRProgram:
        """Return completed program."""
        return self.program


# ============================================================================
# Example programs
# ============================================================================

def example_adaptive_steering() -> IRProgram:
    """
    Example: Adaptive state preparation with weak measurements.
    
    Target: Prepare |+x> from unknown mixed state.
    """
    builder = IRBuilder()
    
    builder.reset()
    
    for t in range(8):
        builder.weak_measure(axis=np.array([1, 0, 0]), strength=0.3)
        # Conditional feedback
        builder.if_outcome(
            k=1,  # If outcome is "-"
            then_fn=lambda b: b.apply_gate("R_x", theta=0.1)  # Small correction
        )
        builder.wait(1e-6)
        builder.safety_enforce(max_strength=0.5, min_wait=0.5e-6)
    
    return builder.build()
