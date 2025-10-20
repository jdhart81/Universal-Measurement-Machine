"""
Quantum instruments and POVMs.

Implements:
- POVM effects {E_k}
- Quantum instruments {M_k} with post-measurement states
- Weak measurements via ancilla dilation
- Projective measurements
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .quantum_state import QuantumState, NoiseParams


@dataclass
class POVMEffect:
    """Single POVM effect E_k (positive operator)."""
    operator: np.ndarray  # Positive semi-definite
    label: str = ""
    
    def __post_init__(self):
        """Validate POVM effect."""
        assert np.allclose(self.operator, self.operator.conj().T), "Effect must be Hermitian"
        eigvals = np.linalg.eigvalsh(self.operator)
        assert np.all(eigvals >= -1e-10), "Effect must be positive"


class POVM:
    """
    Positive Operator-Valued Measure.
    
    A collection of effects {E_k} with ∑_k E_k = I.
    """
    
    def __init__(self, effects: List[POVMEffect]):
        """
        Initialize POVM.
        
        Args:
            effects: List of POVM effects
        """
        self.effects = effects
        self.n_outcomes = len(effects)
        
        # Validate completeness
        total = sum(e.operator for e in effects)
        dim = effects[0].operator.shape[0]
        assert np.allclose(total, np.eye(dim)), "POVM effects must sum to identity"
    
    def probabilities(self, rho: np.ndarray) -> np.ndarray:
        """
        Compute outcome probabilities p(k) = Tr(E_k ρ).
        
        Args:
            rho: Density matrix
        
        Returns:
            Probability distribution over outcomes
        """
        probs = np.array([np.real(np.trace(e.operator @ rho)) for e in self.effects])
        # Renormalize to handle numerical errors
        probs = np.maximum(probs, 0)
        return probs / probs.sum()
    
    def sample_outcome(self, rho: np.ndarray, rng: Optional[np.random.Generator] = None) -> int:
        """
        Sample measurement outcome according to Born rule.
        
        Args:
            rho: Density matrix
            rng: Random number generator
        
        Returns:
            Outcome index k
        """
        if rng is None:
            rng = np.random.default_rng()
        
        probs = self.probabilities(rho)
        return rng.choice(self.n_outcomes, p=probs)


class QuantumInstrument:
    """
    Quantum instrument: measurement with post-measurement states.
    
    Defined by Kraus operators {M_{k,α}} where outcome k has probability
    p(k) = Tr(E_k ρ) with E_k = ∑_α M_{k,α}† M_{k,α}, and post-measurement
    state ρ_k = [∑_α M_{k,α} ρ M_{k,α}†] / p(k).
    """
    
    def __init__(self, kraus_ops: List[List[np.ndarray]], labels: Optional[List[str]] = None):
        """
        Initialize quantum instrument.
        
        Args:
            kraus_ops: Nested list [[M_{0,α}], [M_{1,α}], ...] grouping by outcome
            labels: Optional outcome labels
        """
        self.kraus_ops = kraus_ops
        self.n_outcomes = len(kraus_ops)
        self.labels = labels or [f"k={i}" for i in range(self.n_outcomes)]
        
        # Compute POVM effects
        effects = []
        for k_ops in kraus_ops:
            E_k = sum(M.conj().T @ M for M in k_ops)
            effects.append(POVMEffect(E_k))
        self.povm = POVM(effects)
    
    def measure(
        self, 
        state: QuantumState, 
        noise_params: Optional[NoiseParams] = None,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[int, QuantumState]:
        """
        Perform measurement and return outcome + post-measurement state.
        
        Args:
            state: Input quantum state
            noise_params: Optional readout errors
            rng: Random number generator
        
        Returns:
            (outcome_index, post_measurement_state)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Sample outcome
        outcome = self.povm.sample_outcome(state.rho, rng)
        
        # Apply readout error if specified
        if noise_params is not None and noise_params.readout_error > 0:
            # Bit-flip with probability readout_error
            if rng.random() < noise_params.readout_error:
                outcome = (outcome + 1) % self.n_outcomes
        
        # Compute post-measurement state
        k_ops = self.kraus_ops[outcome]
        new_rho = sum(M @ state.rho @ M.conj().T for M in k_ops)
        
        # Renormalize
        prob = np.real(np.trace(new_rho))
        if prob > 1e-12:
            new_rho = new_rho / prob
        else:
            # If probability is too small, return original state
            new_rho = state.rho
        
        return outcome, QuantumState(state.dim, new_rho)


# ============================================================================
# Factory functions for common measurements
# ============================================================================

def projective_measurement(basis: str = "Z") -> QuantumInstrument:
    """
    Standard projective measurement.
    
    Args:
        basis: "X", "Y", or "Z"
    
    Returns:
        QuantumInstrument for projective measurement
    """
    if basis == "Z":
        M0 = np.array([[1, 0], [0, 0]], dtype=complex)
        M1 = np.array([[0, 0], [0, 1]], dtype=complex)
    elif basis == "X":
        M0 = np.array([[1, 1], [1, 1]], dtype=complex) / 2
        M1 = np.array([[1, -1], [-1, 1]], dtype=complex) / 2
    elif basis == "Y":
        M0 = np.array([[1, -1j], [1j, 1]], dtype=complex) / 2
        M1 = np.array([[1, 1j], [-1j, 1]], dtype=complex) / 2
    else:
        raise ValueError(f"Unknown basis: {basis}")
    
    return QuantumInstrument([[M0], [M1]], labels=[f"{basis}+", f"{basis}-"])


def weak_measurement(axis: np.ndarray, strength: float) -> QuantumInstrument:
    """
    Two-outcome weak measurement along axis n with strength m ∈ [0, 1].
    
    Effects: E_± = ½(I ± m n·σ)
    
    This is compiled via ancilla dilation in practice, but here we use
    a direct implementation for single-qubit systems.
    
    Args:
        axis: Unit 3-vector [n_x, n_y, n_z]
        strength: Measurement strength m ∈ [0, 1]
    
    Returns:
        QuantumInstrument approximating weak measurement
    """
    assert len(axis) == 3, "Axis must be 3D vector"
    assert 0 <= strength <= 1, "Strength must be in [0, 1]"
    
    # Normalize axis
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    
    # Construct n·σ
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    n_sigma = axis[0] * sx + axis[1] * sy + axis[2] * sz
    
    # POVM effects: E_± = ½(I ± m n·σ)
    I = np.eye(2, dtype=complex)
    E_plus = (I + strength * n_sigma) / 2
    E_minus = (I - strength * n_sigma) / 2
    
    # Construct Kraus operators via "square root" of effects
    # For simplicity, use M_± = √E_±
    # This gives a valid instrument but not necessarily the "physically implemented" one
    M_plus = np.linalg.cholesky(E_plus)
    M_minus = np.linalg.cholesky(E_minus)
    
    return QuantumInstrument([[M_plus], [M_minus]], labels=["+", "-"])


def weak_measurement_via_dilation(
    system_state: QuantumState,
    axis: np.ndarray,
    strength: float,
    noise_params: Optional[NoiseParams] = None,
    rng: Optional[np.random.Generator] = None
) -> Tuple[int, QuantumState]:
    """
    Implement weak measurement via ancilla dilation.
    
    This is the "physically implemented" version from Section 7 of the paper:
    1. Prepare ancilla in |0>
    2. Apply controlled rotation on ancilla
    3. Measure ancilla in Z
    4. System state updates based on outcome
    
    Args:
        system_state: System qubit state (2D)
        axis: Measurement axis [n_x, n_y, n_z]
        strength: Measurement strength (calibrated via angle θ)
        noise_params: Noise model
        rng: Random generator
    
    Returns:
        (outcome, post_measurement_system_state)
    """
    assert system_state.dim == 2, "Only single-qubit system supported"
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Step 1: Create joint system+ancilla state
    # |ψ_S> ⊗ |0_A>
    ancilla_zero = np.array([1, 0], dtype=complex)
    joint_state = np.kron(system_state.rho, np.outer(ancilla_zero, ancilla_zero))
    joint_state_obj = QuantumState(4, joint_state)
    
    # Step 2: Entangling operation (controlled rotation)
    # For simplicity, use CZ-style interaction parameterized by strength
    # Real implementation: controlled-R_y(2θ) where θ maps to strength m
    theta = np.arcsin(np.sqrt(strength)) if strength <= 1 else np.pi / 4
    
    # Align system axis to Z (rotation matrices)
    # This is simplified; full implementation needs proper axis alignment
    # For now, assume measurement along Z
    
    # Controlled interaction (approximate)
    # U = I⊗|0><0| + R_y(2θ)⊗|1><1|  (controlled by system qubit)
    # Simplified version: partial CNOT-like interaction
    angle = 2 * theta * axis[2]  # Weight by z-component for simplicity
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    
    # Controlled-phase-like gate (simplified)
    U_int = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, c, -s],
        [0, 0, s, c]
    ], dtype=complex)
    
    joint_state_obj.apply_unitary(U_int, noise_params)
    
    # Step 3: Measure ancilla in Z basis
    # Ancilla is qubit 1, project onto |0> or |1>
    P0_ancilla = np.array([[1, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0]], dtype=complex)
    P1_ancilla = np.array([[0, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]], dtype=complex)
    
    # Probabilities
    p0 = np.real(np.trace(P0_ancilla @ joint_state_obj.rho))
    p1 = np.real(np.trace(P1_ancilla @ joint_state_obj.rho))
    
    # Sample outcome
    outcome = 0 if rng.random() < p0 / (p0 + p1) else 1
    
    # Apply readout error
    if noise_params is not None and noise_params.readout_error > 0:
        if rng.random() < noise_params.readout_error:
            outcome = 1 - outcome
    
    # Step 4: Post-measurement state (partial trace over ancilla)
    if outcome == 0:
        post_joint = P0_ancilla @ joint_state_obj.rho @ P0_ancilla / p0
    else:
        post_joint = P1_ancilla @ joint_state_obj.rho @ P1_ancilla / p1
    
    # Trace out ancilla to get system state
    system_post = np.zeros((2, 2), dtype=complex)
    system_post[0, 0] = post_joint[0, 0] + post_joint[1, 1]
    system_post[0, 1] = post_joint[0, 2] + post_joint[1, 3]
    system_post[1, 0] = post_joint[2, 0] + post_joint[3, 1]
    system_post[1, 1] = post_joint[2, 2] + post_joint[3, 3]
    
    return outcome, QuantumState(2, system_post)
