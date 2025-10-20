"""
Basic tests for UMM components.
"""

import numpy as np
import pytest

from umm.core import QuantumState, UMMSimulator, IRBuilder
from umm.core.quantum_state import ket_0, ket_plus, pauli_x
from umm.intent import ObjectiveParser


def test_quantum_state_init():
    """Test quantum state initialization."""
    state = QuantumState(2)
    assert state.dim == 2
    assert np.allclose(state.rho[0, 0], 1.0)
    assert state.purity() > 0.99


def test_quantum_state_unitary():
    """Test unitary evolution."""
    state = QuantumState(2)
    X = pauli_x()
    state.apply_unitary(X)
    
    # |0> -> |1>
    assert np.allclose(state.rho[1, 1], 1.0)


def test_fidelity():
    """Test fidelity computation."""
    state = QuantumState(2)
    target = ket_0()
    
    fidelity = state.fidelity(target)
    assert fidelity > 0.99


def test_bloch_vector():
    """Test Bloch vector conversion."""
    psi_plus = ket_plus()
    rho = np.outer(psi_plus, psi_plus.conj())
    state = QuantumState(2, rho)
    
    bloch = state.to_bloch_vector()
    # |+> should be at [1, 0, 0]
    assert np.allclose(bloch, [1, 0, 0], atol=1e-10)


def test_ir_builder():
    """Test IR program builder."""
    builder = IRBuilder()
    program = (builder
               .reset()
               .weak_measure(axis=np.array([1, 0, 0]), strength=0.3)
               .wait(1e-6)
               .build())
    
    assert len(program.instructions) == 3
    assert program.validate()


def test_simulator_basic():
    """Test basic simulator execution."""
    simulator = UMMSimulator(n_qubits=1, T1=50e-6, T2=50e-6)
    
    builder = IRBuilder()
    program = (builder
               .reset()
               .weak_measure(axis=np.array([1, 0, 0]), strength=0.3)
               .wait(1e-6)
               .build())
    
    result = simulator.execute_program(program)
    assert result.final_state is not None
    assert len(result.outcomes) >= 0


def test_intent_parser_dsl():
    """Test DSL parsing."""
    parser = ObjectiveParser()
    
    dsl = "STATE_PREP |+x> WITH_COST WEIGHTED 0.5*BACKACTION + 0.2*TIME"
    parsed = parser.parse_dsl(dsl)
    
    assert parsed.task == "state_prep"
    assert parsed.target_state is not None
    assert parsed.cost_weights["backaction"] == 0.5


def test_intent_parser_json():
    """Test JSON parsing."""
    parser = ObjectiveParser()
    
    json_obj = {
        "task": "state_prep",
        "target_state": "|+x>",
        "constraints": {"budget_time_us": 10.0},
        "cost_weights": {"backaction": 0.5, "time": 0.2}
    }
    
    parsed = parser.parse_json(json_obj)
    assert parsed.task == "state_prep"


def test_reward_compilation():
    """Test reward function compilation."""
    parser = ObjectiveParser()
    
    objective = {
        "task": "state_prep",
        "target_state": "|+x>",
        "cost_weights": {"backaction": 0.5, "time": 0.0}
    }
    
    reward_fn, constraints = parser.compile(objective)
    
    # Test reward function
    from umm.core.quantum_state import ket_plus
    state = QuantumState.from_bloch_vector([1, 0, 0])
    reward = reward_fn(state, [])
    
    assert reward > 0  # High fidelity with |+x> should give positive reward


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
