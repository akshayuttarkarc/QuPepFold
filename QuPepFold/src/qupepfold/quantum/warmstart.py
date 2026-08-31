"""Warm-start initialization for quantum circuits.

Two strategies:
1. Basis state: Prepare exact |z0⟩ from annealer result
2. Biased Ry: Prepare superposition biased toward annealer result
"""

import math
from typing import Optional

try:
    from qiskit import QuantumCircuit
except ImportError as e:
    raise ImportError("qiskit is required. Install with: pip install qiskit") from e


def basis_state_init(circuit: QuantumCircuit, bits: str) -> None:
    """Initialize circuit to a basis state by applying X gates.
    
    Modifies circuit in-place. Apply this BEFORE any other gates.
    
    Args:
        circuit: QuantumCircuit to modify.
        bits: Bitstring (big-endian: bit[0] = qubit 0).
        
    Example:
        >>> qc = QuantumCircuit(4)
        >>> basis_state_init(qc, "1010")  # |1,0,1,0⟩
    """
    if len(bits) != circuit.num_qubits:
        raise ValueError(f"bits length {len(bits)} != num_qubits {circuit.num_qubits}")
    
    for i, bit in enumerate(bits):
        if bit == '1':
            circuit.x(i)
        elif bit != '0':
            raise ValueError(f"Invalid bit '{bit}' at position {i}")


def biased_ry_init(
    circuit: QuantumCircuit,
    bits: str,
    bias: float = 0.8,
) -> None:
    """Initialize with biased superposition toward a target state.
    
    Uses Ry rotations such that P(bit=1) = bias for target '1' bits
    and P(bit=1) = 1-bias for target '0' bits.
    
    The angle θ is computed as: θ = 2 * arcsin(sqrt(p))
    where p = probability of measuring |1⟩.
    
    Args:
        circuit: QuantumCircuit to modify.
        bits: Target bitstring (big-endian).
        bias: Probability bias for matching target (0.5 < bias < 1).
        
    Example:
        >>> qc = QuantumCircuit(4)
        >>> biased_ry_init(qc, "1010", bias=0.8)
        # Now measuring qubit 0 gives |1⟩ with P≈0.8
    """
    if len(bits) != circuit.num_qubits:
        raise ValueError(f"bits length {len(bits)} != num_qubits {circuit.num_qubits}")
    
    if not 0 < bias < 1:
        raise ValueError(f"bias must be in (0, 1), got {bias}")
    
    # Precompute angles
    theta_one = 2 * math.asin(math.sqrt(bias))      # P(1) = bias
    theta_zero = 2 * math.asin(math.sqrt(1 - bias)) # P(1) = 1 - bias
    
    for i, bit in enumerate(bits):
        if bit == '1':
            circuit.ry(theta_one, i)
        elif bit == '0':
            circuit.ry(theta_zero, i)
        else:
            raise ValueError(f"Invalid bit '{bit}' at position {i}")


def create_warmstart_circuit(
    n_qubits: int,
    bits: str,
    mode: str = "basis",
    bias: float = 0.8,
) -> QuantumCircuit:
    """Create a circuit with warm-start initialization.
    
    Args:
        n_qubits: Number of qubits.
        bits: Target bitstring.
        mode: 'basis' or 'biased_ry'.
        bias: Bias for biased_ry mode.
        
    Returns:
        Initialized QuantumCircuit.
    """
    qc = QuantumCircuit(n_qubits)
    
    if mode == "basis":
        basis_state_init(qc, bits)
    elif mode == "biased_ry":
        biased_ry_init(qc, bits, bias)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return qc
