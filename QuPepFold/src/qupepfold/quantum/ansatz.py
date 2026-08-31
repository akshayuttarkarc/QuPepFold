"""Shallow parameterized ansatz for fragment optimization.

Designed for 10-qubit fragments (6 AA, 5 turns) on NISQ hardware.
Keep depth minimal to reduce noise accumulation.
"""

from typing import Optional
import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
except ImportError as e:
    raise ImportError("qiskit is required. Install with: pip install qiskit") from e


def build_fragment_ansatz(
    n_qubits: int,
    depth: int = 2,
    with_warmstart: bool = False,
    warmstart_bits: Optional[str] = None,
) -> QuantumCircuit:
    """Build a shallow parameterized ansatz for fragment optimization.
    
    Architecture:
    1. Optional warm-start layer (X gates for |1⟩ positions)
    2. Initial Hadamard layer (superposition)
    3. Repeated mixing layers:
       - Ry rotation on each qubit (parameterized)
       - Entangling CX chain
    
    Args:
        n_qubits: Number of qubits (typically 10 for 6-AA fragment).
        depth: Number of mixing layers.
        with_warmstart: If True, initialize based on warmstart_bits.
        warmstart_bits: Bitstring for warm-start (length n_qubits).
        
    Returns:
        Parameterized QuantumCircuit.
        
    Example:
        >>> qc = build_fragment_ansatz(10, depth=2)
        >>> qc.num_parameters
        20  # 10 qubits * 2 depths
    """
    qc = QuantumCircuit(n_qubits)
    
    # Warm-start initialization
    if with_warmstart and warmstart_bits:
        if len(warmstart_bits) != n_qubits:
            raise ValueError(f"warmstart_bits length {len(warmstart_bits)} != n_qubits {n_qubits}")
        for i, bit in enumerate(warmstart_bits):
            if bit == '1':
                qc.x(i)
    
    # Initial superposition (skip if warm-starting with basis state)
    if not with_warmstart:
        for i in range(n_qubits):
            qc.h(i)
    
    # Parameter vector for all Ry gates
    n_params = n_qubits * depth
    theta = ParameterVector('θ', n_params)
    param_idx = 0
    
    for d in range(depth):
        # Ry rotation layer
        for i in range(n_qubits):
            qc.ry(theta[param_idx], i)
            param_idx += 1
        
        # Entangling layer: linear CX chain
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        
        # Close the chain for last layer
        if d == depth - 1 and n_qubits > 2:
            qc.cx(n_qubits - 1, 0)
    
    return qc


def build_hardware_efficient_ansatz(
    n_qubits: int,
    depth: int = 1,
) -> QuantumCircuit:
    """Build a hardware-efficient ansatz suitable for real QPUs.
    
    Uses only Ry, Rz, and CX gates which are native on most hardware.
    
    Args:
        n_qubits: Number of qubits.
        depth: Number of layers.
        
    Returns:
        Parameterized QuantumCircuit.
    """
    qc = QuantumCircuit(n_qubits)
    
    # Parameters: 2 angles per qubit per layer (Ry + Rz)
    n_params = 2 * n_qubits * depth
    theta = ParameterVector('θ', n_params)
    param_idx = 0
    
    for d in range(depth):
        # Single qubit rotations
        for i in range(n_qubits):
            qc.ry(theta[param_idx], i)
            param_idx += 1
            qc.rz(theta[param_idx], i)
            param_idx += 1
        
        # Entangling: alternating CX pattern
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            qc.cx(i, i + 1)
    
    return qc


def count_parameters(circuit: QuantumCircuit) -> int:
    """Count the number of parameters in a circuit."""
    return circuit.num_parameters


def get_initial_parameters(
    n_params: int,
    strategy: str = "random",
    seed: int = 42,
) -> np.ndarray:
    """Generate initial parameter values.
    
    Args:
        n_params: Number of parameters.
        strategy: 'random', 'zeros', or 'small'.
        seed: Random seed.
        
    Returns:
        Parameter array.
    """
    rng = np.random.default_rng(seed)
    
    if strategy == "random":
        return rng.uniform(-np.pi, np.pi, size=n_params)
    elif strategy == "zeros":
        return np.zeros(n_params)
    elif strategy == "small":
        return rng.uniform(-0.1, 0.1, size=n_params)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
