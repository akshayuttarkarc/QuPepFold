"""Abstract base class for sampler backends.

All backends must implement run_sampler() which takes bound circuits
and returns probability dictionaries.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np

try:
    from qiskit import QuantumCircuit
except ImportError:
    QuantumCircuit = None


class SamplerBackend(ABC):
    """Abstract base class for quantum sampler backends.
    
    Subclasses must implement run_sampler() for executing circuits.
    """
    
    @abstractmethod
    def run_sampler(
        self,
        circuits: List["QuantumCircuit"],
        param_values: List[np.ndarray],
        shots: int,
    ) -> List[Dict[str, float]]:
        """Execute circuits and return probability distributions.
        
        Args:
            circuits: List of parameterized QuantumCircuits.
            param_values: List of parameter value arrays, one per circuit.
            shots: Number of shots per circuit.
            
        Returns:
            List of probability dicts {bitstring: probability}.
            Bitstrings are in big-endian order (qubit 0 is MSB).
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close any open sessions or connections."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def counts_to_probs(counts: Dict[str, int], shots: int) -> Dict[str, float]:
    """Convert raw counts to probability distribution.
    
    Args:
        counts: Dict of {bitstring: count}.
        shots: Total number of shots.
        
    Returns:
        Dict of {bitstring: probability}.
    """
    return {bits: count / shots for bits, count in counts.items()}


def reverse_bitstring_endianness(bitstring: str) -> str:
    """Convert between Qiskit little-endian and our big-endian convention.
    
    Qiskit returns bitstrings as little-endian (qubit 0 is rightmost).
    We use big-endian (qubit 0 is leftmost).
    
    Args:
        bitstring: Input bitstring.
        
    Returns:
        Reversed bitstring.
    """
    return bitstring[::-1]
