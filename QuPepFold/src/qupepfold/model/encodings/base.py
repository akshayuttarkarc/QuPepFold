"""Abstract base class for encoding schemes.

An EncodingScheme defines how to:
  1. Estimate resource costs (qubits, terms, circuit depth).
  2. Decode a bitstring into a list of turn codes.
  3. Produce a penalty summary for debugging.

Energy tables are built externally (energy_fragment.py) using the scheme's
decode() to convert state indices to turns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EncodingSpec:
    """Resource cost estimate for an encoding of a specific fragment size.

    Attributes:
        n_qubits: Number of qubits required.
        n_terms: Number of Hamiltonian/energy terms.
        circuit_depth_estimate: Estimated ansatz circuit depth (parameterised layers).
        cnot_estimate: Estimated CNOT gate count.
        memory_estimate_mb: Approximate memory for full energy table (2^n_qubits × 8 bytes).
        feasibility: "simulator_ok", "hardware_feasible", or "too_large".
        notes: Optional additional notes.
    """
    n_qubits: int
    n_terms: int
    circuit_depth_estimate: int
    cnot_estimate: int
    memory_estimate_mb: float
    feasibility: str  # "simulator_ok" | "hardware_feasible" | "too_large"
    notes: Optional[str] = None


class EncodingScheme(ABC):
    """Abstract base for encoding schemes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this scheme, e.g. 'turn'."""
        ...

    @abstractmethod
    def qubit_estimate(self, n_turns: int) -> int:
        """Return number of qubits needed for n_turns conformational turns."""
        ...

    @abstractmethod
    def term_count(self, n_turns: int, n_residues: int) -> int:
        """Return approximate number of energy terms in the Hamiltonian."""
        ...

    @abstractmethod
    def decode(self, bitstring: str) -> List[int]:
        """Decode a measurement bitstring to a list of integer turn codes.

        Args:
            bitstring: Binary string, length = qubit_estimate(n_turns).

        Returns:
            List of turn codes compatible with model.lattice.trace_positions().
        """
        ...

    @abstractmethod
    def penalty_summary(self) -> Dict[str, float]:
        """Return active penalty weights for display/logging."""
        ...

    def expected_circuit_depth(self, n_turns: int, ansatz_depth: int) -> int:
        """Estimate ansatz circuit depth.

        Default formula: ansatz_depth * (qubits + qubits - 1 CNOTs).
        Subclasses may override.
        """
        q = self.qubit_estimate(n_turns)
        return ansatz_depth * (1 + q - 1)  # Ry layer + CX chain

    def estimate_cost(self, n_turns: int, n_residues: int,
                      ansatz_depth: int = 2) -> EncodingSpec:
        """Compute full resource estimate for a fragment."""
        q = self.qubit_estimate(n_turns)
        mem_mb = (2 ** q) * 8 / (1024 ** 2)  # float64 table

        if q <= 20:
            feasibility = "simulator_ok"
        elif q <= 30:
            feasibility = "hardware_feasible"
        else:
            feasibility = "too_large"

        return EncodingSpec(
            n_qubits=q,
            n_terms=self.term_count(n_turns, n_residues),
            circuit_depth_estimate=self.expected_circuit_depth(n_turns, ansatz_depth),
            cnot_estimate=ansatz_depth * (q - 1),
            memory_estimate_mb=round(mem_mb, 3),
            feasibility=feasibility,
        )
