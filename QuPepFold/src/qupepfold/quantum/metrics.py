"""Quantum metrics tracking for scientific transparency.

Reports circuit depth, gate counts, total evaluations, and CVaR diagnostics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class QuantumMetrics:
    """Metrics for quantum circuit execution.
    
    Tracks real scaling metrics beyond just qubit count:
    - Circuit depth and gate counts
    - Total shots and evaluations
    - CVaR diagnostic data
    """
    # Circuit structure
    n_qubits: int = 0
    circuit_depth: int = 0
    cnot_count: int = 0
    single_qubit_gates: int = 0
    
    # Execution totals
    shots_per_circuit: int = 0
    spsa_iterations: int = 0
    n_fragments: int = 0
    
    # CVaR diagnostics
    self_avoiding_fraction: List[float] = field(default_factory=list)
    cvar_improvements: List[float] = field(default_factory=list)
    
    @property
    def total_shots(self) -> int:
        """Total shots = shots × 2 × iterations × fragments (SPSA uses 2 evals/iter)."""
        return self.shots_per_circuit * 2 * self.spsa_iterations * self.n_fragments
    
    @property
    def total_circuit_evaluations(self) -> int:
        """Total circuit evaluations across all fragments."""
        return 2 * self.spsa_iterations * self.n_fragments + self.n_fragments  # +1 per fragment for final sampling
    
    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        return {
            "n_qubits": self.n_qubits,
            "circuit_depth": self.circuit_depth,
            "cnot_count": self.cnot_count,
            "single_qubit_gates": self.single_qubit_gates,
            "shots_per_circuit": self.shots_per_circuit,
            "spsa_iterations": self.spsa_iterations,
            "n_fragments": self.n_fragments,
            "total_shots": self.total_shots,
            "total_circuit_evaluations": self.total_circuit_evaluations,
            "avg_self_avoiding_fraction": np.mean(self.self_avoiding_fraction) if self.self_avoiding_fraction else None,
            "cvar_improvement_count": sum(1 for x in self.cvar_improvements if x > 0),
        }


@dataclass
class HybridWinStats:
    """Statistics for SA vs hybrid comparison.
    
    Tracks win rate to justify hybrid approach.
    """
    sa_wins: int = 0
    hybrid_wins: int = 0
    ties: int = 0
    
    # Energy improvements when hybrid wins
    hybrid_improvements: List[float] = field(default_factory=list)
    
    @property
    def total_runs(self) -> int:
        return self.sa_wins + self.hybrid_wins + self.ties
    
    @property
    def hybrid_win_rate(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.hybrid_wins / self.total_runs
    
    def record_result(self, sa_energy: float, hybrid_energy: float, tolerance: float = 1e-6):
        """Record result of SA vs hybrid comparison."""
        if hybrid_energy < sa_energy - tolerance:
            self.hybrid_wins += 1
            self.hybrid_improvements.append(sa_energy - hybrid_energy)
        elif sa_energy < hybrid_energy - tolerance:
            self.sa_wins += 1
        else:
            self.ties += 1
    
    def to_dict(self) -> Dict:
        return {
            "sa_wins": self.sa_wins,
            "hybrid_wins": self.hybrid_wins,
            "ties": self.ties,
            "hybrid_win_rate": self.hybrid_win_rate,
            "avg_hybrid_improvement": np.mean(self.hybrid_improvements) if self.hybrid_improvements else 0.0,
        }


def extract_circuit_metrics(circuit) -> Dict[str, int]:
    """Extract gate counts and depth from a circuit.
    
    Args:
        circuit: Qiskit QuantumCircuit
        
    Returns:
        Dict with depth, cnot_count, single_qubit_gates
    """
    from qiskit.converters import circuit_to_dag
    
    dag = circuit_to_dag(circuit)
    
    cnot_count = 0
    single_qubit_gates = 0
    
    for node in dag.op_nodes():
        if node.op.num_qubits == 2:
            cnot_count += 1
        elif node.op.num_qubits == 1:
            single_qubit_gates += 1
    
    return {
        "depth": circuit.depth(),
        "cnot_count": cnot_count,
        "single_qubit_gates": single_qubit_gates,
    }
