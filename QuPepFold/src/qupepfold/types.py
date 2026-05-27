"""Core data structures for QuPepFold.

All dataclasses use explicit typing and are immutable where possible.
Bit indexing convention: big-endian (most significant bit first).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
import numpy as np


@dataclass
class FoldConfig:
    """Configuration for the folding pipeline.
    
    Attributes:
        overlap_penalty: Penalty for lattice self-intersection.
        contact_cutoff: Distance cutoff for contacts (lattice units).
        fragment_length: Number of amino acids per fragment (default 6).
        overlap_turns: Number of overlapping turns between fragments (default 2).
        sa_steps: Simulated annealing steps for global search.
        sa_t_init: Initial temperature for SA.
        sa_t_final: Final temperature for SA.
        ansatz_depth: Number of mixing layers in quantum ansatz.
        shots: Number of shots per circuit execution.
        spsa_iterations: Number of SPSA optimizer iterations.
        spsa_a: SPSA learning rate parameter.
        spsa_c: SPSA perturbation size parameter.
        warmstart_mode: 'basis' or 'biased_ry'.
        warmstart_bias: Bias probability for biased_ry mode.
        top_k_candidates: Number of top candidates to extract per fragment.
        top_m_by_energy: Number to keep after energy filtering.
        backend: 'aer' or 'runtime'.
        ibm_backend: IBM backend name for runtime mode.
        relax_force_field: Force field for OpenMM relaxation.
        seed: Random seed for reproducibility.
    """
    # Lattice/contact parameters
    overlap_penalty: float = 1000.0
    contact_cutoff: float = 1.0
    contact_min_sep: int = 2
    
    # Geometric constraint penalties
    # CRITICAL: These must be low enough that MJ contact energy can dominate.
    # Original VQE used control bits to selectively apply penalties; without
    # control bits, unconditional penalties overwhelm favorable MJ contacts.
    # MJ values typically range from -4 to +4, so penalties must be < |MJ|.
    lam_back: float = 5.0     # Adjacent equal turns (reduced from 50)
    lam_dis: float = 0.0      # Disabled - requires control bits for proper use
    lam_loc: float = 0.0      # Disabled - requires control bits for proper use
    
    # Fragment parameters
    fragment_length: int = 7
    overlap_turns: int = 2
    
    # Global SA parameters
    sa_steps: int = 5000
    sa_t_init: float = 100.0
    sa_t_final: float = 0.1
    sa_restarts: int = 3      # Number of SA restarts (tries)
    
    # Quantum parameters
    ansatz_depth: int = 2
    shots: int = 2000
    spsa_iterations: int = 50
    spsa_a: float = 0.05  # Smaller step for local refinement
    spsa_c: float = 0.05  # Smaller perturbation for local search
    warmstart_mode: Literal["basis", "biased_ry"] = "biased_ry"  # Use biased for local exploration
    warmstart_bias: float = 0.95  # High bias = stay close to SA solution
    
    # CVaR parameters
    cvar_alpha: float = 0.25  # CVaR tail probability (0.025 = focus on best 2.5%)
    optimization_tries: int = 50  # Number of optimization restarts
    
    # Candidate extraction
    top_k_candidates: int = 20
    top_m_by_energy: int = 10
    
    # Backend selection
    backend: Literal["aer", "runtime"] = "aer"
    ibm_backend: Optional[str] = None
    ibm_token: Optional[str] = None  # IBM Quantum API token
    
    # Output parameters
    export_prob_threshold: float = 0.02  # Min probability for PDB export
    
    # Relaxation
    relax_force_field: str = "amber14-all"
    relax_max_iterations: int = 500
    
    # Encoding selection
    encoding: str = "turn"  # "turn", "hp", "constrained_local"
    
    # Optimizer selection
    optimizer: str = "spsa"  # "spsa", "cobyla", "gradient", "nelder_mead"
    
    # Adaptive shot allocation
    adaptive_shots: bool = False
    
    # Fragment strategy
    fragment_strategy: str = "fixed_window"  # "fixed_window", "disorder", "domain", "user_defined"
    
    # Stopping criteria
    energy_tolerance: float = 1e-4  # Stop SA/VQE when improvement < this
    max_wall_clock_seconds: Optional[float] = None  # Overall timeout
    early_stop_patience: int = 10  # Stop after N iterations without improvement
    
    # Caching
    cache_dir: Optional[str] = None  # None = no caching
    cache_approx_tolerance: int = 2  # Hamming distance for approximate cache match
    
    # Reproducibility
    seed: int = 42


@dataclass
class FragmentSpec:
    """Specification for a sequence fragment.
    
    Attributes:
        start_res: Starting residue index (inclusive, 0-based).
        end_res: Ending residue index (exclusive, 0-based).
        n_turns: Number of turns = end_res - start_res - 1.
        n_bits: Number of bits = 2 * n_turns.
        overlap_left_turns: Turns overlapping with previous fragment.
        overlap_right_turns: Turns overlapping with next fragment.
        sequence: Amino acid sequence for this fragment.
    """
    start_res: int
    end_res: int
    sequence: str
    overlap_left_turns: int = 0
    overlap_right_turns: int = 0
    priority: float = 0.0  # Higher = process first
    parent_protein_id: Optional[str] = None
    preprocessing_metadata: Dict = field(default_factory=dict)
    
    @property
    def n_turns(self) -> int:
        return self.end_res - self.start_res - 1
    
    @property
    def n_bits(self) -> int:
        return 2 * self.n_turns
    
    def __post_init__(self):
        if self.end_res <= self.start_res:
            raise ValueError(f"end_res ({self.end_res}) must be > start_res ({self.start_res})")
        if len(self.sequence) != (self.end_res - self.start_res):
            raise ValueError(f"sequence length {len(self.sequence)} != fragment length {self.end_res - self.start_res}")


@dataclass
class FragmentEnergyTable:
    """Precomputed energy lookup table for a fragment.
    
    For a 6-AA fragment: n_turns=5, n_bits=10, 2^10=1024 states.
    
    Attributes:
        n_bits: Number of bits encoding the fragment.
        energies: Array of shape (2**n_bits,) with energy for each bitstring.
        bit_indexing: Always 'big_endian' for consistency.
        fragment_spec: The fragment this table corresponds to.
    """
    n_bits: int
    energies: np.ndarray
    fragment_spec: FragmentSpec
    bit_indexing: str = "big_endian"
    
    def energy_of_bitstring(self, bits: str) -> float:
        """Get energy for a bitstring."""
        if len(bits) != self.n_bits:
            raise ValueError(f"Bitstring length {len(bits)} != n_bits {self.n_bits}")
        idx = int(bits, 2)  # big-endian: bits[0] is MSB
        return float(self.energies[idx])
    
    def expected_energy(self, prob_dict: Dict[str, float]) -> float:
        """Compute expected energy E = Σ p(z) * E(z)."""
        total = 0.0
        for bits, prob in prob_dict.items():
            total += prob * self.energy_of_bitstring(bits)
        return total
    
    def get_best_candidates(self, top_k: int = 10) -> List['FragmentCandidate']:
        """Get the top-K candidates with the lowest (most negative) energy.
        
        This directly examines all 2^n_bits states in the precomputed table
        and returns the ones with the lowest energy values.
        
        Args:
            top_k: Number of best candidates to return.
            
        Returns:
            List of FragmentCandidate objects sorted by energy (lowest first).
        """
        # Get indices sorted by energy (ascending = lowest first)
        sorted_indices = np.argsort(self.energies)[:top_k]
        
        candidates = []
        for idx in sorted_indices:
            # Convert index to bitstring (big-endian)
            bits = format(idx, f'0{self.n_bits}b')
            energy = float(self.energies[idx])
            candidates.append(FragmentCandidate(
                bits=bits,
                energy=energy,
                probability=1.0 / (2 ** self.n_bits),  # Uniform probability
                meta={"source": "energy_table"}
            ))
        
        return candidates
    
    def __post_init__(self):
        expected_size = 2 ** self.n_bits
        if self.energies.shape != (expected_size,):
            raise ValueError(f"energies shape {self.energies.shape} != expected ({expected_size},)")


@dataclass
class FragmentCandidate:
    """A candidate solution for a fragment.
    
    Attributes:
        bits: Bitstring encoding the turn configuration.
        energy: Energy of this configuration.
        probability: Sampling probability (if from quantum).
        meta: Additional metadata (backend, shots, etc.).
    """
    bits: str
    energy: float
    probability: float = 0.0
    meta: Dict = field(default_factory=dict)


@dataclass
class FoldResult:
    """Complete result of a folding run.
    
    Attributes:
        sequence: Input amino acid sequence.
        global_bits: Bitstring from classical global search.
        global_energy: Energy of global search result.
        fragment_candidates: List of candidate lists per fragment.
        stitched_bits: Final stitched bitstring.
        stitched_energy: Energy of stitched result.
        pdb_path: Path to backbone-only PDB.
        relaxed_pdb_path: Path to relaxed PDB.
        provenance: Metadata dict (git hash, versions, timestamps, etc.).
    """
    sequence: str
    global_bits: str
    global_energy: float
    fragment_candidates: List[List[FragmentCandidate]]
    stitched_bits: str
    stitched_energy: float
    pdb_path: Optional[str] = None
    relaxed_pdb_path: Optional[str] = None
    provenance: Dict = field(default_factory=dict)
    selected_candidates: Optional[List[FragmentCandidate]] = None
    energy_breakdown: Optional[Dict[str, float]] = None
    # Scientific metrics
    quantum_metrics: Optional[Dict] = None  # Circuit depth, CNOTs, total evaluations
    run_metrics: Optional["RunMetrics"] = None  # Structured timing/metrics (from metrics.py)
    convergence_trace: Optional[List[float]] = None  # SA convergence trace
    win_source: str = "stitched"  # "global_sa" or "stitched" - which method produced final result
    
    @property
    def energy(self) -> float:
        """Best (lowest) energy from global or stitched."""
        return min(self.global_energy, self.stitched_energy)
    
    @property
    def best_bits(self) -> str:
        """Bitstring corresponding to best energy."""
        if self.global_energy <= self.stitched_energy:
            return self.global_bits
        return self.stitched_bits

