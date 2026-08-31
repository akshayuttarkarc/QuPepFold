"""Simulated Annealing for global backbone search on lattice model.

State = list of turn codes (length N-1 for N residues).
Neighbor = change one turn to another of 4 states.
Objective = full-chain lattice energy with all Hamiltonian terms.
"""

from typing import Tuple, List, Optional, Callable
import numpy as np

from ..types import FoldConfig
from ..model.encoding import encode_turns
from ..model.lattice import trace_positions, count_overlaps, contacts
from ..model.mj import build_mj_matrix
from ..model.energy_fragment import (
    compute_backbone_penalty,
    compute_geometric_constraints,
    DEFAULT_LAM_BACK,
    DEFAULT_LAM_DIS,
    DEFAULT_LAM_LOC,
)


def compute_chain_energy(
    turns: List[int],
    mj_matrix: np.ndarray,
    config: FoldConfig,
    include_geometric_constraints: bool = True,
) -> float:
    """Compute full-chain lattice energy with all Hamiltonian terms.
    
    Energy terms (matching original exact_hamiltonian):
    1. E_overlap: Self-intersection penalty
    2. E_contact: MJ contact energy
    3. E_back: Adjacent equal turns penalty (lamBack)
    4. E_geo: Distance and locality constraints (lamDis, lamLoc)
    
    Args:
        turns: List of turn codes.
        mj_matrix: MJ interaction matrix.
        config: FoldConfig with all parameters.
        include_geometric_constraints: Include lamDis/lamLoc terms.
        
    Returns:
        Total energy.
    """
    n_residues = len(turns) + 1
    positions = trace_positions(turns)
    
    # Energy term 1: Overlap penalty
    n_overlaps = count_overlaps(positions)
    e_overlap = config.overlap_penalty * n_overlaps
    
    # Energy term 2: Contact energy (MJ potential)
    contact_pairs = contacts(
        positions,
        min_sep=config.contact_min_sep,
        cutoff=config.contact_cutoff
    )
    e_contact = sum(mj_matrix[i, j] for i, j in contact_pairs)
    
    # Energy term 3: Backbone turn variety penalty
    lam_back = getattr(config, 'lam_back', DEFAULT_LAM_BACK)
    e_back = compute_backbone_penalty(turns, lam_back)
    
    # Energy term 4: Geometric constraints (distance + locality)
    e_geo = 0.0
    if include_geometric_constraints:
        lam_dis = getattr(config, 'lam_dis', DEFAULT_LAM_DIS)
        lam_loc = getattr(config, 'lam_loc', DEFAULT_LAM_LOC)
        if (lam_dis > 0 or lam_loc > 0) and contact_pairs:
            e_geo = compute_geometric_constraints(
                turns, n_residues, lam_dis, lam_loc,
                positions=positions, contact_pairs=contact_pairs
            )
    
    return e_overlap + e_contact + e_back + e_geo


def simulated_annealing(
    sequence: str,
    config: FoldConfig,
    callback: Optional[Callable[[int, List[int], float, float], None]] = None,
) -> Tuple[List[int], float, str]:
    """Run simulated annealing to find optimal turn configuration.
    
    Args:
        sequence: Amino acid sequence.
        config: FoldConfig with SA parameters.
        callback: Optional callback(step, turns, energy, temperature).
        
    Returns:
        Tuple of (best_turns, best_energy, best_bitstring).
        
    Example:
        >>> turns, energy, bits = simulated_annealing("ACDEFGHI", config)
        >>> len(turns)
        7  # N-1 turns for N=8 residues
    """
    n_residues = len(sequence)
    n_turns = n_residues - 1
    
    # Build MJ matrix
    mj_matrix = build_mj_matrix(sequence, seed=config.seed)
    
    # Initialize random state
    rng = np.random.default_rng(config.seed)
    current_turns = list(rng.integers(0, 4, size=n_turns))
    
    current_energy = compute_chain_energy(current_turns, mj_matrix, config)
    
    best_turns = current_turns.copy()
    best_energy = current_energy
    
    # Temperature schedule (exponential)
    t_init = config.sa_t_init
    t_final = config.sa_t_final
    n_steps = config.sa_steps
    
    # alpha such that t_init * alpha^n_steps = t_final
    alpha = (t_final / t_init) ** (1.0 / n_steps) if n_steps > 0 else 1.0
    temperature = t_init
    
    for step in range(n_steps):
        # Propose neighbor: change one random turn
        neighbor = current_turns.copy()
        idx = rng.integers(0, n_turns)
        old_turn = neighbor[idx]
        # Choose a different turn
        new_turn = (old_turn + rng.integers(1, 4)) % 4
        neighbor[idx] = new_turn
        
        neighbor_energy = compute_chain_energy(neighbor, mj_matrix, config)
        
        # Acceptance criterion (Metropolis)
        delta = neighbor_energy - current_energy
        accept = False
        
        if delta < 0:
            accept = True
        else:
            # Accept with probability exp(-delta/T)
            prob = np.exp(-delta / temperature) if temperature > 1e-10 else 0.0
            accept = rng.random() < prob
        
        if accept:
            current_turns = neighbor
            current_energy = neighbor_energy
            
            if current_energy < best_energy:
                best_turns = current_turns.copy()
                best_energy = current_energy
        
        # Cool down
        temperature *= alpha
        
        # Callback for monitoring
        if callback is not None:
            callback(step, current_turns, current_energy, temperature)
    
    best_bitstring = encode_turns(best_turns)
    return best_turns, best_energy, best_bitstring


def sa_with_restarts(
    sequence: str,
    config: FoldConfig,
    n_restarts: int = 5,
) -> Tuple[List[int], float, str]:
    """Run SA with multiple random restarts.
    
    Args:
        sequence: Amino acid sequence.
        config: FoldConfig.
        n_restarts: Number of independent SA runs.
        
    Returns:
        Best result across all restarts.
    """
    best_overall_turns = None
    best_overall_energy = float('inf')
    best_overall_bits = ""
    
    for restart in range(n_restarts):
        # Modify seed for each restart
        restart_config = FoldConfig(**{
            **vars(config),
            'seed': config.seed + restart * 1000,
        })
        
        turns, energy, bits = simulated_annealing(sequence, restart_config)
        
        if energy < best_overall_energy:
            best_overall_turns = turns
            best_overall_energy = energy
            best_overall_bits = bits
    
    return best_overall_turns, best_overall_energy, best_overall_bits
