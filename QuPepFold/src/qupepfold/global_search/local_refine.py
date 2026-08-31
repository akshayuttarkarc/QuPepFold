"""Local refinement for post-stitch optimization.

Short simulated annealing initialized at stitched solution to ensure
the final result is locally optimal under the true global energy function.
"""

from typing import Tuple, List, Optional
import numpy as np

from ..types import FoldConfig
from ..model.encoding import encode_turns, decode_turns
from ..model.mj import build_mj_matrix
from .anneal_sa import compute_chain_energy


def local_refine(
    initial_bitstring: str,
    sequence: str,
    config: FoldConfig,
    steps: int = 2000,
    t_init: float = 0.5,
    t_final: float = 0.01,
    verbose: bool = False,
) -> Tuple[str, float, bool]:
    """Local refinement via short SA initialized at given solution.
    
    This ensures the stitched solution is locally optimal under the
    true global energy function (with all cross-fragment interactions).
    
    Args:
        initial_bitstring: Starting bitstring from stitching.
        sequence: Amino acid sequence.
        config: FoldConfig with energy parameters.
        steps: Number of local search steps (default: 500).
        t_init: Initial temperature (low for local search).
        t_final: Final temperature.
        verbose: Print progress.
        
    Returns:
        Tuple of (refined_bitstring, refined_energy, improved).
    """
    n_residues = len(sequence)
    n_turns = n_residues - 1
    
    # Decode initial solution
    initial_turns = decode_turns(initial_bitstring)
    
    # Build MJ matrix
    mj_matrix = build_mj_matrix(sequence, seed=config.seed)
    
    # Initialize from stitched solution
    current_turns = list(initial_turns)
    current_energy = compute_chain_energy(current_turns, mj_matrix, config)
    
    best_turns = current_turns.copy()
    best_energy = current_energy
    initial_energy = current_energy
    
    # Temperature schedule
    alpha = (t_final / t_init) ** (1.0 / steps) if steps > 0 else 1.0
    temperature = t_init
    
    rng = np.random.default_rng(config.seed + 12345)  # Different seed from global SA
    
    for step in range(steps):
        # Propose neighbor: change one random turn
        neighbor = current_turns.copy()
        idx = rng.integers(0, n_turns)
        old_turn = neighbor[idx]
        new_turn = (old_turn + rng.integers(1, 4)) % 4
        neighbor[idx] = new_turn
        
        neighbor_energy = compute_chain_energy(neighbor, mj_matrix, config)
        
        # Acceptance criterion (Metropolis)
        delta = neighbor_energy - current_energy
        accept = False
        
        if delta < 0:
            accept = True
        else:
            prob = np.exp(-delta / temperature) if temperature > 1e-10 else 0.0
            accept = rng.random() < prob
        
        if accept:
            current_turns = neighbor
            current_energy = neighbor_energy
            
            if current_energy < best_energy:
                best_turns = current_turns.copy()
                best_energy = current_energy
        
        temperature *= alpha
    
    best_bitstring = encode_turns(best_turns)
    improved = best_energy < initial_energy - 1e-6
    
    if verbose:
        if improved:
            print(f"  Local refinement: {initial_energy:.2f} → {best_energy:.2f} (improved)")
        else:
            print(f"  Local refinement: {best_energy:.2f} (no improvement)")
    
    return best_bitstring, best_energy, improved
