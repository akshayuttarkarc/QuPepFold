"""Parallel Tempering (Replica Exchange) for enhanced global search.

Runs multiple SA chains at different temperatures and exchanges states
between adjacent temperature levels to improve exploration.
"""

from typing import Tuple, List, Optional
import numpy as np

from ..types import FoldConfig
from ..model.encoding import encode_turns
from ..model.lattice import trace_positions, count_overlaps, contacts
from ..model.mj import build_mj_matrix
from .anneal_sa import compute_chain_energy


def parallel_tempering(
    sequence: str,
    config: FoldConfig,
    n_replicas: int = 4,
    exchange_interval: int = 100,
) -> Tuple[List[int], float, str]:
    """Run parallel tempering with replica exchange.
    
    Args:
        sequence: Amino acid sequence.
        config: FoldConfig with SA parameters.
        n_replicas: Number of temperature replicas.
        exchange_interval: Steps between exchange attempts.
        
    Returns:
        Tuple of (best_turns, best_energy, best_bitstring).
    """
    n_residues = len(sequence)
    n_turns = n_residues - 1
    
    # Build MJ matrix
    mj_matrix = build_mj_matrix(sequence, seed=config.seed)
    rng = np.random.default_rng(config.seed)
    
    # Temperature ladder (geometric)
    t_min = config.sa_t_final
    t_max = config.sa_t_init
    temperatures = np.geomspace(t_min, t_max, n_replicas)
    
    # Initialize replicas with random states
    replicas = [list(rng.integers(0, 4, size=n_turns)) for _ in range(n_replicas)]
    energies = [
        compute_chain_energy(
            r, mj_matrix,
            config.overlap_penalty,
            config.contact_min_sep,
            config.contact_cutoff,
        )
        for r in replicas
    ]
    
    best_turns = replicas[0].copy()
    best_energy = energies[0]
    
    n_steps = config.sa_steps
    
    for step in range(n_steps):
        # Local moves for each replica
        for i in range(n_replicas):
            neighbor = replicas[i].copy()
            idx = rng.integers(0, n_turns)
            new_turn = (neighbor[idx] + rng.integers(1, 4)) % 4
            neighbor[idx] = new_turn
            
            neighbor_energy = compute_chain_energy(
                neighbor, mj_matrix,
                config.overlap_penalty,
                config.contact_min_sep,
                config.contact_cutoff,
            )
            
            delta = neighbor_energy - energies[i]
            temp = temperatures[i]
            
            if delta < 0 or rng.random() < np.exp(-delta / temp):
                replicas[i] = neighbor
                energies[i] = neighbor_energy
                
                if energies[i] < best_energy:
                    best_turns = replicas[i].copy()
                    best_energy = energies[i]
        
        # Replica exchange between adjacent temperatures
        if step > 0 and step % exchange_interval == 0:
            # Pick random pair of adjacent replicas
            i = rng.integers(0, n_replicas - 1)
            j = i + 1
            
            # Metropolis exchange criterion
            beta_i = 1.0 / temperatures[i]
            beta_j = 1.0 / temperatures[j]
            delta_beta = beta_i - beta_j
            delta_e = energies[i] - energies[j]
            
            # Acceptance: exp((beta_i - beta_j) * (E_i - E_j))
            accept_prob = min(1.0, np.exp(delta_beta * delta_e))
            
            if rng.random() < accept_prob:
                # Swap replicas
                replicas[i], replicas[j] = replicas[j], replicas[i]
                energies[i], energies[j] = energies[j], energies[i]
    
    best_bitstring = encode_turns(best_turns)
    return best_turns, best_energy, best_bitstring
