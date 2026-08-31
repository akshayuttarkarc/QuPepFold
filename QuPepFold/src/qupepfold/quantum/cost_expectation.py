"""Expected energy computation from sampler results.

Computes ⟨E⟩ = Σ p(z) * E(z) where E(z) is looked up from
the precomputed energy table.
"""

from typing import Dict

from ..types import FragmentEnergyTable


def expected_energy(
    prob_dict: Dict[str, float],
    energy_table: FragmentEnergyTable,
) -> float:
    """Compute expected energy from probability distribution.
    
    E = Σ p(z) * E_table[z]  (only over valid, self-avoiding states)
    
    Args:
        prob_dict: {bitstring: probability} from sampler.
        energy_table: Precomputed energy lookup table.
        
    Returns:
        Expected energy value (computed only over valid states).
    """
    total = 0.0
    valid_prob_sum = 0.0
    
    for bits, prob in prob_dict.items():
        if len(bits) != energy_table.n_bits:
            continue
        energy = energy_table.energy_of_bitstring(bits)
        
        # Skip forbidden states (self-intersecting)
        if energy == float('inf') or energy > 1e10:
            continue
            
        total += prob * energy
        valid_prob_sum += prob
    
    # Renormalize over valid states
    if valid_prob_sum > 0:
        return total / valid_prob_sum
    return 1000.0  # Large finite penalty instead of INF to keep optimizer stable


def cvar_energy(
    prob_dict: Dict[str, float],
    energy_table: FragmentEnergyTable,
    alpha: float = 0.1,
) -> float:
    """Compute CVaR (Conditional Value at Risk) of energy.
    
    CVaR_α is the expected energy in the lowest α-fraction of the distribution.
    Only considers valid (self-avoiding) states.
    
    Args:
        prob_dict: {bitstring: probability} from sampler.
        energy_table: Precomputed energy lookup table.
        alpha: Tail probability (0 < alpha <= 1).
        
    Returns:
        CVaR energy value.
    """
    diagnostics = cvar_energy_with_diagnostics(prob_dict, energy_table, alpha)
    return diagnostics['cvar']


def cvar_energy_with_diagnostics(
    prob_dict: Dict[str, float],
    energy_table: FragmentEnergyTable,
    alpha: float = 0.1,
) -> dict:
    """Compute CVaR with full diagnostics for scientific validation.
    
    Args:
        prob_dict: {bitstring: probability} from sampler.
        energy_table: Precomputed energy lookup table.
        alpha: Tail probability (0 < alpha <= 1).
        
    Returns:
        dict with:
        - cvar: CVaR energy value
        - valid_fraction: fraction of probability mass on valid states
        - n_valid_samples: number of valid bitstrings sampled
        - n_total_samples: total number of bitstrings sampled
        - tail_cutoff_energy: energy threshold for alpha tail
        - n_tail_samples: number of states in the CVaR tail
    """
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    
    # Build list of (energy, probability) pairs - filter forbidden states
    pairs = []
    n_total = 0
    n_valid = 0
    total_prob = 0.0
    valid_prob = 0.0
    
    for bits, prob in prob_dict.items():
        if len(bits) != energy_table.n_bits:
            continue
        n_total += 1
        total_prob += prob
        
        energy = energy_table.energy_of_bitstring(bits)
        
        # Filter forbidden states (self-intersecting)
        if energy == float('inf') or energy > 1e10:
            continue
            
        n_valid += 1
        valid_prob += prob
        pairs.append((energy, prob))
    
    if not pairs:
        return {
            'cvar': 1000.0,
            'valid_fraction': 0.0,
            'n_valid_samples': 0,
            'n_total_samples': n_total,
            'tail_cutoff_energy': 1000.0,
            'n_tail_samples': 0,
        }
    
    # Renormalize probabilities over valid states
    if valid_prob > 0:
        pairs = [(e, p / valid_prob) for e, p in pairs]
    
    # Sort by energy (ascending)
    pairs.sort(key=lambda x: x[0])
    
    # Accumulate probability up to alpha
    cvar = 0.0
    cumulative_prob = 0.0
    n_tail = 0
    tail_cutoff = pairs[-1][0]  # Will be updated
    
    for energy, prob in pairs:
        if cumulative_prob + prob <= alpha:
            cvar += prob * energy
            cumulative_prob += prob
            n_tail += 1
            tail_cutoff = energy
        else:
            # Partial contribution from this state
            remaining = alpha - cumulative_prob
            cvar += remaining * energy
            cumulative_prob = alpha
            n_tail += 1
            tail_cutoff = energy
            break
    
    cvar_value = cvar / alpha if alpha > 0 else 0.0
    
    return {
        'cvar': cvar_value,
        'valid_fraction': valid_prob / total_prob if total_prob > 0 else 0.0,
        'n_valid_samples': n_valid,
        'n_total_samples': n_total,
        'tail_cutoff_energy': tail_cutoff,
        'n_tail_samples': n_tail,
    }


def best_bitstring(
    prob_dict: Dict[str, float],
    energy_table: FragmentEnergyTable,
    by: str = "energy",
) -> tuple:
    """Find best bitstring by energy or probability.
    
    Args:
        prob_dict: {bitstring: probability} from sampler.
        energy_table: Precomputed energy lookup table.
        by: 'energy' (lowest) or 'probability' (highest).
        
    Returns:
        Tuple of (bitstring, energy, probability).
    """
    best = None
    
    for bits, prob in prob_dict.items():
        if len(bits) != energy_table.n_bits:
            continue
        
        energy = energy_table.energy_of_bitstring(bits)
        
        if best is None:
            best = (bits, energy, prob)
        elif by == "energy" and energy < best[1]:
            best = (bits, energy, prob)
        elif by == "probability" and prob > best[2]:
            best = (bits, energy, prob)
    
    return best


def top_k_candidates(
    prob_dict: Dict[str, float],
    energy_table: FragmentEnergyTable,
    k: int = 20,
    by: str = "probability",
) -> list:
    """Extract top-K candidates from sampler results.
    
    Args:
        prob_dict: {bitstring: probability} from sampler.
        energy_table: Energy lookup table.
        k: Number of candidates to return.
        by: 'probability' or 'energy'.
        
    Returns:
        List of (bitstring, energy, probability) tuples.
    """
    candidates = []
    
    for bits, prob in prob_dict.items():
        if len(bits) == energy_table.n_bits:
            energy = energy_table.energy_of_bitstring(bits)
            candidates.append((bits, energy, prob))
    
    if by == "probability":
        candidates.sort(key=lambda x: -x[2])  # Descending by probability
    else:
        candidates.sort(key=lambda x: x[1])   # Ascending by energy
    
    return candidates[:k]
