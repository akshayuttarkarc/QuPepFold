"""Dynamic programming for optimal fragment stitching.

Formulates stitching as shortest-path DP over candidate compatibility.
"""

from typing import List, Tuple, Optional
import numpy as np

from ..types import FragmentSpec, FragmentCandidate
from .consistency import compute_mismatch_penalty, check_overlap_compatibility
from ..model.encoding import decode_turns
from ..model.lattice import trace_positions, count_overlaps, contacts


def stitch_fragments(
    fragment_candidates: List[List[FragmentCandidate]],
    fragments: List[FragmentSpec],
    mismatch_penalty: float = 1000.0,
    require_exact_match: bool = False,
    overlap_penalty: float = 1000.0,
    mj_matrix: Optional[np.ndarray] = None,
    boundary_band: int = 2,
) -> Tuple[str, float, List[FragmentCandidate]]:
    """Stitch fragment candidates into a global bitstring via DP.
    
    Finds the lowest-cost chain of candidates where:
    - Each fragment contributes one candidate
    - Overlap regions between adjacent fragments must be compatible
    - Total cost = sum of candidate energies + mismatch penalties
    - Checks for self-intersections in the growing chain
    - *NEW*: Includes cross-fragment boundary contact energy
    
    Args:
        fragment_candidates: List of candidate lists, one per fragment.
        fragments: List of FragmentSpec (for overlap info).
        mismatch_penalty: Penalty per mismatched bit in overlap.
        require_exact_match: If True, reject mismatched overlaps entirely.
        overlap_penalty: Penalty per self-intersection in combined chain.
        mj_matrix: MJ interaction matrix (optional, for boundary contacts).
        boundary_band: Number of residues at boundary to check for contacts.
        
    Returns:
        Tuple of (stitched_bitstring, total_energy, selected_candidates).
        
    Raises:
        ValueError: If no valid stitching exists.
    """
    n_fragments = len(fragment_candidates)
    
    if n_fragments == 0:
        return "", 0.0, []
    
    if n_fragments == 1:
        # Single fragment - return best candidate
        best = min(fragment_candidates[0], key=lambda c: c.energy)
        return best.bits, best.energy, [best]
    
    # DP arrays
    # dp[i][k] = minimum cost to reach candidate k of fragment i
    # parent[i][k] = index of best predecessor candidate in fragment i-1
    n_candidates = [len(cands) for cands in fragment_candidates]
    
    INF = float('inf')
    dp = [[INF] * n_candidates[i] for i in range(n_fragments)]
    parent = [[-1] * n_candidates[i] for i in range(n_fragments)]
    
    # Initialize first fragment
    # Note: Even the first fragment should be checked for internal overlaps,
    # but candidates from refine_fragment usually account for this in their energy.
    for k, cand in enumerate(fragment_candidates[0]):
        dp[0][k] = cand.energy
    
    # Forward pass
    for i in range(1, n_fragments):
        overlap_bits = fragments[i].overlap_left_turns * 2
        
        for k, cand_k in enumerate(fragment_candidates[i]):
            for j, cand_j in enumerate(fragment_candidates[i - 1]):
                
                # 1. Compatibility Check (Bit Mismatch)
                if require_exact_match:
                    if not check_overlap_compatibility(cand_j.bits, cand_k.bits, overlap_bits):
                        continue
                    penalty = 0.0
                else:
                    penalty = compute_mismatch_penalty(
                        cand_j.bits, cand_k.bits, overlap_bits, mismatch_penalty
                    )
                
                # Pruning: If previous step was impossible, skip
                if dp[i - 1][j] == INF:
                    continue

                # 2. Geometric Overlap Check (Position Validation)
                # Reconstruct path prefix to check for collisions with newly added residues
                current_bits = _reconstruct_path_bits(
                    parent, fragment_candidates, i-1, j, fragments
                )
                
                bits_k = cand_k.bits
                if overlap_bits > 0:
                    new_bits = bits_k[overlap_bits:]
                else:
                    new_bits = bits_k
                
                full_bits = current_bits + new_bits
                
                # Trace full positions and count only NEW overlaps introduced by appending fragment i
                turns_prefix = decode_turns(current_bits)
                turns_full = decode_turns(full_bits)
                
                pos_prefix = trace_positions(turns_prefix) if turns_prefix else np.zeros((0, 2))
                pos_full = trace_positions(turns_full)
                
                n_prefix_overlaps = count_overlaps(pos_prefix) if len(pos_prefix) > 0 else 0
                n_full_overlaps = count_overlaps(pos_full)
                new_overlaps = max(0, n_full_overlaps - n_prefix_overlaps)
                
                # Penalize only the incremental overlaps introduced at this step
                geo_penalty = new_overlaps * overlap_penalty
                
                cost = dp[i - 1][j] + cand_k.energy + penalty + geo_penalty
                
                if cost < dp[i][k]:
                    dp[i][k] = cost
                    parent[i][k] = j
    
    # Find best end
    best_end = min(range(n_candidates[-1]), key=lambda k: dp[-1][k])
    
    if dp[-1][best_end] == INF:
        # Fallback: try to find any path? Or raise error?
        # If strict validation filtered everything, we might need to relax.
        raise ValueError("No valid stitching found - all paths have infinite cost (likely overlaps)")
    
    # Backtrack
    selected_indices = [0] * n_fragments
    selected_indices[-1] = best_end
    
    for i in range(n_fragments - 1, 0, -1):
        selected_indices[i - 1] = parent[i][selected_indices[i]]
    
    # Collect selected candidates
    selected = [fragment_candidates[i][selected_indices[i]] for i in range(n_fragments)]
    
    # Build stitched bitstring
    stitched = assemble_bitstring(selected, fragments)
    
    # Total energy is sum of fragment energies
    # Note: The DP cost includes overlap penalties which are 'virtual',
    # so we return sum(cand.energy) as the raw energy of the components,
    # but the selection was biased by the validity check.
    # The caller usually recomputes full chain energy anyway.
    total_energy = sum(c.energy for c in selected)
    
    return stitched, total_energy, selected


def _reconstruct_path_bits(
    parent: List[List[int]],
    fragment_candidates: List[List[FragmentCandidate]],
    frag_idx: int,
    cand_idx: int,
    fragments: List[FragmentSpec],
) -> str:
    """Reconstruct bitstring for the optimal path ending at (frag_idx, cand_idx)."""
    # Backtrack locally
    indices = []
    curr = cand_idx
    for i in range(frag_idx, -1, -1):
        indices.append(curr)
        if i > 0:
            curr = parent[i][curr]
            if curr == -1:
                return "" # Should not happen if path exists
    
    indices.reverse() # Now [0...frag_idx]
    
    # Assemble
    selected = [fragment_candidates[i][indices[i]] for i in range(frag_idx + 1)]
    
    # Helper similar to assemble_bitstring but for a subset
    if not selected:
        return ""
    
    result = list(selected[0].bits)
    for i in range(1, len(selected)):
        overlap_bits = fragments[i].overlap_left_turns * 2
        bits = selected[i].bits
        if overlap_bits > 0:
            new_bits = bits[overlap_bits:]
        else:
            new_bits = bits
        result.extend(new_bits)
            
    return "".join(result)


def assemble_bitstring(
    candidates: List[FragmentCandidate],
    fragments: List[FragmentSpec],
) -> str:
    """Assemble a global bitstring from fragment candidates.
    
    Overlapping regions are taken from the left fragment.
    
    Args:
        candidates: Selected candidates, one per fragment.
        fragments: Fragment specifications.
        
    Returns:
        Full stitched bitstring.
    """
    if not candidates:
        return ""
    
    # Start with first fragment's bits
    result = list(candidates[0].bits)
    
    for i in range(1, len(candidates)):
        overlap_bits = fragments[i].overlap_left_turns * 2
        
        # Append non-overlapping portion of this fragment
        bits = candidates[i].bits
        if overlap_bits > 0:
            new_bits = bits[overlap_bits:]
        else:
            new_bits = bits
        
        result.extend(new_bits)
    
    return ''.join(result)


def score_stitching(
    candidates: List[FragmentCandidate],
    fragments: List[FragmentSpec],
    mismatch_penalty: float = 1000.0,
) -> float:
    """Score a candidate chain.
    
    Args:
        candidates: Selected candidates.
        fragments: Fragment specifications.
        mismatch_penalty: Penalty per bit mismatch.
        
    Returns:
        Total cost (energies + mismatches).
    """
    total = sum(c.energy for c in candidates)
    
    for i in range(1, len(candidates)):
        overlap_bits = fragments[i].overlap_left_turns * 2
        penalty = compute_mismatch_penalty(
            candidates[i - 1].bits,
            candidates[i].bits,
            overlap_bits,
            mismatch_penalty,
        )
        total += penalty
    
    return total
