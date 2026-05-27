"""Fragment generation with overlap management.

Fragments slide across the sequence with overlap to enable consistent stitching.
Supports multiple strategies via config.fragment_strategy.
"""

from typing import List, Iterator

from ..types import FoldConfig, FragmentSpec
from .fragment_strategies import get_strategy, FixedWindowStrategy


def generate_fragments(
    sequence: str,
    config: FoldConfig,
) -> List[FragmentSpec]:
    """Generate overlapping fragment specifications for a sequence.
    
    Delegates to the strategy specified by config.fragment_strategy.
    Default is 'fixed_window' (backward-compatible sliding window).
    
    Fragments are windows of length config.fragment_length that slide
    across the sequence with overlap of config.overlap_turns turns.
    
    For a sequence of length N with fragment_length=6 and overlap_turns=2:
    - Fragment 0: residues 0-6 (turns 0-4)
    - Fragment 1: residues 4-10 (turns 4-8), overlapping 2 turns with Fragment 0
    - etc.
    
    Args:
        sequence: Full amino acid sequence.
        config: FoldConfig with fragment_length and overlap_turns.
        
    Returns:
        List of FragmentSpec objects.
        
    Raises:
        ValueError: If sequence is too short for fragmentation.
        
    Example:
        >>> specs = generate_fragments("ACDEFGHIKLMN", config)
        >>> len(specs)
        2  # With fragment_length=6, overlap_turns=2
    """
    # Dispatch to strategy if not default fixed_window
    strategy_name = getattr(config, 'fragment_strategy', 'fixed_window')
    if strategy_name != 'fixed_window':
        strategy = get_strategy(strategy_name)
        return strategy.generate(sequence, config)
    
    # Original fixed-window logic (kept inline for backward compatibility)
    n = len(sequence)
    frag_len = config.fragment_length
    overlap_turns = config.overlap_turns
    
    if n < frag_len:
        # Single fragment covering entire sequence
        return [FragmentSpec(
            start_res=0,
            end_res=n,
            sequence=sequence,
            overlap_left_turns=0,
            overlap_right_turns=0,
        )]
    
    # Calculate stride: residues to advance between fragment starts
    # overlap_turns turns = (overlap_turns + 1) residues of overlap
    # So stride = frag_len - (overlap_turns + 1) residues
    overlap_residues = overlap_turns + 1
    stride = frag_len - overlap_residues
    
    if stride <= 0:
        raise ValueError(
            f"overlap_turns ({overlap_turns}) too large for fragment_length ({frag_len}). "
            f"Need overlap_turns < fragment_length - 1"
        )
    
    fragments = []
    start = 0
    frag_idx = 0
    
    while start < n:
        end = min(start + frag_len, n)
        
        # Determine overlap flags
        is_first = (frag_idx == 0)
        is_last = (end >= n)
        
        overlap_left = 0 if is_first else overlap_turns
        
        # Check if there will be a next fragment
        next_start = start + stride
        has_next = (next_start < n) and (next_start + frag_len <= n + overlap_residues)
        overlap_right = overlap_turns if has_next else 0
        
        fragments.append(FragmentSpec(
            start_res=start,
            end_res=end,
            sequence=sequence[start:end],
            overlap_left_turns=overlap_left,
            overlap_right_turns=overlap_right,
        ))
        
        start += stride
        frag_idx += 1
        
        # Handle final fragment covering remaining residues
        if start < n and start + frag_len > n:
            # Last fragment might be shorter or we extend to cover
            end = n
            if end - start < 3:  # Too short for meaningful fragment
                # Extend previous fragment instead
                break
            
            fragments.append(FragmentSpec(
                start_res=start,
                end_res=end,
                sequence=sequence[start:end],
                overlap_left_turns=overlap_turns,
                overlap_right_turns=0,
            ))
            break
    
    return fragments


def fragment_to_global_indices(
    fragment: FragmentSpec,
    local_turn_idx: int,
) -> int:
    """Convert local turn index within fragment to global turn index.
    
    Args:
        fragment: FragmentSpec.
        local_turn_idx: Turn index within fragment (0 to n_turns-1).
        
    Returns:
        Global turn index (0 to N-2 for N residues).
    """
    return fragment.start_res + local_turn_idx


def get_overlap_bitstring_slice(
    fragment: FragmentSpec,
    which: str,
) -> slice:
    """Get the slice of a bitstring corresponding to overlap region.
    
    Args:
        fragment: FragmentSpec.
        which: 'left' or 'right'.
        
    Returns:
        Slice object for the overlap portion of a bitstring.
    """
    n_bits = fragment.n_bits
    
    if which == 'left':
        # First overlap_left_turns * 2 bits
        n_overlap_bits = fragment.overlap_left_turns * 2
        return slice(0, n_overlap_bits)
    elif which == 'right':
        # Last overlap_right_turns * 2 bits
        n_overlap_bits = fragment.overlap_right_turns * 2
        return slice(n_bits - n_overlap_bits, n_bits)
    else:
        raise ValueError(f"which must be 'left' or 'right', got '{which}'")
