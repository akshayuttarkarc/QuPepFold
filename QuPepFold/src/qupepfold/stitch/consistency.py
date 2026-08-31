"""Overlap consistency checking and mismatch penalty computation."""

from typing import Optional


def check_overlap_compatibility(
    bits_a: str,
    bits_b: str,
    overlap_bits: int,
) -> bool:
    """Check if two candidates are compatible on their overlap region.
    
    The last `overlap_bits` of candidate A must match
    the first `overlap_bits` of candidate B.
    
    Args:
        bits_a: Bitstring from fragment A.
        bits_b: Bitstring from fragment B.
        overlap_bits: Number of overlapping bits (= overlap_turns * 2).
        
    Returns:
        True if overlap regions match exactly.
        
    Example:
        >>> check_overlap_compatibility("0011001100", "001100", 4)
        True  # Last 4 bits of A ("1100") == first 4 bits of B ("0011")? No!
    """
    if overlap_bits <= 0:
        return True  # No overlap to check
    
    if overlap_bits > len(bits_a) or overlap_bits > len(bits_b):
        return False
    
    # Last overlap_bits of A
    a_overlap = bits_a[-overlap_bits:]
    # First overlap_bits of B
    b_overlap = bits_b[:overlap_bits]
    
    return a_overlap == b_overlap


def compute_mismatch_penalty(
    bits_a: str,
    bits_b: str,
    overlap_bits: int,
    penalty_per_bit: float = 100.0,
) -> float:
    """Compute mismatch penalty between overlapping regions.
    
    For soft overlap constraints, compute penalty proportional
    to Hamming distance between overlap regions.
    
    Args:
        bits_a: Bitstring from fragment A.
        bits_b: Bitstring from fragment B.
        overlap_bits: Number of overlapping bits.
        penalty_per_bit: Penalty per mismatched bit.
        
    Returns:
        Total mismatch penalty (0 if perfect match).
    """
    if overlap_bits <= 0:
        return 0.0
    
    if overlap_bits > len(bits_a) or overlap_bits > len(bits_b):
        return float('inf')
    
    a_overlap = bits_a[-overlap_bits:]
    b_overlap = bits_b[:overlap_bits]
    
    # Hamming distance
    mismatches = sum(a != b for a, b in zip(a_overlap, b_overlap))
    
    return mismatches * penalty_per_bit


def get_overlap_region(bits: str, which: str, overlap_bits: int) -> str:
    """Extract overlap region from a bitstring.
    
    Args:
        bits: Full bitstring.
        which: 'left' (first bits) or 'right' (last bits).
        overlap_bits: Number of bits in overlap.
        
    Returns:
        Overlap region bitstring.
    """
    if which == 'left':
        return bits[:overlap_bits]
    elif which == 'right':
        return bits[-overlap_bits:]
    else:
        raise ValueError(f"which must be 'left' or 'right', got '{which}'")
