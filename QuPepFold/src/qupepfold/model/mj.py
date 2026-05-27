"""MJ-like coarse contact potential.

This implements a simplified Miyazawa-Jernigan-like interaction matrix.
Note: This is NOT the actual MJ matrix from literature, but a coarse
approximation using hydrophobicity-based interactions.

For scientific use, consider loading the actual MJ matrix from published data.
"""

from typing import Dict
import numpy as np

# Standard amino acid codes
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Hydrophobicity scale (Kyte-Doolittle, normalized to [-1, 1])
# More negative = more hydrophilic, more positive = more hydrophobic
HYDROPHOBICITY = {
    'A':  0.44,  'C':  0.63,  'D': -0.92,  'E': -0.90,
    'F':  0.73,  'G':  0.00,  'H': -0.46,  'I':  0.88,
    'K': -0.99,  'L':  0.85,  'M':  0.53,  'N': -0.79,
    'P':  0.04,  'Q': -0.85,  'R': -1.00,  'S': -0.36,
    'T': -0.26,  'V':  0.79,  'W':  0.40,  'Y':  0.15,
}


def build_mj_matrix(
    sequence: str,
    scale: float = -4.0,
    seed: int = 42
) -> np.ndarray:
    """Build an MJ-like interaction matrix for a sequence.
    
    The interaction energy between residues i and j is computed as:
    E_ij = scale * H_i * H_j + noise
    
    where H is the hydrophobicity of the amino acid and noise is small
    random perturbation for breaking symmetry.
    
    Args:
        sequence: Amino acid sequence (one-letter codes).
        scale: Scaling factor for interactions (negative = favorable).
        seed: Random seed for reproducibility.
        
    Returns:
        Symmetric (N, N) matrix of interaction energies.
        
    Raises:
        ValueError: If sequence contains invalid amino acid codes.
        
    Example:
        >>> mj = build_mj_matrix("ACDEF")
        >>> mj.shape
        (5, 5)
        >>> mj[0, 1] == mj[1, 0]  # Symmetric
        True
    """
    # Validate sequence
    for i, aa in enumerate(sequence):
        if aa not in HYDROPHOBICITY:
            raise ValueError(f"Invalid amino acid '{aa}' at position {i}")
    
    n = len(sequence)
    rng = np.random.default_rng(seed)
    
    # Build matrix
    mj = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        hi = HYDROPHOBICITY[sequence[i]]
        for j in range(i, n):
            hj = HYDROPHOBICITY[sequence[j]]
            
            # Base interaction: hydrophobic residues attract each other
            base = scale * hi * hj
            
            # Small noise for symmetry breaking (±5%)
            noise = rng.uniform(-0.05, 0.05) * abs(base) if base != 0 else 0
            
            mj[i, j] = base + noise
            mj[j, i] = mj[i, j]  # Symmetric
    
    return mj


def get_contact_energy(
    mj_matrix: np.ndarray,
    contacts: list,
) -> float:
    """Compute total contact energy from MJ matrix.
    
    Args:
        mj_matrix: Interaction matrix from build_mj_matrix.
        contacts: List of (i, j) contact pairs.
        
    Returns:
        Sum of interaction energies for all contacts.
    """
    total = 0.0
    for i, j in contacts:
        total += mj_matrix[i, j]
    return total


def get_aa_type_matrix(
    scale: float = -4.0,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """Build a generic amino acid type interaction matrix.
    
    This returns a 20x20 dictionary of interaction energies
    between all pairs of standard amino acids.
    
    Args:
        scale: Scaling factor for interactions.
        seed: Random seed for reproducibility.
        
    Returns:
        Nested dict: result[aa1][aa2] = interaction energy.
    """
    rng = np.random.default_rng(seed)
    matrix = {}
    
    for aa1 in AMINO_ACIDS:
        matrix[aa1] = {}
        h1 = HYDROPHOBICITY[aa1]
        for aa2 in AMINO_ACIDS:
            h2 = HYDROPHOBICITY[aa2]
            base = scale * h1 * h2
            noise = rng.uniform(-0.05, 0.05) * abs(base) if base != 0 else 0
            matrix[aa1][aa2] = base + noise
    
    return matrix
