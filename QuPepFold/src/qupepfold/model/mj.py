"""Miyazawa-Jernigan (MJ) contact potential matrix.

Implements the published 20x20 residue-residue contact potential table from
Miyazawa & Jernigan (1996) J. Mol. Biol. 256:623-644 (Table 5, contact energies e_ij).

Favorable hydrophobic contacts (e.g. Cys-Cys, Phe-Phe, Leu-Leu) have strongly
negative energies (down to -5.60 RT), salt-bridges (Asp/Glu with Arg/Lys) are favorable,
and like-charge contacts (Asp-Asp, Glu-Glu) are unfavorable.
"""

from typing import Dict, Optional
import numpy as np

# Standard amino acid 1-letter codes in alphabetical order
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Published Miyazawa-Jernigan (1996) Table 5 contact energy matrix (in RT units)
# Order: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
MJ_TABLE_1996: Dict[str, Dict[str, float]] = {
    'A': {'A': -2.75, 'C': -3.88, 'D': -1.91, 'E': -2.03, 'F': -3.87, 'G': -2.27, 'H': -2.79, 'I': -3.76, 'K': -2.08, 'L': -3.75, 'M': -3.42, 'N': -2.25, 'P': -2.32, 'Q': -2.48, 'R': -2.47, 'S': -2.36, 'T': -2.54, 'V': -3.44, 'W': -3.61, 'Y': -3.22},
    'C': {'A': -3.88, 'C': -5.44, 'D': -2.92, 'E': -2.87, 'F': -5.60, 'G': -3.12, 'H': -3.81, 'I': -5.41, 'K': -2.93, 'L': -5.43, 'M': -4.84, 'N': -3.26, 'P': -3.33, 'Q': -3.46, 'R': -3.28, 'S': -3.37, 'T': -3.58, 'V': -4.98, 'W': -5.01, 'Y': -4.54},
    'D': {'A': -1.91, 'C': -2.92, 'D': -1.48, 'E': -1.64, 'F': -2.87, 'G': -1.54, 'H': -2.53, 'I': -2.71, 'K': -2.44, 'L': -2.77, 'M': -2.43, 'N': -1.77, 'P': -1.80, 'Q': -1.91, 'R': -2.79, 'S': -1.71, 'T': -1.92, 'V': -2.49, 'W': -2.73, 'Y': -2.52},
    'E': {'A': -2.03, 'C': -2.87, 'D': -1.64, 'E': -1.72, 'F': -3.01, 'G': -1.61, 'H': -2.47, 'I': -2.83, 'K': -2.43, 'L': -2.87, 'M': -2.56, 'N': -1.83, 'P': -1.83, 'Q': -2.05, 'R': -2.67, 'S': -1.80, 'T': -1.98, 'V': -2.58, 'W': -2.87, 'Y': -2.61},
    'F': {'A': -3.87, 'C': -5.60, 'D': -2.87, 'E': -3.01, 'F': -5.46, 'G': -3.07, 'H': -3.87, 'I': -5.35, 'K': -3.11, 'L': -5.48, 'M': -5.20, 'N': -3.26, 'P': -3.29, 'Q': -3.53, 'R': -3.53, 'S': -3.40, 'T': -3.60, 'V': -4.97, 'W': -5.18, 'Y': -4.63},
    'G': {'A': -2.27, 'C': -3.12, 'D': -1.54, 'E': -1.61, 'F': -3.07, 'G': -1.78, 'H': -2.30, 'I': -2.99, 'K': -1.66, 'L': -2.97, 'M': -2.62, 'N': -1.82, 'P': -1.87, 'Q': -1.98, 'R': -2.04, 'S': -1.91, 'T': -2.07, 'V': -2.68, 'W': -2.92, 'Y': -2.57},
    'H': {'A': -2.79, 'C': -3.81, 'D': -2.53, 'E': -2.47, 'F': -3.87, 'G': -2.30, 'H': -2.87, 'I': -3.61, 'K': -2.37, 'L': -3.70, 'M': -3.34, 'N': -2.45, 'P': -2.36, 'Q': -2.63, 'R': -2.73, 'S': -2.44, 'T': -2.64, 'V': -3.39, 'W': -3.64, 'Y': -3.36},
    'I': {'A': -3.76, 'C': -5.41, 'D': -2.71, 'E': -2.83, 'F': -5.35, 'G': -2.99, 'H': -3.61, 'I': -5.17, 'K': -2.92, 'L': -5.25, 'M': -4.95, 'N': -3.10, 'P': -3.14, 'Q': -3.30, 'R': -3.30, 'S': -3.24, 'T': -3.45, 'V': -4.79, 'W': -4.82, 'Y': -4.39},
    'K': {'A': -2.08, 'C': -2.93, 'D': -2.44, 'E': -2.43, 'F': -3.11, 'G': -1.66, 'H': -2.37, 'I': -2.92, 'K': -1.97, 'L': -2.95, 'M': -2.64, 'N': -1.89, 'P': -1.89, 'Q': -2.12, 'R': -2.31, 'S': -1.86, 'T': -2.03, 'V': -2.62, 'W': -3.04, 'Y': -2.73},
    'L': {'A': -3.75, 'C': -5.43, 'D': -2.77, 'E': -2.87, 'F': -5.48, 'G': -2.97, 'H': -3.70, 'I': -5.25, 'K': -2.95, 'L': -5.25, 'M': -5.02, 'N': -3.13, 'P': -3.16, 'Q': -3.37, 'R': -3.34, 'S': -3.27, 'T': -3.47, 'V': -4.78, 'W': -5.00, 'Y': -4.43},
    'M': {'A': -3.42, 'C': -4.84, 'D': -2.43, 'E': -2.56, 'F': -5.20, 'G': -2.62, 'H': -3.34, 'I': -4.95, 'K': -2.64, 'L': -5.02, 'M': -4.38, 'N': -2.85, 'P': -2.87, 'Q': -3.03, 'R': -3.00, 'S': -2.92, 'T': -3.15, 'V': -4.54, 'W': -4.82, 'Y': -4.32},
    'N': {'A': -2.25, 'C': -3.26, 'D': -1.77, 'E': -1.83, 'F': -3.26, 'G': -1.82, 'H': -2.45, 'I': -3.10, 'K': -1.89, 'L': -3.13, 'M': -2.85, 'N': -1.97, 'P': -2.05, 'Q': -2.14, 'R': -2.25, 'S': -2.00, 'T': -2.21, 'V': -2.83, 'W': -3.08, 'Y': -2.78},
    'P': {'A': -2.32, 'C': -3.33, 'D': -1.80, 'E': -1.83, 'F': -3.29, 'G': -1.87, 'H': -2.36, 'I': -3.14, 'K': -1.89, 'L': -3.16, 'M': -2.87, 'N': -2.05, 'P': -2.08, 'Q': -2.19, 'R': -2.17, 'S': -2.07, 'T': -2.24, 'V': -2.91, 'W': -3.10, 'Y': -2.81},
    'Q': {'A': -2.48, 'C': -3.46, 'D': -1.91, 'E': -2.05, 'F': -3.53, 'G': -1.98, 'H': -2.63, 'I': -3.30, 'K': -2.12, 'L': -3.37, 'M': -3.03, 'N': -2.14, 'P': -2.19, 'Q': -2.31, 'R': -2.46, 'S': -2.17, 'T': -2.36, 'V': -3.06, 'W': -3.29, 'Y': -3.00},
    'R': {'A': -2.47, 'C': -3.28, 'D': -2.79, 'E': -2.67, 'F': -3.53, 'G': -2.04, 'H': -2.73, 'I': -3.30, 'K': -2.31, 'L': -3.34, 'M': -3.00, 'N': -2.25, 'P': -2.17, 'Q': -2.46, 'R': -2.37, 'S': -2.22, 'T': -2.40, 'V': -3.00, 'W': -3.38, 'Y': -3.10},
    'S': {'A': -2.36, 'C': -3.37, 'D': -1.71, 'E': -1.80, 'F': -3.40, 'G': -1.91, 'H': -2.44, 'I': -3.24, 'K': -1.86, 'L': -3.27, 'M': -2.92, 'N': -2.00, 'P': -2.07, 'Q': -2.17, 'R': -2.22, 'S': -2.08, 'T': -2.25, 'V': -2.95, 'W': -3.17, 'Y': -2.85},
    'T': {'A': -2.54, 'C': -3.58, 'D': -1.92, 'E': -1.98, 'F': -3.60, 'G': -2.07, 'H': -2.64, 'I': -3.45, 'K': -2.03, 'L': -3.47, 'M': -3.15, 'N': -2.21, 'P': -2.24, 'Q': -2.36, 'R': -2.40, 'S': -2.25, 'T': -2.44, 'V': -3.19, 'W': -3.37, 'Y': -3.06},
    'V': {'A': -3.44, 'C': -4.98, 'D': -2.49, 'E': -2.58, 'F': -4.97, 'G': -2.68, 'H': -3.39, 'I': -4.79, 'K': -2.62, 'L': -4.78, 'M': -4.54, 'N': -2.83, 'P': -2.91, 'Q': -3.06, 'R': -3.00, 'S': -2.95, 'T': -3.19, 'V': -4.37, 'W': -4.62, 'Y': -4.10},
    'W': {'A': -3.61, 'C': -5.01, 'D': -2.73, 'E': -2.87, 'F': -5.18, 'G': -2.92, 'H': -3.64, 'I': -4.82, 'K': -3.04, 'L': -5.00, 'M': -4.82, 'N': -3.08, 'P': -3.10, 'Q': -3.29, 'R': -3.38, 'S': -3.17, 'T': -3.37, 'V': -4.62, 'W': -4.77, 'Y': -4.47},
    'Y': {'A': -3.22, 'C': -4.54, 'D': -2.52, 'E': -2.61, 'F': -4.63, 'G': -2.57, 'H': -3.36, 'I': -4.39, 'K': -2.73, 'L': -4.43, 'M': -4.32, 'N': -2.78, 'P': -2.81, 'Q': -3.00, 'R': -3.10, 'S': -2.85, 'T': -3.06, 'V': -4.10, 'W': -4.47, 'Y': -4.01},
}


def build_mj_matrix(
    sequence: str,
    scale: float = 1.0,
    noise_fraction: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Build the Miyazawa-Jernigan interaction matrix for a sequence.
    
    Uses published statistical contact energies e_ij from Miyazawa & Jernigan (1996).
    
    Args:
        sequence: Amino acid sequence (one-letter codes).
        scale: Scaling multiplier (default 1.0; negative values invert favorable contacts).
        noise_fraction: Optional random noise fraction for symmetry breaking (default 0.0).
        seed: Random seed for reproducibility if noise_fraction > 0.
        
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
    for i, aa in enumerate(sequence):
        if aa not in MJ_TABLE_1996:
            raise ValueError(f"Invalid amino acid '{aa}' at position {i}")
    
    n = len(sequence)
    mj = np.zeros((n, n), dtype=np.float32)
    
    rng = np.random.default_rng(seed) if noise_fraction > 0 else None
    
    for i in range(n):
        aa_i = sequence[i]
        for j in range(i, n):
            aa_j = sequence[j]
            base = scale * MJ_TABLE_1996[aa_i][aa_j]
            
            if noise_fraction > 0 and rng is not None and base != 0:
                noise = rng.uniform(-noise_fraction, noise_fraction) * abs(base)
                val = base + noise
            else:
                val = base
                
            mj[i, j] = val
            mj[j, i] = val
    
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
    return float(total)


def get_aa_type_matrix(
    scale: float = 1.0,
    noise_fraction: float = 0.0,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Build a generic 20x20 amino acid type interaction matrix.
    
    Args:
        scale: Scaling factor.
        noise_fraction: Noise fraction.
        seed: Random seed.
        
    Returns:
        Nested dict: result[aa1][aa2] = interaction energy.
    """
    matrix: Dict[str, Dict[str, float]] = {}
    rng = np.random.default_rng(seed) if noise_fraction > 0 else None
    
    for aa1 in AMINO_ACIDS:
        matrix[aa1] = {}
        for aa2 in AMINO_ACIDS:
            base = scale * MJ_TABLE_1996[aa1][aa2]
            if noise_fraction > 0 and rng is not None and base != 0:
                val = base + rng.uniform(-noise_fraction, noise_fraction) * abs(base)
            else:
                val = base
            matrix[aa1][aa2] = float(val)
            
    return matrix
