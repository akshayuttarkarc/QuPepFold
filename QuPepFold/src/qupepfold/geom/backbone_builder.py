"""Build 3D backbone coordinates from turn codes.

Converts lattice turn codes → backbone dihedrals (φ, ψ) → 3D coordinates.
Uses standard peptide geometry for bond lengths and angles.
"""

import math
from typing import List, Tuple, Dict
import numpy as np

# Standard peptide geometry (Angstroms and radians)
BOND_LENGTHS = {
    "C-N": 1.329,   # Peptide bond
    "N-CA": 1.458,  # Alpha carbon
    "CA-C": 1.525,  # Carbonyl carbon
    "C=O": 1.229,   # Carbonyl oxygen
}

BOND_ANGLES = {
    "C-N-CA": math.radians(121.7),
    "N-CA-C": math.radians(110.4),
    "CA-C-N": math.radians(116.2),
    "CA-C-O": math.radians(120.8),
}

OMEGA_TRANS = math.radians(180.0)  # Trans peptide bond

# Turn code → (φ, ψ) dihedral mapping
# These represent secondary structure preferences
TURN_DIHEDRALS = {
    0: (math.radians(-60), math.radians(-45)),   # Helix-like
    1: (math.radians(-135), math.radians(135)),  # Beta-like
    2: (math.radians(-75), math.radians(145)),   # PPII-like
    3: (math.radians(-60), math.radians(140)),   # Extended/coil
}

# Three-letter amino acid codes
AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def turns_to_dihedrals(turns: List[int], n_residues: int) -> Tuple[List[float], List[float]]:
    """Convert turn codes to backbone dihedrals.
    
    Args:
        turns: List of turn codes (length n_residues - 1).
        n_residues: Number of residues.
        
    Returns:
        Tuple of (phi_list, psi_list), each length n_residues.
    """
    phis = [0.0] * n_residues
    psis = [0.0] * n_residues
    
    for i in range(n_residues):
        # φ comes from the previous turn
        t_prev = turns[i - 1] if i > 0 and i - 1 < len(turns) else 1
        # ψ comes from the current turn
        t_next = turns[i] if i < len(turns) else 2
        
        phis[i] = TURN_DIHEDRALS[t_prev][0]
        psis[i] = TURN_DIHEDRALS[t_next][1]
    
    return phis, psis


def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else np.zeros_like(v)


def _orthonormal_frame(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build orthonormal frame from three points."""
    cb = _normalize(b - c)
    t = b - a
    n = np.cross(t, cb)
    
    if np.linalg.norm(n) < 1e-8:
        # Fallback for collinear points
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, cb)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        n = np.cross(tmp, cb)
    
    n = _normalize(n)
    m = _normalize(np.cross(n, cb))
    return m, n, cb


def _place_atom(
    p_a: np.ndarray,
    p_b: np.ndarray,
    p_c: np.ndarray,
    bond_len: float,
    angle_rad: float,
    dihedral_rad: float,
) -> np.ndarray:
    """Place atom D given A-B-C and geometry parameters."""
    m, n, cb = _orthonormal_frame(p_a, p_b, p_c)
    
    x = -bond_len * math.cos(angle_rad)
    y = bond_len * math.cos(dihedral_rad) * math.sin(angle_rad)
    z = bond_len * math.sin(dihedral_rad) * math.sin(angle_rad)
    
    d = p_c + x * cb + y * m + z * n
    return d


def _seed_first_residue() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate coordinates for the first residue."""
    n1 = np.array([0.0, 0.0, 0.0])
    ca1 = np.array([BOND_LENGTHS["N-CA"], 0.0, 0.0])
    
    ang = BOND_ANGLES["N-CA-C"]
    vx, vy = -math.cos(ang), math.sin(ang)
    c1 = ca1 + BOND_LENGTHS["CA-C"] * np.array([vx, vy, 0.0])
    
    o1 = _place_atom(n1, ca1, c1, BOND_LENGTHS["C=O"], BOND_ANGLES["CA-C-O"], 0.0)
    
    return n1, ca1, c1, o1


def build_backbone_coords(
    sequence: str,
    phis: List[float],
    psis: List[float],
) -> List[Dict]:
    """Build 3D backbone coordinates from dihedrals.
    
    Args:
        sequence: Amino acid sequence.
        phis: List of φ angles (radians).
        psis: List of ψ angles (radians).
        
    Returns:
        List of atom dicts with 'name', 'resname', 'resid', 'coords'.
    """
    n_residues = len(sequence)
    atoms = []
    
    # First residue
    n1, ca1, c1, o1 = _seed_first_residue()
    
    prev_a, prev_b, prev_c = n1, ca1, c1
    
    for i in range(n_residues):
        if i == 0:
            n_pos, ca_pos, c_pos, o_pos = n1, ca1, c1, o1
        else:
            phi, psi = phis[i], psis[i]
            
            # Place N
            n_pos = _place_atom(prev_a, prev_b, prev_c, BOND_LENGTHS["C-N"], BOND_ANGLES["CA-C-N"], OMEGA_TRANS)
            # Place CA
            ca_pos = _place_atom(prev_b, prev_c, n_pos, BOND_LENGTHS["N-CA"], BOND_ANGLES["C-N-CA"], phi)
            # Place C
            c_pos = _place_atom(prev_c, n_pos, ca_pos, BOND_LENGTHS["CA-C"], BOND_ANGLES["N-CA-C"], psi)
            # Place O
            o_pos = _place_atom(n_pos, ca_pos, c_pos, BOND_LENGTHS["C=O"], BOND_ANGLES["CA-C-O"], 0.0)
            
            prev_a, prev_b, prev_c = n_pos, ca_pos, c_pos
        
        resname = AA3.get(sequence[i], "UNK")
        resid = i + 1
        
        # Add backbone atoms
        atoms.append({"name": "N", "resname": resname, "resid": resid, "coords": n_pos})
        atoms.append({"name": "CA", "resname": resname, "resid": resid, "coords": ca_pos})
        
        # Add CB for non-glycine residues
        if sequence[i] != "G":
            cb_pos = _compute_cb_position(n_pos, ca_pos, c_pos)
            atoms.append({"name": "CB", "resname": resname, "resid": resid, "coords": cb_pos})
        
        atoms.append({"name": "C", "resname": resname, "resid": resid, "coords": c_pos})
        atoms.append({"name": "O", "resname": resname, "resid": resid, "coords": o_pos})
    
    return atoms


def _compute_cb_position(n_pos: np.ndarray, ca_pos: np.ndarray, c_pos: np.ndarray) -> np.ndarray:
    """Compute CB position from backbone atoms."""
    v1 = _normalize(n_pos - ca_pos)
    v2 = _normalize(c_pos - ca_pos)
    
    # CB is roughly opposite to the average of N-CA and C-CA vectors
    u = v1 + v2
    if np.linalg.norm(u) < 1e-8:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, v1)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        u = _normalize(tmp)
    else:
        u = _normalize(u)
    
    n = np.cross(v1, v2)
    if np.linalg.norm(n) < 1e-8:
        n = np.array([0.0, 0.0, 1.0])
    n = _normalize(n)
    
    # Tetrahedral geometry
    dir_cb = _normalize(0.943 * u + 0.333 * n)
    cb_pos = ca_pos + 1.53 * dir_cb
    
    return cb_pos


def build_from_turns(sequence: str, turns: List[int]) -> List[Dict]:
    """Build backbone from turn codes (convenience function).
    
    Args:
        sequence: Amino acid sequence.
        turns: Turn codes (length n_residues - 1).
        
    Returns:
        List of atom dicts.
    """
    phis, psis = turns_to_dihedrals(turns, len(sequence))
    return build_backbone_coords(sequence, phis, psis)
