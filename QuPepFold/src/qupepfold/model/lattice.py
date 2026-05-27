"""2D lattice walk and contact detection.

Turn codes define relative direction changes on a 2D square lattice.
Starting direction is +x axis.
"""

from typing import List, Tuple, Set
import numpy as np

# Direction vectors for heading states (dx, dy)
# Heading 0 = +x, 1 = +y, 2 = -x, 3 = -y
DIRECTION_VECTORS = {
    0: (1, 0),   # +x (right)
    1: (0, 1),   # +y (up)
    2: (-1, 0),  # -x (left)
    3: (0, -1),  # -y (down)
}

# Turn code effects on heading
# 0 = straight, 1 = left turn, 2 = right turn, 3 = reverse
TURN_DELTA = {
    0: 0,   # straight: no heading change
    1: 1,   # left: +90° (counterclockwise)
    2: -1,  # right: -90° (clockwise)
    3: 2,   # reverse: 180°
}


def trace_positions(turns: List[int], start_pos: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Walk turns on 2D lattice and return positions for each residue.
    
    The first residue is at start_pos. Each turn determines the direction
    to the next residue.
    
    Args:
        turns: List of turn codes (length N-1 for N residues).
        start_pos: Starting position for first residue.
        
    Returns:
        Array of shape (N, 2) with (x, y) positions.
        
    Example:
        >>> trace_positions([0, 0, 1])  # 4 residues: straight, straight, left
        array([[0, 0],
               [1, 0],
               [2, 0],
               [2, 1]])
    """
    n_residues = len(turns) + 1
    positions = np.zeros((n_residues, 2), dtype=np.int32)
    positions[0] = start_pos
    
    heading = 0  # Start facing +x
    
    for i, turn in enumerate(turns):
        # Update heading based on turn
        heading = (heading + TURN_DELTA[turn]) % 4
        
        # Get direction vector for new heading
        dx, dy = DIRECTION_VECTORS[heading]
        
        # Place next residue
        positions[i + 1] = positions[i] + (dx, dy)
    
    return positions


def count_overlaps(positions: np.ndarray) -> int:
    """Count the number of self-intersecting positions.
    
    An overlap occurs when two or more residues occupy the same lattice position.
    
    Args:
        positions: Array of shape (N, 2) with positions.
        
    Returns:
        Number of duplicate positions (total positions - unique positions).
    """
    # Convert to set of tuples for uniqueness check
    unique = set(map(tuple, positions))
    return len(positions) - len(unique)


def get_overlapping_pairs(positions: np.ndarray) -> List[Tuple[int, int]]:
    """Find all pairs of residues that occupy the same position.
    
    Args:
        positions: Array of shape (N, 2) with positions.
        
    Returns:
        List of (i, j) pairs where i < j and positions[i] == positions[j].
    """
    pairs = []
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if np.array_equal(positions[i], positions[j]):
                pairs.append((i, j))
    return pairs


def contacts(
    positions: np.ndarray,
    min_sep: int = 2,
    cutoff: float = 1.0
) -> List[Tuple[int, int]]:
    """Find residue pairs in contact.
    
    Two residues are in contact if:
    1. They are at least min_sep apart in sequence (|j - i| >= min_sep)
    2. Their lattice distance is <= cutoff
    
    Args:
        positions: Array of shape (N, 2) with positions.
        min_sep: Minimum sequence separation for contacts.
        cutoff: Maximum lattice distance for contact.
        
    Returns:
        List of (i, j) pairs in contact, where i < j.
        
    Example:
        >>> pos = np.array([[0,0], [1,0], [1,1], [0,1]])  # square
        >>> contacts(pos, min_sep=2, cutoff=1.0)
        [(0, 3)]  # residue 0 and 3 are adjacent on lattice
    """
    contact_pairs = []
    n = len(positions)
    cutoff_sq = cutoff ** 2
    
    for i in range(n):
        for j in range(i + min_sep, n):
            # Compute squared Euclidean distance
            diff = positions[j] - positions[i]
            dist_sq = np.sum(diff ** 2)
            
            if dist_sq <= cutoff_sq:
                contact_pairs.append((i, j))
    
    return contact_pairs


def lattice_distance(positions: np.ndarray, i: int, j: int) -> float:
    """Compute Euclidean distance between two residues on lattice.
    
    Args:
        positions: Array of shape (N, 2) with positions.
        i, j: Residue indices.
        
    Returns:
        Euclidean distance.
    """
    return float(np.linalg.norm(positions[j] - positions[i]))


def radius_of_gyration(positions: np.ndarray) -> float:
    """Compute radius of gyration for a set of positions.
    
    Rg = sqrt(1/N * Σ |r_i - r_mean|²)
    
    Args:
        positions: Array of shape (N, 2) with positions.
        
    Returns:
        Radius of gyration.
    """
    center = np.mean(positions, axis=0)
    diffs = positions - center
    return float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))
