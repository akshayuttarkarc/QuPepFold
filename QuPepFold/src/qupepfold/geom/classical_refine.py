"""Classical local refinement for assembled protein models.

Provides:
  smooth_junctions()   : Scipy-based local minimization at fragment boundaries
  remove_clashes()     : Iterative steric clash resolution via position nudging
  coarse_score()       : Quick geometric quality check (Rg, contact order)
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize


# ── Clash detection & removal ─────────────────────────────────────────────────

def find_clashes(
    positions: np.ndarray,
    threshold: float = 1.2,
    min_sep: int = 3,
) -> List[Tuple[int, int]]:
    """Find pairs of residues with steric clashes.

    Args:
        positions: (N, 3) or (N, 2) array of Cα positions.
        threshold: Minimum allowed inter-residue distance (Å or lattice units).
        min_sep: Minimum sequence separation to consider (default: 3).

    Returns:
        List of (i, j) clashing residue index pairs.
    """
    clashes = []
    n = len(positions)
    for i in range(n):
        for j in range(i + min_sep, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < threshold:
                clashes.append((i, j))
    return clashes


def remove_clashes(
    positions: np.ndarray,
    threshold: float = 1.2,
    min_sep: int = 3,
    max_iterations: int = 20,
    nudge_strength: float = 0.1,
) -> np.ndarray:
    """Iteratively nudge clashing residues apart.

    For each clashing pair, displace the lighter residue (higher index)
    along the inter-residue vector by nudge_strength.

    Args:
        positions: (N, D) position array (modified in-place copy).
        threshold: Clash distance cutoff.
        min_sep: Minimum sequence separation.
        max_iterations: Maximum nudge iterations.
        nudge_strength: Displacement per nudge step.

    Returns:
        Updated (N, D) position array.
    """
    pos = positions.copy().astype(float)

    for iteration in range(max_iterations):
        clashes = find_clashes(pos, threshold, min_sep)
        if not clashes:
            break
        for i, j in clashes:
            vec = pos[j] - pos[i]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                # Random nudge to break symmetry
                vec = np.random.default_rng(iteration * 1000 + i + j).uniform(
                    -1, 1, size=pos.shape[1]
                )
                norm = np.linalg.norm(vec)
            direction = vec / norm
            gap = threshold - norm
            pos[j] += direction * (gap + nudge_strength) * 0.5
            pos[i] -= direction * (gap + nudge_strength) * 0.5

    return pos


# ── Junction smoothing ────────────────────────────────────────────────────────

def smooth_junctions(
    positions: np.ndarray,
    junction_residues: List[int],
    n_neighbors: int = 2,
    n_iterations: int = 5,
) -> np.ndarray:
    """Smooth positions at fragment junction residues using local averaging.

    For each junction residue, applies a Gaussian-weighted average of the
    n_neighbors surrounding positions to reduce sharp kinks.

    Args:
        positions: (N, D) position array.
        junction_residues: List of residue indices at fragment boundaries.
        n_neighbors: Number of neighbors on each side to include (default: 2).
        n_iterations: Number of smoothing passes (default: 5).

    Returns:
        Smoothed (N, D) position array.
    """
    pos = positions.copy().astype(float)
    n = len(pos)

    weights_template = np.exp(
        -0.5 * np.arange(-n_neighbors, n_neighbors + 1) ** 2
    )
    weights_template /= weights_template.sum()

    for _ in range(n_iterations):
        for j in junction_residues:
            idxs = list(range(
                max(0, j - n_neighbors), min(n, j + n_neighbors + 1)
            ))
            w = weights_template[
                (n_neighbors - (j - max(0, j - n_neighbors))):
                (n_neighbors + min(n, j + n_neighbors + 1) - j)
            ]
            if len(w) != len(idxs):
                continue
            w = w / w.sum()
            pos[j] = sum(w[k] * pos[idxs[k]] for k in range(len(idxs)))

    return pos


def scipy_minimize_junctions(
    positions: np.ndarray,
    junction_residues: List[int],
    bond_length_target: float = 1.0,
    n_neighbors: int = 3,
) -> np.ndarray:
    """Minimize local strain at junction residues using scipy L-BFGS-B.

    Minimizes bond-length deviations around junctions while keeping
    distant residues fixed.

    Args:
        positions: (N, D) position array.
        junction_residues: Indices of boundary residues.
        bond_length_target: Target Cα-Cα bond length (lattice units).
        n_neighbors: Residues on each side included in minimization.

    Returns:
        Refined (N, D) position array.
    """
    pos = positions.copy().astype(float)
    n, d = pos.shape

    # Build set of mobile residue indices
    mobile = set()
    for j in junction_residues:
        for k in range(max(0, j - n_neighbors), min(n, j + n_neighbors + 1)):
            mobile.add(k)
    mobile_list = sorted(mobile)

    if len(mobile_list) < 2:
        return pos

    # Extract mobile positions as flat vector
    x0 = pos[mobile_list].flatten()

    def energy(x_flat):
        """Bond length strain energy for mobile residues."""
        pos_mobile = x_flat.reshape(len(mobile_list), d)
        # Build full position array with mobile updated
        pos_full = pos.copy()
        for k, idx in enumerate(mobile_list):
            pos_full[idx] = pos_mobile[k]

        e = 0.0
        for i in range(n - 1):
            diff = pos_full[i + 1] - pos_full[i]
            dist = np.linalg.norm(diff)
            e += (dist - bond_length_target) ** 2
        return e

    result = minimize(energy, x0, method="L-BFGS-B",
                      options={"maxiter": 200, "ftol": 1e-6})
    optimized = result.x.reshape(len(mobile_list), d)
    for k, idx in enumerate(mobile_list):
        pos[idx] = optimized[k]

    return pos
