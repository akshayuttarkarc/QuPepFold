"""Coarse-grained model scoring: geometric and physical quality metrics.

Provides ModelScore and score_model() to quickly evaluate assembled structures
without requiring OpenMM or other heavy tools.

Metrics:
  - radius_of_gyration: Compactness (lower = more compact)
  - contact_order      : Sequence-normalized average loop length of contacts
  - clash_count        : Number of steric clashes
  - n_contacts         : Total non-covalent contacts
  - ramachandran_pct   : Fraction of residues in "good" backbone regions (heuristic)
  - per_region_confidence: Dict mapping region_id → confidence [0,1]
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ModelScore:
    """Quality metrics for an assembled protein model.

    Attributes:
        total_energy: Raw lattice energy from the folding Hamiltonian.
        radius_of_gyration: Rg in lattice units (lower = more compact).
        contact_order: Sequence-normalized mean loop length (higher = more complex fold).
        clash_count: Number of steric clashes found.
        n_contacts: Total non-covalent residue contacts.
        ramachandran_pct: Fraction of turns in "good" backbone regions [0, 1].
        per_region_confidence: Per-fragment confidence scores from stitching.
        overall_confidence: Mean of per_region_confidence values.
    """
    total_energy: float = 0.0
    radius_of_gyration: float = 0.0
    contact_order: float = 0.0
    clash_count: int = 0
    n_contacts: int = 0
    ramachandran_pct: float = 1.0
    per_region_confidence: Dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 1.0

    def summary(self) -> str:
        lines = [
            "=== Model Score ===",
            f"  Energy:          {self.total_energy:.3f}",
            f"  Rg:              {self.radius_of_gyration:.3f}",
            f"  Contact order:   {self.contact_order:.3f}",
            f"  Clashes:         {self.clash_count}",
            f"  Contacts:        {self.n_contacts}",
            f"  Ramachandran %:  {self.ramachandran_pct * 100:.1f}%",
            f"  Confidence:      {self.overall_confidence:.3f}",
        ]
        if self.per_region_confidence:
            lines.append("  Per-region confidence:")
            for k, v in self.per_region_confidence.items():
                lines.append(f"    {k}: {v:.3f}")
        return "\n".join(lines)


def radius_of_gyration(positions: np.ndarray) -> float:
    """Compute radius of gyration of a set of Cα positions.

    Rg = sqrt(mean(|r_i - r_com|^2))

    Args:
        positions: (N, D) array of positions.

    Returns:
        Radius of gyration.
    """
    if len(positions) == 0:
        return 0.0
    com = positions.mean(axis=0)
    diff = positions - com
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


def contact_order(
    positions: np.ndarray,
    cutoff: float = 1.5,
    min_sep: int = 3,
) -> float:
    """Compute absolute contact order (ACO).

    ACO = mean(|i - j|) over all contacts (i, j) with distance < cutoff.

    Args:
        positions: (N, D) array of positions.
        cutoff: Contact distance cutoff.
        min_sep: Minimum sequence separation to count as contact.

    Returns:
        Absolute contact order.
    """
    n = len(positions)
    loop_lengths = []
    for i in range(n):
        for j in range(i + min_sep, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d <= cutoff:
                loop_lengths.append(j - i)
    if not loop_lengths:
        return 0.0
    return float(np.mean(loop_lengths))


def count_clashes(
    positions: np.ndarray,
    threshold: float = 0.8,
    min_sep: int = 3,
) -> int:
    """Count pairs of residues with distances below threshold.

    Args:
        positions: (N, D) array.
        threshold: Hard clash cutoff (lattice units).
        min_sep: Minimum sequence separation.

    Returns:
        Number of clashing pairs.
    """
    from ..geom.classical_refine import find_clashes
    return len(find_clashes(positions, threshold=threshold, min_sep=min_sep))


def ramachandran_quality(turns: List[int]) -> float:
    """Heuristic Ramachandran quality: fraction of turns without back-tracking.

    On a 2D lattice, back-tracking (U-turn = opposite direction) is always
    an "outlier" conformation.

    Args:
        turns: List of integer turn codes {0,1,2,3}.

    Returns:
        Fraction of turns in allowed regions [0, 1].
    """
    if len(turns) <= 1:
        return 1.0
    bad = 0
    for i in range(1, len(turns)):
        # Back-track: turn code differs by 2 (opposite directions)
        if abs(turns[i] - turns[i - 1]) == 2:
            bad += 1
    return 1.0 - bad / (len(turns) - 1)


def score_model(
    positions: np.ndarray,
    total_energy: float,
    turns: Optional[List[int]] = None,
    per_region_confidence: Optional[Dict[str, float]] = None,
    clash_threshold: float = 0.8,
    contact_cutoff: float = 1.5,
    contact_min_sep: int = 3,
) -> ModelScore:
    """Compute a coarse-grained quality score for an assembled model.

    Args:
        positions: (N, D) array of Cα positions.
        total_energy: Raw Hamiltonian energy.
        turns: Optional list of turn codes for Ramachandran check.
        per_region_confidence: Per-fragment confidence scores from stitching.
        clash_threshold: Hard clash detection distance.
        contact_cutoff: Contact counting distance.
        contact_min_sep: Minimum sequence separation for contacts.

    Returns:
        ModelScore.
    """
    rg = radius_of_gyration(positions)
    co = contact_order(positions, cutoff=contact_cutoff, min_sep=contact_min_sep)
    clashes = count_clashes(positions, threshold=clash_threshold, min_sep=contact_min_sep)

    # Count total contacts
    n = len(positions)
    n_contacts = 0
    for i in range(n):
        for j in range(i + contact_min_sep, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d <= contact_cutoff:
                n_contacts += 1

    rama_pct = ramachandran_quality(turns) if turns else 1.0
    region_conf = per_region_confidence or {}
    overall_conf = float(np.mean(list(region_conf.values()))) if region_conf else 1.0

    return ModelScore(
        total_energy=float(total_energy),
        radius_of_gyration=rg,
        contact_order=co,
        clash_count=clashes,
        n_contacts=n_contacts,
        ramachandran_pct=rama_pct,
        per_region_confidence=region_conf,
        overall_confidence=overall_conf,
    )
