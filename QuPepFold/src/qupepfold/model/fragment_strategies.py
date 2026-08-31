"""Fragment generation strategies for different protein analysis modes.

Provides pluggable strategies for splitting proteins into overlapping fragments:
- FixedWindowStrategy: Sliding window with uniform overlap (current default)
- DisorderAwareStrategy: Prioritise intrinsically disordered regions
- DomainBoundaryStrategy: Split at predicted domain/SS boundaries
- UserDefinedStrategy: Explicit user-supplied residue ranges
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict
import math

from ..types import FoldConfig, FragmentSpec


# ── Amino-acid property tables ──────────────────────────────────────────────

# Kyte-Doolittle hydrophobicity scale (higher = more hydrophobic)
_KD_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Disorder-promoting residues (from Dunker et al.)
_DISORDER_PROMOTING = set("AGQSPEKRD")
_ORDER_PROMOTING = set("WFYILMVNC")


class FragmentStrategy(ABC):
    """Abstract base class for fragment generation strategies."""

    @abstractmethod
    def generate(self, sequence: str, config: FoldConfig) -> List[FragmentSpec]:
        """Generate a list of FragmentSpec objects for the given sequence.

        Args:
            sequence: Full amino acid sequence (uppercase, validated).
            config: FoldConfig with fragment_length, overlap_turns, etc.

        Returns:
            List of FragmentSpec, ordered by start_res, with priority scores.
        """
        ...

    @staticmethod
    def _clamp_fragments(
        fragments: List[FragmentSpec],
        sequence: str,
        config: FoldConfig,
    ) -> List[FragmentSpec]:
        """Ensure every fragment has a valid sequence slice and overlap flags."""
        out = []
        for f in fragments:
            s = max(0, f.start_res)
            e = min(len(sequence), f.end_res)
            if e - s < 3:
                continue  # too short
            out.append(FragmentSpec(
                start_res=s,
                end_res=e,
                sequence=sequence[s:e],
                overlap_left_turns=f.overlap_left_turns,
                overlap_right_turns=f.overlap_right_turns,
                priority=f.priority,
                parent_protein_id=f.parent_protein_id,
                preprocessing_metadata=f.preprocessing_metadata,
            ))
        return out


# ── Concrete strategies ─────────────────────────────────────────────────────

class FixedWindowStrategy(FragmentStrategy):
    """Sliding window with uniform overlap (the original QuPepFold approach)."""

    def generate(self, sequence: str, config: FoldConfig) -> List[FragmentSpec]:
        n = len(sequence)
        frag_len = config.fragment_length
        overlap_turns = config.overlap_turns

        if n < frag_len:
            return [FragmentSpec(
                start_res=0, end_res=n, sequence=sequence,
                overlap_left_turns=0, overlap_right_turns=0,
            )]

        overlap_residues = overlap_turns + 1
        stride = frag_len - overlap_residues
        if stride <= 0:
            raise ValueError(
                f"overlap_turns ({overlap_turns}) too large for "
                f"fragment_length ({frag_len}). Need overlap_turns < fragment_length - 1"
            )

        fragments: List[FragmentSpec] = []
        start = 0
        frag_idx = 0

        while start < n:
            end = min(start + frag_len, n)
            is_first = frag_idx == 0
            next_start = start + stride
            has_next = (next_start < n) and (next_start + frag_len <= n + overlap_residues)

            overlap_left = 0 if is_first else overlap_turns
            overlap_right = overlap_turns if has_next else 0

            fragments.append(FragmentSpec(
                start_res=start, end_res=end,
                sequence=sequence[start:end],
                overlap_left_turns=overlap_left,
                overlap_right_turns=overlap_right,
            ))

            start += stride
            frag_idx += 1

            if start < n and start + frag_len > n:
                end = n
                if end - start < 3:
                    break
                fragments.append(FragmentSpec(
                    start_res=start, end_res=end,
                    sequence=sequence[start:end],
                    overlap_left_turns=overlap_turns,
                    overlap_right_turns=0,
                ))
                break

        return fragments


class DisorderAwareStrategy(FragmentStrategy):
    """Score residues by disorder propensity; prioritise disordered regions.

    Uses a simple composition-based disorder score (window average of
    disorder-promoting vs order-promoting residue frequency).  This is a
    lightweight heuristic — for production use, integrate IUPred or ESMFold
    confidence scores.
    """

    def __init__(self, disorder_window: int = 11):
        self.disorder_window = disorder_window

    def _disorder_profile(self, sequence: str) -> List[float]:
        """Compute per-residue disorder score in [0, 1]."""
        n = len(sequence)
        half = self.disorder_window // 2
        scores = []
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window = sequence[lo:hi]
            d_count = sum(1 for aa in window if aa in _DISORDER_PROMOTING)
            score = d_count / len(window)
            scores.append(score)
        return scores

    def generate(self, sequence: str, config: FoldConfig) -> List[FragmentSpec]:
        n = len(sequence)
        frag_len = config.fragment_length
        overlap_turns = config.overlap_turns

        # Fall back to fixed window if too short
        if n < frag_len:
            return FixedWindowStrategy().generate(sequence, config)

        disorder = self._disorder_profile(sequence)
        overlap_residues = overlap_turns + 1
        stride = frag_len - overlap_residues
        if stride <= 0:
            raise ValueError("overlap too large for fragment_length")

        # Score every possible window
        window_scores: List[Tuple[int, float]] = []
        for s in range(0, n - frag_len + 1):
            avg = sum(disorder[s:s + frag_len]) / frag_len
            window_scores.append((s, avg))

        # Greedy cover: pick highest-scoring non-overlapping-enough windows
        window_scores.sort(key=lambda x: -x[1])  # highest disorder first
        covered = set()
        chosen_starts: List[int] = []

        for s, score in window_scores:
            # Check coverage
            rng = set(range(s, s + frag_len))
            uncovered = rng - covered
            if len(uncovered) < stride:
                continue  # already well-covered
            chosen_starts.append(s)
            covered.update(rng)
            if len(covered) >= n:
                break

        # Ensure full coverage by adding any gaps
        chosen_starts.sort()
        all_covered = set()
        for s in chosen_starts:
            all_covered.update(range(s, min(s + frag_len, n)))
        gaps = set(range(n)) - all_covered
        while gaps:
            g = min(gaps)
            s = max(0, g - frag_len // 2)
            e = min(n, s + frag_len)
            s = max(0, e - frag_len)
            chosen_starts.append(s)
            chosen_starts.sort()
            gaps -= set(range(s, e))

        chosen_starts = sorted(set(chosen_starts))

        # Build FragmentSpec list with disorder-based priority
        fragments: List[FragmentSpec] = []
        for idx, s in enumerate(chosen_starts):
            e = min(s + frag_len, n)
            is_first = idx == 0
            is_last = idx == len(chosen_starts) - 1
            avg_dis = sum(disorder[s:e]) / (e - s)

            fragments.append(FragmentSpec(
                start_res=s, end_res=e,
                sequence=sequence[s:e],
                overlap_left_turns=0 if is_first else overlap_turns,
                overlap_right_turns=0 if is_last else overlap_turns,
                priority=avg_dis,
                preprocessing_metadata={"strategy": "disorder", "avg_disorder": avg_dis},
            ))

        return self._clamp_fragments(fragments, sequence, config)


class DomainBoundaryStrategy(FragmentStrategy):
    """Split at predicted secondary-structure boundaries.

    Uses a simplified Chou-Fasman-like helix/sheet propensity to identify
    regions that are likely coil/loop and therefore good fragment boundaries.
    """

    # Simplified propensities (higher = more structured)
    _HELIX_PROP = {
        "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
        "E": 1.51, "Q": 1.11, "G": 0.57, "H": 1.00, "I": 1.08,
        "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
        "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
    }

    def _structure_profile(self, sequence: str, window: int = 7) -> List[float]:
        """Smoothed structure propensity (higher = more structured)."""
        n = len(sequence)
        half = window // 2
        profile = []
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            s = sum(self._HELIX_PROP.get(aa, 1.0) for aa in sequence[lo:hi])
            profile.append(s / (hi - lo))
        return profile

    def generate(self, sequence: str, config: FoldConfig) -> List[FragmentSpec]:
        n = len(sequence)
        frag_len = config.fragment_length
        overlap_turns = config.overlap_turns

        if n < frag_len:
            return FixedWindowStrategy().generate(sequence, config)

        profile = self._structure_profile(sequence)
        overlap_residues = overlap_turns + 1

        # Find local minima in structure propensity (= good break points)
        break_points = [0]
        min_dist = frag_len - overlap_residues
        for i in range(min_dist, n - min_dist):
            window = profile[max(0, i - 2):i + 3]
            if profile[i] == min(window) and profile[i] < 1.0:
                if not break_points or i - break_points[-1] >= min_dist:
                    break_points.append(i)
        if break_points[-1] != n:
            break_points.append(n)

        # Build fragments between break points
        fragments: List[FragmentSpec] = []
        for idx in range(len(break_points) - 1):
            s = break_points[idx]
            e = break_points[idx + 1]
            # Extend to at least frag_len
            if e - s < frag_len and idx < len(break_points) - 2:
                e = min(n, s + frag_len)
            e = min(n, e)
            if e - s < 3:
                continue
            is_first = idx == 0
            is_last = idx == len(break_points) - 2
            avg_struct = sum(profile[s:e]) / (e - s)

            fragments.append(FragmentSpec(
                start_res=s, end_res=e,
                sequence=sequence[s:e],
                overlap_left_turns=0 if is_first else overlap_turns,
                overlap_right_turns=0 if is_last else overlap_turns,
                priority=1.0 - avg_struct,  # Less structured → higher priority
                preprocessing_metadata={"strategy": "domain", "avg_structure": avg_struct},
            ))

        if not fragments:
            return FixedWindowStrategy().generate(sequence, config)

        return self._clamp_fragments(fragments, sequence, config)


class UserDefinedStrategy(FragmentStrategy):
    """Accept explicit user-supplied residue ranges.

    Args:
        regions: List of (start, end) tuples (0-based, exclusive end).
    """

    def __init__(self, regions: List[Tuple[int, int]]):
        self.regions = regions

    def generate(self, sequence: str, config: FoldConfig) -> List[FragmentSpec]:
        overlap_turns = config.overlap_turns
        fragments: List[FragmentSpec] = []

        for idx, (s, e) in enumerate(sorted(self.regions)):
            s = max(0, s)
            e = min(len(sequence), e)
            if e - s < 3:
                continue
            is_first = idx == 0
            is_last = idx == len(self.regions) - 1

            fragments.append(FragmentSpec(
                start_res=s, end_res=e,
                sequence=sequence[s:e],
                overlap_left_turns=0 if is_first else overlap_turns,
                overlap_right_turns=0 if is_last else overlap_turns,
                priority=0.0,
                preprocessing_metadata={"strategy": "user_defined"},
            ))

        return self._clamp_fragments(fragments, sequence, config)


# ── Strategy factory ────────────────────────────────────────────────────────

_STRATEGY_MAP: Dict[str, type] = {
    "fixed_window": FixedWindowStrategy,
    "disorder": DisorderAwareStrategy,
    "domain": DomainBoundaryStrategy,
    # "user_defined" requires regions arg, handled separately
}


def get_strategy(name: str, **kwargs) -> FragmentStrategy:
    """Get a FragmentStrategy instance by name.

    Args:
        name: One of "fixed_window", "disorder", "domain", "user_defined".
        **kwargs: Extra args passed to the strategy constructor.

    Returns:
        FragmentStrategy instance.
    """
    if name == "user_defined":
        regions = kwargs.get("regions", [])
        return UserDefinedStrategy(regions=regions)
    cls = _STRATEGY_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown fragment strategy: '{name}'. "
                         f"Available: {list(_STRATEGY_MAP.keys()) + ['user_defined']}")
    return cls(**kwargs)
