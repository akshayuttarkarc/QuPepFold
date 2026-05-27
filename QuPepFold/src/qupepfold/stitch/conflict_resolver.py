"""Conflict resolver for fragment assembly.

When adjacent fragments disagree on overlap turn codes, the resolver
applies one of three strategies to pick the final turns for the
overlap region and reports per-region confidence.

Strategies:
  "winner_take_all" : Use the lower-energy fragment's overlap bits
  "vote"            : Majority vote among top candidates from both sides
  "resubmit"        : Flag for VQE re-run (caller must handle)
"""

from typing import List, Optional, Tuple, Dict
import numpy as np

from ..types import FragmentSpec, FragmentCandidate


# ── Confidence scoring ────────────────────────────────────────────────────────

def overlap_agreement_score(
    left_bits: str,
    right_bits: str,
    overlap_left_turns: int,
) -> float:
    """Compute overlap agreement as fraction of matching bits.

    Args:
        left_bits: Bitstring of the left fragment candidate.
        right_bits: Bitstring of the right fragment candidate.
        overlap_left_turns: Number of shared turns between fragments.

    Returns:
        Agreement score in [0, 1]. 1.0 = perfect match.
    """
    n_overlap_bits = overlap_left_turns * 2
    if n_overlap_bits == 0:
        return 1.0

    left_overlap = left_bits[-n_overlap_bits:]
    right_overlap = right_bits[:n_overlap_bits]

    if len(left_overlap) != len(right_overlap):
        return 0.0

    matches = sum(a == b for a, b in zip(left_overlap, right_overlap))
    return matches / n_overlap_bits


def region_confidence(
    candidates: List[FragmentCandidate],
    energy_range: float = 10.0,
) -> float:
    """Estimate confidence for a fragment region based on candidate spread.

    High confidence = one clear winner, all candidates clustered.
    Low confidence = many near-degenerate states, VQE uncertain.

    Args:
        candidates: List of FragmentCandidate objects (sorted by energy).
        energy_range: Reference energy range for normalisation.

    Returns:
        Confidence in [0, 1]. 1.0 = certain.
    """
    if not candidates:
        return 0.0
    if len(candidates) == 1:
        return 0.9

    best_energy = candidates[0].energy
    second_energy = candidates[1].energy
    gap = second_energy - best_energy

    # Normalise gap to [0, 1]
    conf = min(1.0, gap / max(energy_range, 1e-6))
    return round(conf, 4)


# ── Conflict resolver ─────────────────────────────────────────────────────────

class ConflictResolver:
    """Resolve disagreements in overlap regions between adjacent fragment candidates.

    Args:
        strategy: "winner_take_all", "vote", or "resubmit".
        mismatch_threshold: Minimum mismatch fraction to trigger resolution.
    """

    def __init__(
        self,
        strategy: str = "winner_take_all",
        mismatch_threshold: float = 0.0,
    ):
        valid = {"winner_take_all", "vote", "resubmit"}
        if strategy not in valid:
            raise ValueError(f"strategy must be one of {valid}, got {strategy!r}")
        self.strategy = strategy
        self.mismatch_threshold = mismatch_threshold

    def resolve(
        self,
        left_candidates: List[FragmentCandidate],
        right_candidates: List[FragmentCandidate],
        overlap_turns: int,
        fragment_idx: int = 0,
    ) -> Tuple[str, str, Dict]:
        """Resolve conflict between left and right fragment candidates.

        Args:
            left_candidates: Candidates from the left fragment (sorted by energy).
            right_candidates: Candidates from the right fragment.
            overlap_turns: Number of shared turns.
            fragment_idx: Index of the right fragment (for logging).

        Returns:
            Tuple of (left_bits_resolved, right_bits_resolved, report_dict).
            The report dict contains: agreement, strategy_used, confidence,
            needs_resubmit.
        """
        if not left_candidates or not right_candidates:
            return (
                left_candidates[0].bits if left_candidates else "",
                right_candidates[0].bits if right_candidates else "",
                {"agreement": 1.0, "strategy_used": "passthrough", "needs_resubmit": False},
            )

        left_best = left_candidates[0]
        right_best = right_candidates[0]
        agreement = overlap_agreement_score(left_best.bits, right_best.bits, overlap_turns)

        report: Dict = {
            "fragment_idx": fragment_idx,
            "agreement": agreement,
            "strategy_used": self.strategy,
            "left_energy": left_best.energy,
            "right_energy": right_best.energy,
            "needs_resubmit": False,
            "left_confidence": region_confidence(left_candidates),
            "right_confidence": region_confidence(right_candidates),
        }

        # Below threshold → no conflict
        if agreement >= 1.0 - self.mismatch_threshold:
            report["strategy_used"] = "no_conflict"
            return left_best.bits, right_best.bits, report

        if self.strategy == "winner_take_all":
            left_out, right_out = self._winner_take_all(left_best, right_best, overlap_turns)
        elif self.strategy == "vote":
            left_out, right_out = self._vote(
                left_candidates, right_candidates, overlap_turns
            )
        elif self.strategy == "resubmit":
            report["needs_resubmit"] = True
            left_out = left_best.bits
            right_out = right_best.bits
        else:
            left_out, right_out = left_best.bits, right_best.bits

        return left_out, right_out, report

    def _winner_take_all(
        self,
        left: FragmentCandidate,
        right: FragmentCandidate,
        overlap_turns: int,
    ) -> Tuple[str, str]:
        """Use the lower-energy fragment's overlap bits for both."""
        n_ov = overlap_turns * 2
        if left.energy <= right.energy:
            # Left wins: adopt left's tail into right's head
            winner_ov = left.bits[-n_ov:] if n_ov > 0 else ""
            right_new = winner_ov + right.bits[n_ov:]
            return left.bits, right_new
        else:
            # Right wins: adopt right's head into left's tail
            winner_ov = right.bits[:n_ov] if n_ov > 0 else ""
            left_new = left.bits[:-n_ov] + winner_ov if n_ov > 0 else left.bits
            return left_new, right.bits

    def _vote(
        self,
        left_candidates: List[FragmentCandidate],
        right_candidates: List[FragmentCandidate],
        overlap_turns: int,
    ) -> Tuple[str, str]:
        """Majority vote on the overlap bits across top-K candidates."""
        n_ov = overlap_turns * 2
        if n_ov == 0:
            return left_candidates[0].bits, right_candidates[0].bits

        # Collect overlap bits from all candidates
        vote_counts: Dict[str, float] = {}

        for c in left_candidates:
            ov = c.bits[-n_ov:]
            vote_counts[ov] = vote_counts.get(ov, 0) + (1 / max(c.energy, 1e-6))

        for c in right_candidates:
            ov = c.bits[:n_ov]
            vote_counts[ov] = vote_counts.get(ov, 0) + (1 / max(c.energy, 1e-6))

        # Plurality winner
        winner_ov = max(vote_counts, key=lambda k: vote_counts[k])

        left_new = left_candidates[0].bits[:-n_ov] + winner_ov
        right_new = winner_ov + right_candidates[0].bits[n_ov:]
        return left_new, right_new

    def resolve_chain(
        self,
        all_candidates: List[List[FragmentCandidate]],
        fragments: List[FragmentSpec],
    ) -> Tuple[List[str], List[Dict]]:
        """Resolve conflicts across the entire fragment chain.

        Args:
            all_candidates: Per-fragment candidate lists.
            fragments: FragmentSpec list.

        Returns:
            Tuple of (resolved_bits_per_fragment, report_list).
        """
        n = len(all_candidates)
        resolved_bits = [cands[0].bits for cands in all_candidates]
        reports = []

        for i in range(1, n):
            overlap_turns = fragments[i].overlap_left_turns
            left_cands = [
                FragmentCandidate(
                    bits=resolved_bits[i - 1],
                    energy=all_candidates[i - 1][0].energy,
                    probability=all_candidates[i - 1][0].probability,
                )
            ]
            left_new, right_new, report = self.resolve(
                left_cands,
                all_candidates[i],
                overlap_turns,
                fragment_idx=i,
            )
            resolved_bits[i - 1] = left_new
            resolved_bits[i] = right_new
            reports.append(report)

        return resolved_bits, reports
