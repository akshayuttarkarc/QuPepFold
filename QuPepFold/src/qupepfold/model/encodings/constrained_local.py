"""Constrained Local encoding: turn-based with additional restraint terms.

Extends TurnEncoding with:
  1. Clash prevention: infinite penalty for self-intersecting states
  2. Overlap consistency: soft penalty when boundary turns differ from neighbour
  3. Secondary structure rewards: reward turns consistent with predicted SS

This encoding has the same qubit count as TurnEncoding (2 × n_turns) but
produces a sparser, better-conditioned energy table by eliminating invalid
states up-front and adding physicality terms.

Args:
    clash_penalty: Hard penalty for self-intersecting states (default: 1e6).
    ss_reward: Reward per turn consistent with predicted SS (default: -0.5).
    overlap_penalty: Per-bit overlap mismatch penalty (default: 5.0).
"""

from typing import Dict, List, Optional
from .base import EncodingScheme


class ConstrainedLocalEncoding(EncodingScheme):
    """Turn encoding with physicality constraints and SS rewards."""

    def __init__(
        self,
        clash_penalty: float = 1e6,
        ss_reward: float = -0.5,
        overlap_penalty: float = 5.0,
    ):
        self.clash_penalty = clash_penalty
        self.ss_reward = ss_reward
        self.overlap_penalty = overlap_penalty

    @property
    def name(self) -> str:
        return "constrained_local"

    def qubit_estimate(self, n_turns: int) -> int:
        # Same as turn encoding
        return 2 * n_turns

    def term_count(self, n_turns: int, n_residues: int) -> int:
        # MJ contacts + overlap consistency + SS reward
        return (
            n_residues * (n_residues - 1) // 2  # MJ contacts
            + n_turns                             # backbone penalties
            + n_turns - 1                         # consecutive overlap terms
            + n_turns                             # SS reward terms
        )

    def decode(self, bitstring: str) -> List[int]:
        """Decode 2-bit groups to turn codes.

        Identical to TurnEncoding — constraints are in the energy function.
        """
        n = len(bitstring)
        if n % 2 != 0:
            raise ValueError(f"Bitstring length {n} must be even")
        turns = []
        for i in range(0, n, 2):
            high = int(bitstring[i])
            low = int(bitstring[i + 1])
            turns.append(high * 2 + low)
        return turns

    def penalty_summary(self) -> Dict[str, float]:
        return {
            "clash_penalty": self.clash_penalty,
            "ss_reward": self.ss_reward,
            "overlap_penalty": self.overlap_penalty,
            "bits_per_turn": 2,
            "qubit_scale": "2 × n_turns",
        }

    def apply_clash_filter(self, energy_table_energies, fragment) -> None:
        """Set infinite energy for all self-intersecting states (in-place).

        Called by build_energy_table() when encoding == "constrained_local".

        Args:
            energy_table_energies: numpy array of shape (2^n_bits,)
            fragment: FragmentSpec
        """
        import numpy as np
        from ...model.lattice import trace_positions, count_overlaps
        from ...model.encoding import index_to_bitstring

        n_bits = fragment.n_bits
        n_states = 2 ** n_bits

        for idx in range(n_states):
            if energy_table_energies[idx] >= 1e9:
                continue  # already filtered
            bits = index_to_bitstring(idx, n_bits)
            turns = self.decode(bits)
            positions = trace_positions(turns)
            if count_overlaps(positions) > 0:
                energy_table_energies[idx] = np.inf

    def apply_ss_rewards(
        self,
        energy_table_energies,
        fragment,
        ss_profile: Optional[List[int]] = None,
    ) -> None:
        """Add secondary-structure rewards for consistent turns (in-place).

        Args:
            energy_table_energies: numpy array of shape (2^n_bits,).
            fragment: FragmentSpec.
            ss_profile: Per-residue SS class list (0=coil, 1=helix, 2=sheet).
                        If None, no rewards applied.
        """
        if ss_profile is None:
            return

        import numpy as np
        from ...model.encoding import index_to_bitstring

        n_bits = fragment.n_bits
        n_states = 2 ** n_bits
        start = fragment.start_res

        for idx in range(n_states):
            if energy_table_energies[idx] >= 1e9:
                continue
            bits = index_to_bitstring(idx, n_bits)
            turns = self.decode(bits)
            reward = 0.0
            for t_idx, turn in enumerate(turns):
                res_idx = start + t_idx
                if res_idx >= len(ss_profile):
                    break
                ss = ss_profile[res_idx]
                if ss == 1:  # helix: favour straight turns (0 or 2)
                    if turn in (0, 2):
                        reward += self.ss_reward
                elif ss == 2:  # sheet: favour alternating turns
                    if turn in (1, 3):
                        reward += self.ss_reward
            energy_table_energies[idx] += reward
