"""Turn encoding: 2 bits per conformational turn, 4 possible directions.

This is the original QuPepFold encoding:
  - Each lattice turn ∈ {0, 1, 2, 3} → 2-bit binary (big-endian)
  - n qubits = 2 × n_turns
  - 4^n_turns total conformational states

The 4 turn directions on a 2D lattice are:
  0 (00) → +x  (East)
  1 (01) → +y  (North)
  2 (10) → -x  (West)
  3 (11) → -y  (South)
"""

from typing import Dict, List
from .base import EncodingScheme


class TurnEncoding(EncodingScheme):
    """Original 2-bit-per-turn encoding used in the PLOS One paper."""

    @property
    def name(self) -> str:
        return "turn"

    def qubit_estimate(self, n_turns: int) -> int:
        return 2 * n_turns

    def term_count(self, n_turns: int, n_residues: int) -> int:
        # MJ contact terms: O(n^2), backbone terms: O(n)
        return n_residues * (n_residues - 1) // 2 + n_turns

    def decode(self, bitstring: str) -> List[int]:
        """Decode 2-bit groups to turn codes.

        Args:
            bitstring: Binary string of length 2*n_turns.

        Returns:
            List of int turn codes in {0, 1, 2, 3}.

        Example:
            >>> TurnEncoding().decode("0110")
            [1, 2]
        """
        n = len(bitstring)
        if n % 2 != 0:
            raise ValueError(f"Bitstring length {n} must be even for turn encoding")
        turns = []
        for i in range(0, n, 2):
            high = int(bitstring[i])
            low = int(bitstring[i + 1])
            turns.append(high * 2 + low)
        return turns

    def penalty_summary(self) -> Dict[str, float]:
        return {
            "bits_per_turn": 2,
            "states_per_turn": 4,
            "qubit_scale": "2 × n_turns",
        }
