"""HP (Hydrophobic-Polar) encoding: binary H/P classification per residue.

The HP model simplifies amino acids to two types:
  H (Hydrophobic): high-energy penalty when not in contact with other H residues
  P (Polar):       low-energy neutral contribution

Encoding:
  - 1 bit per turn (left = 0, right = 1 on a 2D lattice walk)
  - For 3D: 2 bits per turn (6 possible directions compressed to 4 + penalty)
  - n_qubits = n_turns  (vs 2×n_turns for TurnEncoding → 50% fewer qubits)

HP energy function:
  E = -1 × (number of H-H non-covalent contacts)
  where contact means adjacent lattice cells not bonded in sequence.

The HP turn codes are different from the turn encoding:
  Turn encoding uses {0,1,2,3} for absolute directions.
  HP uses relative turns {0,1} for (straight, turn-left):
    0 → same direction as previous step
    1 → turn 90° left
  But for practical use we map to 4 directions:
    0 (0-bit) → same as previous
    1 (1-bit) → turn 90° (alternating)
  We keep 2-bit representation but treat as {straight, left, right, back=penalty}
  with back transitions penalised strongly.

This implementation:
  - Uses 2 bits per turn internally for compatibility with existing energy tables
  - Adds a strong self-reversal penalty (back = {0→2, 1→3, 2→0, 3→1})
  - HP classification based on Kyte-Doolittle hydrophobicity (KD ≥ 0 = H)
"""

from typing import Dict, List
import numpy as np
from .base import EncodingScheme

# Kyte-Doolittle scale: residues with score >= 0 are "Hydrophobic"
_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def is_hydrophobic(aa: str) -> bool:
    """Return True if amino acid is classified as hydrophobic."""
    return _KD.get(aa.upper(), 0.0) >= 0.0


class HPEncoding(EncodingScheme):
    """HP lattice model encoding.

    Uses 2 bits per turn for compatibility with the energy table infrastructure,
    but penalises back-tracking moves (U-turns) to emulate a true HP walk.

    Args:
        back_penalty: Energy penalty for U-turn moves (default 100.0).
        hp_contact_energy: Energy reward for each H-H non-covalent contact (default -1.0).
    """

    def __init__(self, back_penalty: float = 100.0, hp_contact_energy: float = -1.0):
        self.back_penalty = back_penalty
        self.hp_contact_energy = hp_contact_energy

    @property
    def name(self) -> str:
        return "hp"

    def qubit_estimate(self, n_turns: int) -> int:
        # Still 2 bits per turn for compatibility; actual HP only needs 1 bit
        # but 2-bit keeps the energy table infrastructure unchanged
        return 2 * n_turns

    def term_count(self, n_turns: int, n_residues: int) -> int:
        # H-H contact terms (only between H residues) + back-penalty terms
        n_contacts_upper = n_residues * (n_residues - 1) // 2
        n_back = n_turns  # one per turn
        return n_contacts_upper + n_back

    def decode(self, bitstring: str) -> List[int]:
        """Decode 2-bit groups to turn codes {0,1,2,3}.

        Same as TurnEncoding.decode() — the HP energy distinction is handled
        in the energy function, not in the turn representation.
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
            "back_penalty": self.back_penalty,
            "hp_contact_energy": self.hp_contact_energy,
            "bits_per_turn": 2,
            "qubit_scale": "2 × n_turns (HP model)",
        }

    def hp_classification(self, sequence: str) -> List[bool]:
        """Return per-residue boolean HP classification (True = H)."""
        return [is_hydrophobic(aa) for aa in sequence]

    def hp_energy(self, turns: List[int], sequence: str) -> float:
        """Compute HP lattice energy for a given conformation.

        Args:
            turns: List of turn codes.
            sequence: Amino acid sequence.

        Returns:
            HP energy (negative = stable).
        """
        from ...model.lattice import trace_positions, contacts

        hp = self.hp_classification(sequence)
        positions = trace_positions(turns)

        # Back-tracking penalty
        energy = 0.0
        for i in range(1, len(turns)):
            # Back = opposite of previous direction
            # {0↔2, 1↔3}
            if (turns[i] ^ 2) == turns[i - 1]:
                energy += self.back_penalty

        # H-H contact energy
        contact_pairs = contacts(positions, min_sep=3, cutoff=1.5)
        for i, j in contact_pairs:
            if hp[i] and hp[j]:
                energy += self.hp_contact_energy

        return energy
