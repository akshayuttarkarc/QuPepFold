"""Error mitigation utilities for quantum measurement results.

Provides:
  - ReadoutMitigation: calibration-matrix-based readout error correction
  - PostSelectionFilter: discard physically invalid bitstrings
  - MitigationPipeline: chain multiple mitigation steps
"""

from typing import Dict, List, Optional, Callable
import numpy as np


class ReadoutMitigation:
    """Calibration-matrix-based readout error mitigation.

    Builds a calibration matrix A where A[i,j] = P(measure i | prepared j).
    Applies the inverse A^{-1} to observed probability distributions to
    recover the ideal (noise-free) distribution.

    Note: For production, use M3 or Qiskit's built-in mitigation.
    This implementation is a lightweight approximation.

    Args:
        n_qubits: Number of qubits.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self._cal_matrix: Optional[np.ndarray] = None

    def calibrate(self, cal_probs: Dict[str, Dict[str, float]]) -> None:
        """Build calibration matrix from calibration experiments.

        Args:
            cal_probs: Dict mapping prepared_state → measured probability dict.
                       Each prepared_state is a bitstring (length n_qubits).
                       Example: {"00": {"00": 0.95, "01": 0.03, "10": 0.02},
                                 "01": {"01": 0.93, ...}}
        """
        n = self.n_states
        A = np.zeros((n, n))

        for prep_bits, meas_probs in cal_probs.items():
            j = int(prep_bits, 2)
            for meas_bits, prob in meas_probs.items():
                i = int(meas_bits, 2)
                if i < n and j < n:
                    A[i, j] = prob

        # Normalize columns
        col_sums = A.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        A = A / col_sums

        # Store pseudoinverse for stability
        self._cal_matrix = np.linalg.pinv(A)

    def apply(self, prob_dict: Dict[str, float]) -> Dict[str, float]:
        """Apply readout mitigation to a probability dictionary.

        Args:
            prob_dict: Measured probability dict {bitstring: prob}.

        Returns:
            Mitigated probability dict (negative probabilities clipped to 0).
        """
        if self._cal_matrix is None:
            return prob_dict  # No calibration → pass through

        n = self.n_states
        prob_vec = np.zeros(n)
        for bits, p in prob_dict.items():
            idx = int(bits, 2)
            if idx < n:
                prob_vec[idx] = p

        mitigated = self._cal_matrix @ prob_vec
        mitigated = np.clip(mitigated, 0, None)  # Remove negatives

        # Renormalise
        total = mitigated.sum()
        if total > 1e-9:
            mitigated /= total

        return {
            format(i, f"0{self.n_qubits}b"): float(mitigated[i])
            for i in range(n)
            if mitigated[i] > 1e-10
        }


class PostSelectionFilter:
    """Discard bitstrings that violate physical constraints.

    Physical constraints checked:
      1. Self-intersection: positions on the lattice overlap
      2. Boundary consistency: overlap bits match warm-start boundaries

    Args:
        fragment: FragmentSpec being refined.
        warmstart_bits: Optional boundary bits to enforce.
    """

    def __init__(self, fragment, warmstart_bits: Optional[str] = None):
        self.fragment = fragment
        self.warmstart_bits = warmstart_bits
        self.n_left = fragment.overlap_left_turns * 2
        self.n_right = fragment.overlap_right_turns * 2

    def __call__(self, prob_dict: Dict[str, float]) -> Dict[str, float]:
        """Filter invalid bitstrings from a probability distribution.

        Args:
            prob_dict: {bitstring: probability}.

        Returns:
            Filtered and renormalised probability dict.
        """
        from ..model.encoding import decode_turns
        from ..model.lattice import trace_positions, count_overlaps

        filtered = {}
        for bits, prob in prob_dict.items():
            # Boundary consistency check
            if self.warmstart_bits:
                if self.n_left > 0:
                    if not bits.startswith(self.warmstart_bits[:self.n_left]):
                        continue
                if self.n_right > 0:
                    if not bits.endswith(self.warmstart_bits[-self.n_right:]):
                        continue

            # Self-intersection check
            try:
                turns = decode_turns(bits)
                positions = trace_positions(turns)
                if count_overlaps(positions) > 0:
                    continue
            except Exception:
                continue

            filtered[bits] = prob

        # Renormalise
        total = sum(filtered.values())
        if total > 1e-9:
            return {b: p / total for b, p in filtered.items()}
        return prob_dict  # All filtered — return original to avoid empty


class MitigationPipeline:
    """Chain multiple mitigation steps.

    Steps are applied in order. Each step is a callable:
        mitigated_probs = step(prob_dict)

    Args:
        steps: List of mitigation callables.
    """

    def __init__(self, steps: List[Callable[[Dict[str, float]], Dict[str, float]]]):
        self.steps = steps

    def apply(self, prob_dict: Dict[str, float]) -> Dict[str, float]:
        """Apply all mitigation steps sequentially."""
        for step in self.steps:
            prob_dict = step(prob_dict)
        return prob_dict

    def __call__(self, prob_dict: Dict[str, float]) -> Dict[str, float]:
        return self.apply(prob_dict)
