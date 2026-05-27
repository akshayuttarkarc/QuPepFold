"""Adaptive shot allocation across fragments.

Initially distributes shots uniformly. After the first VQE pass, reallocates
based on per-fragment energy variance and convergence rate so that
high-variance or slow-converging fragments receive more shots.

Total shot budget is conserved.
"""

from typing import List, Dict
import numpy as np


class AdaptiveShotAllocator:
    """Budget-conserving adaptive shot allocator.

    Usage::

        allocator = AdaptiveShotAllocator(
            n_fragments=5, total_shots=10000, min_shots=200
        )
        # Initial allocation
        shots_per_frag = allocator.initial_allocation()

        # After first VQE pass, update and reallocate
        allocator.update(variances=[0.5, 1.2, 0.3, 0.9, 0.1],
                         convergence_rates=[0.9, 0.6, 0.95, 0.7, 0.98])
        shots_per_frag = allocator.current_allocation()

    Args:
        n_fragments: Number of fragments.
        total_shots: Total shot budget (conserved across reallocation).
        min_shots: Minimum shots per fragment (default: 100).
    """

    def __init__(self, n_fragments: int, total_shots: int, min_shots: int = 100):
        self.n_fragments = n_fragments
        self.total_shots = total_shots
        self.min_shots = max(min_shots, total_shots // (10 * n_fragments))
        self._allocation: List[int] = self._uniform()
        self._variance_history: List[List[float]] = [[] for _ in range(n_fragments)]
        self._converged: List[bool] = [False] * n_fragments

    def _uniform(self) -> List[int]:
        """Distribute shots uniformly across fragments."""
        base = self.total_shots // self.n_fragments
        allocation = [base] * self.n_fragments
        # Distribute remainder to first fragments
        remainder = self.total_shots - base * self.n_fragments
        for i in range(remainder):
            allocation[i] += 1
        return allocation

    def initial_allocation(self) -> List[int]:
        """Return the initial uniform shot allocation."""
        return list(self._allocation)

    def update(
        self,
        variances: List[float],
        convergence_rates: Optional["List[float]"] = None,
    ) -> List[int]:
        """Reallocate shots based on observed per-fragment statistics.

        Args:
            variances: Per-fragment energy variance from last pass.
                       Higher variance → more shots needed.
            convergence_rates: Per-fragment convergence rate in [0,1].
                       Lower rate → more shots needed.
                       If None, only variance is used.

        Returns:
            Updated shot allocation list.
        """
        weights = np.array(variances, dtype=float)
        if weights.sum() < 1e-9:
            weights = np.ones(self.n_fragments, dtype=float)

        # Incorporate convergence rate (lower rate → higher weight)
        if convergence_rates is not None:
            conv = np.array(convergence_rates, dtype=float)
            difficulty = 1.0 - np.clip(conv, 0, 1)
            weights = weights * (1 + difficulty)

        # Mark well-converged fragments as needing minimum shots
        for i, v in enumerate(variances):
            if v < 0.01:
                self._converged[i] = True

        # Force minimum for converged fragments
        weights[self._converged] = 0.0

        # Recompute allocation
        if weights.sum() > 1e-9:
            normalised = weights / weights.sum()
        else:
            normalised = np.ones(self.n_fragments) / self.n_fragments

        # Reserve minimum shots for converged fragments
        n_converged = sum(self._converged)
        reserved = n_converged * self.min_shots
        available = max(0, self.total_shots - reserved)

        new_allocation = [self.min_shots] * self.n_fragments
        for i in range(self.n_fragments):
            if not self._converged[i]:
                new_allocation[i] = int(available * normalised[i]) + self.min_shots

        # Adjust for rounding
        diff = self.total_shots - sum(new_allocation)
        if diff != 0:
            # Give/take from the highest-weight fragment
            max_i = int(np.argmax(weights))
            new_allocation[max_i] += diff

        self._allocation = new_allocation
        return list(self._allocation)

    def current_allocation(self) -> List[int]:
        """Return the current shot allocation."""
        return list(self._allocation)

    def summary(self) -> str:
        """Human-readable allocation table."""
        lines = [
            f"AdaptiveShotAllocator: {self.n_fragments} fragments, "
            f"budget={self.total_shots}",
        ]
        for i, shots in enumerate(self._allocation):
            conv = "✓" if self._converged[i] else "·"
            lines.append(f"  {conv} F{i}: {shots} shots")
        lines.append(f"  Total: {sum(self._allocation)}")
        return "\n".join(lines)


# Allow Optional in body without importing at top (Python 3.9 compat)
from typing import Optional
