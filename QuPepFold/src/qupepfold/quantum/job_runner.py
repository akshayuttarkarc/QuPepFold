"""Fragment job runner: manages per-fragment VQE refinement with timing and restart support.

Provides FragmentJobRunner which wraps refine_all_fragments() with:
  - Per-fragment wall-clock tracking
  - Per-fragment seed management (reproducible)
  - Timeout enforcement
  - Job handle records for inspection
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np

from ..types import FoldConfig, FragmentSpec, FragmentEnergyTable, FragmentCandidate


@dataclass
class JobHandle:
    """Metadata record for a single fragment refinement job.

    Attributes:
        fragment_idx: Index of the fragment in the pipeline.
        fragment_seq: Amino acid sequence of the fragment.
        status: "pending", "running", "done", "failed", "timeout".
        elapsed_seconds: Wall-clock time for this job.
        n_candidates: Number of candidates returned.
        best_energy: Best candidate energy.
        error: Error message if status == "failed".
    """
    fragment_idx: int
    fragment_seq: str
    status: str = "pending"
    elapsed_seconds: float = 0.0
    n_candidates: int = 0
    best_energy: float = float("inf")
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class FragmentJobRunner:
    """Manages sequential VQE refinement jobs for all fragments.

    Wraps refine_fragment() with per-job timing, seed isolation,
    timeout enforcement, and result collection.

    Args:
        config: FoldConfig with optimizer, shots, spsa_iterations etc.
        backend: SamplerBackend to use (Aer or Runtime).
        verbose: Print per-job progress.
    """

    def __init__(self, config: FoldConfig, backend, verbose: bool = False):
        self.config = config
        self.backend = backend
        self.verbose = verbose
        self.handles: List[JobHandle] = []

    def run_all(
        self,
        fragments: List[FragmentSpec],
        energy_tables: List[FragmentEnergyTable],
        mj_matrix: Optional[np.ndarray] = None,
        global_bits: Optional[str] = None,
        global_positions: Optional[np.ndarray] = None,
    ) -> List[List[FragmentCandidate]]:
        """Run VQE refinement for all fragments, returning candidates.

        Args:
            fragments: List of FragmentSpec objects.
            energy_tables: Precomputed energy tables, one per fragment.
            mj_matrix: MJ interaction matrix for context energy.
            global_bits: SA warm-start bitstring.
            global_positions: 3D position array from SA result.

        Returns:
            List of candidate lists, one per fragment.
        """
        from .refine_fragment import refine_fragment
        from ..model.encoding import decode_turns
        from ..model.lattice import trace_positions

        all_candidates: List[List[FragmentCandidate]] = []
        self.handles.clear()

        max_wall_clock = getattr(self.config, "max_wall_clock_seconds", None)
        pipeline_start = time.perf_counter()

        for i, (frag, table) in enumerate(zip(fragments, energy_tables)):
            handle = JobHandle(
                fragment_idx=i,
                fragment_seq=frag.sequence,
                status="running",
            )

            # Check overall timeout
            if max_wall_clock is not None:
                elapsed_total = time.perf_counter() - pipeline_start
                if elapsed_total >= max_wall_clock:
                    if self.verbose:
                        print(f"  ⏱ Timeout reached at fragment {i}, using table-best fallback.")
                    handle.status = "timeout"
                    # Fallback: use table best candidates
                    candidates = table.get_best_candidates(
                        top_k=self.config.top_k_candidates
                    )
                    handle.n_candidates = len(candidates)
                    handle.best_energy = candidates[0].energy if candidates else float("inf")
                    all_candidates.append(candidates)
                    self.handles.append(handle)
                    continue

            if self.verbose:
                print(f"  Fragment {i+1}/{len(fragments)}: {frag.sequence}", flush=True)

            # Extract warm-start bits for this fragment
            warmstart_bits = None
            if global_bits:
                n_turn_frag = frag.end_res - frag.start_res - 1
                bit_start = frag.start_res * 2
                bit_end = bit_start + n_turn_frag * 2
                if bit_end <= len(global_bits):
                    warmstart_bits = global_bits[bit_start:bit_end]

            # Per-fragment seed (offset by index for diversity)
            frag_config = type(self.config)(**{
                **vars(self.config),
                "seed": self.config.seed + i * 31,
            })

            t_start = time.perf_counter()
            try:
                candidates = refine_fragment(
                    fragment=frag,
                    energy_table=table,
                    backend=self.backend,
                    config=frag_config,
                    mj_matrix=mj_matrix,
                    warmstart_bits=warmstart_bits,
                    global_positions=global_positions,
                    verbose=self.verbose,
                    fragment_idx=i,
                )
                handle.status = "done"
                handle.n_candidates = len(candidates)
                handle.best_energy = candidates[0].energy if candidates else float("inf")

            except Exception as exc:
                if self.verbose:
                    print(f"  ⚠ Fragment {i} failed: {exc}")
                handle.status = "failed"
                handle.error = str(exc)
                # Fallback to table best
                candidates = table.get_best_candidates(
                    top_k=self.config.top_k_candidates
                )

            handle.elapsed_seconds = time.perf_counter() - t_start

            if self.verbose:
                best_e = handle.best_energy
                print(f"    → {handle.n_candidates} candidates, best E={best_e:.2f}, "
                      f"took {handle.elapsed_seconds:.1f}s")

            all_candidates.append(candidates)
            self.handles.append(handle)

        return all_candidates

    def summary(self) -> str:
        """Human-readable summary of all job handles."""
        lines = ["=== Fragment Job Summary ==="]
        total = sum(h.elapsed_seconds for h in self.handles)
        for h in self.handles:
            icon = {"done": "✓", "failed": "✗", "timeout": "⏱", "pending": "?"}.get(h.status, "?")
            lines.append(
                f"  {icon} F{h.fragment_idx} ({h.fragment_seq[:6]}…) "
                f"status={h.status} candidates={h.n_candidates} "
                f"bestE={h.best_energy:.2f} t={h.elapsed_seconds:.1f}s"
            )
        lines.append(f"  Total: {len(self.handles)} fragments, {total:.1f}s")
        return "\n".join(lines)
