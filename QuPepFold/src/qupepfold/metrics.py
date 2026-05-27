"""Structured run metrics and timing utilities for QuPepFold.

Captures wall-clock time per stage, circuit depth, shot counts,
convergence traces, and other reproducibility metadata.
"""

import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalar and array types to native Python."""

    def default(self, obj):
        # Numpy integer types
        if isinstance(obj, (np.integer,)):
            return int(obj)
        # Numpy floating types (covers float16, float32, float64, …)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        # Numpy bool
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        # Numpy arrays → list (then elements are handled recursively)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types inside dicts/lists to native Python types."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@dataclass
class StageTimer:
    """Timing info for a single pipeline stage."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        if self.end_time <= 0:
            return 0.0
        return self.end_time - self.start_time


class Timer:
    """Context manager for wall-clock measurement of pipeline stages.

    Usage:
        with Timer("Stage 1: SA") as t:
            ... do work ...
        print(t.elapsed)  # seconds
    """
    def __init__(self, name: str = ""):
        self.name = name
        self._start = 0.0
        self._end = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._end = time.perf_counter()
        return False

    @property
    def elapsed(self) -> float:
        if self._end <= 0:
            return time.perf_counter() - self._start
        return self._end - self._start

    def to_stage_timer(self) -> StageTimer:
        return StageTimer(name=self.name, start_time=self._start, end_time=self._end)


@dataclass
class RunMetrics:
    """Structured metrics captured during a folding run.

    Attributes:
        wall_clock_seconds: Total pipeline wall-clock time.
        stage_timings: Per-stage wall-clock times.
        convergence_trace_sa: SA energy trace across steps.
        convergence_trace_vqe: Per-fragment VQE cost traces.
        circuit_depth: Ansatz circuit depth (transpiled).
        cnot_count: Number of CNOT gates in ansatz.
        total_shots: Total quantum shots across all fragments.
        total_circuit_evaluations: Total circuit evaluations (SPSA evals).
        n_fragments: Number of fragments processed.
        n_qubits_per_fragment: Qubit count per fragment.
        fidelity_estimate: Estimated fidelity of the solution (if computed).
        seed: Random seed used.
        backend_name: Backend identifier.
        optimizer_name: Optimizer used.
        encoding_name: Encoding scheme used.
        cache_hits: Number of fragment cache hits.
        cache_misses: Number of fragment cache misses.
    """
    wall_clock_seconds: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)
    convergence_trace_sa: List[float] = field(default_factory=list)
    convergence_trace_vqe: Dict[int, List[float]] = field(default_factory=dict)
    circuit_depth: int = 0
    cnot_count: int = 0
    total_shots: int = 0
    total_circuit_evaluations: int = 0
    n_fragments: int = 0
    n_qubits_per_fragment: int = 0
    fidelity_estimate: Optional[float] = None
    seed: int = 42
    backend_name: str = "aer"
    optimizer_name: str = "spsa"
    encoding_name: str = "turn"
    cache_hits: int = 0
    cache_misses: int = 0

    def add_stage_timing(self, name: str, elapsed: float) -> None:
        """Record timing for a pipeline stage."""
        self.stage_timings[name] = elapsed

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        All numpy scalars (float32, int32, …) are converted to native Python
        types so the result can be safely passed to :mod:`json`.
        """
        return _convert_numpy(asdict(self))

    def to_json(self, path: str) -> None:
        """Write metrics to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, cls=_NumpyEncoder)

    @classmethod
    def from_json(cls, path: str) -> "RunMetrics":
        """Load metrics from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=== Run Metrics ===",
            f"  Wall-clock:       {self.wall_clock_seconds:.2f}s",
            f"  Backend:          {self.backend_name}",
            f"  Optimizer:        {self.optimizer_name}",
            f"  Encoding:         {self.encoding_name}",
            f"  Seed:             {self.seed}",
            f"  Fragments:        {self.n_fragments}",
            f"  Qubits/fragment:  {self.n_qubits_per_fragment}",
            f"  Total shots:      {self.total_shots}",
            f"  Circuit depth:    {self.circuit_depth}",
            f"  CNOT count:       {self.cnot_count}",
        ]
        if self.fidelity_estimate is not None:
            lines.append(f"  Fidelity est:     {self.fidelity_estimate:.4f}")
        if self.stage_timings:
            lines.append("  Stage timings:")
            for name, t in self.stage_timings.items():
                lines.append(f"    {name}: {t:.2f}s")
        if self.cache_hits or self.cache_misses:
            lines.append(f"  Cache hits/misses: {self.cache_hits}/{self.cache_misses}")
        return "\n".join(lines)
