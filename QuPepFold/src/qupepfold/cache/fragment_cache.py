"""Fragment cache: persistent store for VQE refinement results.

Uses a JSON-directory backend (one file per fragment key) for portability.
Supports exact lookup and approximate matching via Hamming distance.

Usage::

    cache = FragmentCache("/path/to/cache_dir")
    key = make_cache_key(fragment.sequence, config)

    result = cache.get(key)
    if result is None:
        # Run VQE ...
        cache.put(key, candidates)
    else:
        candidates = result.candidates
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..types import FragmentCandidate
from .cache_key import FragmentCacheKey, approximate_match


@dataclass
class CachedFragmentResult:
    """Stored result for a fragment refinement run.

    Attributes:
        key: The cache key identifying this result.
        candidates: List of FragmentCandidate objects.
        best_energy: Best energy among candidates.
        n_evaluations: Total circuit evaluations used.
        elapsed_seconds: Wall-clock time of the original run.
        metadata: Arbitrary extra info.
    """
    key: FragmentCacheKey
    candidates: List[FragmentCandidate]
    best_energy: float = float("inf")
    n_evaluations: int = 0
    elapsed_seconds: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "key": self.key.to_dict(),
            "best_energy": self.best_energy,
            "n_evaluations": self.n_evaluations,
            "elapsed_seconds": self.elapsed_seconds,
            "metadata": self.metadata,
            "candidates": [
                {
                    "bits": c.bits,
                    "energy": float(c.energy),
                    "probability": float(c.probability),
                    "meta": c.meta,
                }
                for c in self.candidates
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CachedFragmentResult":
        key = FragmentCacheKey.from_dict(d["key"])
        candidates = []
        for cd in d.get("candidates", []):
            candidates.append(FragmentCandidate(
                bits=cd["bits"],
                energy=cd["energy"],
                probability=cd["probability"],
                meta=cd.get("meta", {}),
            ))
        return cls(
            key=key,
            candidates=candidates,
            best_energy=d.get("best_energy", float("inf")),
            n_evaluations=d.get("n_evaluations", 0),
            elapsed_seconds=d.get("elapsed_seconds", 0.0),
            metadata=d.get("metadata", {}),
        )


class FragmentCache:
    """JSON-directory fragment result cache.

    Each entry is stored as a JSON file named after the key hash.
    Supports exact lookup and approximate matching.

    Args:
        cache_dir: Path to the cache directory. Created if absent.
        approx_tolerance: Hamming distance threshold for approximate match (default: 2).
    """

    def __init__(self, cache_dir: str, approx_tolerance: int = 2):
        self.cache_dir = cache_dir
        self.approx_tolerance = approx_tolerance
        os.makedirs(cache_dir, exist_ok=True)
        self._index: Dict[str, FragmentCacheKey] = {}
        self._load_index()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _key_to_filename(self, key: FragmentCacheKey) -> str:
        """Derive a safe filename from a cache key."""
        safe_seq = re.sub(r"[^A-Z]", "", key.sequence)[:20]
        return f"{safe_seq}_{key.encoding}_{key.backend}_{key.config_hash[:12]}.json"

    def _filepath(self, key: FragmentCacheKey) -> str:
        return os.path.join(self.cache_dir, self._key_to_filename(key))

    def _load_index(self) -> None:
        """Scan cache_dir and build an in-memory index of keys."""
        self._index.clear()
        for fname in os.listdir(self.cache_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self.cache_dir, fname)
            try:
                with open(fpath) as f:
                    d = json.load(f)
                key = FragmentCacheKey.from_dict(d["key"])
                self._index[fname] = key
            except Exception:
                pass  # Corrupt entry — skip

    # ── Public API ──────────────────────────────────────────────────────────

    def get(self, key: FragmentCacheKey) -> Optional[CachedFragmentResult]:
        """Exact cache lookup.

        Args:
            key: Cache key to look up.

        Returns:
            CachedFragmentResult if found, else None.
        """
        fpath = self._filepath(key)
        if not os.path.exists(fpath):
            return None
        try:
            with open(fpath) as f:
                d = json.load(f)
            return CachedFragmentResult.from_dict(d)
        except Exception:
            return None

    def approximate_get(
        self, key: FragmentCacheKey
    ) -> Optional[CachedFragmentResult]:
        """Approximate cache lookup using Hamming distance on sequences.

        Finds the closest cached key within approx_tolerance.

        Args:
            key: Query cache key.

        Returns:
            Best approximate match, or None.
        """
        best_result = None
        best_dist = self.approx_tolerance + 1

        for fname, stored_key in self._index.items():
            if not approximate_match(key, stored_key, self.approx_tolerance):
                continue
            from .cache_key import hamming_distance
            dist = hamming_distance(key.sequence, stored_key.sequence)
            if dist < best_dist:
                fpath = os.path.join(self.cache_dir, fname)
                try:
                    with open(fpath) as f:
                        d = json.load(f)
                    best_result = CachedFragmentResult.from_dict(d)
                    best_dist = dist
                except Exception:
                    pass

        return best_result

    def put(
        self,
        key: FragmentCacheKey,
        candidates: List[FragmentCandidate],
        elapsed_seconds: float = 0.0,
        n_evaluations: int = 0,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Store a fragment result in the cache.

        Args:
            key: Cache key.
            candidates: Ordered list of FragmentCandidate objects (best first).
            elapsed_seconds: Wall-clock time of the VQE run.
            n_evaluations: Total circuit evaluations.
            metadata: Optional extra info.
        """
        result = CachedFragmentResult(
            key=key,
            candidates=candidates,
            best_energy=candidates[0].energy if candidates else float("inf"),
            n_evaluations=n_evaluations,
            elapsed_seconds=elapsed_seconds,
            metadata=metadata or {},
        )
        fpath = self._filepath(key)
        fname = os.path.basename(fpath)
        with open(fpath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        self._index[fname] = key

    def invalidate(self, key: FragmentCacheKey) -> bool:
        """Remove a cache entry.

        Returns:
            True if the entry existed and was removed.
        """
        fpath = self._filepath(key)
        fname = os.path.basename(fpath)
        if os.path.exists(fpath):
            os.remove(fpath)
            self._index.pop(fname, None)
            return True
        return False

    def clear(self) -> int:
        """Remove all cache entries. Returns number of entries removed."""
        count = 0
        for fname in list(self._index.keys()):
            fpath = os.path.join(self.cache_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
                count += 1
        self._index.clear()
        return count

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "n_entries": len(self._index),
            "cache_dir": self.cache_dir,
            "approx_tolerance": self.approx_tolerance,
        }

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return f"FragmentCache({self.cache_dir!r}, {len(self)} entries)"
