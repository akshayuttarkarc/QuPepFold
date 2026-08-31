"""Fragment cache key: uniquely identifies a cacheable fragment computation.

Key components:
  - sequence_slice: The amino acid sequence of the fragment
  - encoding_type: Encoding scheme name ("turn", "hp", etc.)
  - penalty_profile_hash: Hash of penalty weights (overlap, backbone, etc.)
  - backend_name: Quantum backend identifier
  - config_hash: Hash of cache-relevant config fields

Approximate matching uses Hamming distance on sequence slices.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Optional


# Fields in FoldConfig that affect the energy table (and thus caching)
_CACHE_RELEVANT_FIELDS = (
    "fragment_length",
    "overlap_turns",
    "encoding",
    "overlap_penalty",
    "contact_min_sep",
    "contact_cutoff",
    "lam_back",
    "lam_dis",
    "lam_loc",
    "cvar_alpha",
    "shots",
    "spsa_iterations",
    "ansatz_depth",
    "optimizer",
)


@dataclass(frozen=True)
class FragmentCacheKey:
    """Immutable cache key for a fragment refinement result.

    Attributes:
        sequence: Amino acid sequence of the fragment.
        encoding: Encoding scheme name.
        config_hash: SHA-256 of cache-relevant config fields.
        backend: Backend name.
    """
    sequence: str
    encoding: str
    config_hash: str
    backend: str

    def __str__(self) -> str:
        return f"{self.sequence}|{self.encoding}|{self.backend}|{self.config_hash[:8]}"

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "encoding": self.encoding,
            "config_hash": self.config_hash,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FragmentCacheKey":
        return cls(
            sequence=d["sequence"],
            encoding=d["encoding"],
            config_hash=d["config_hash"],
            backend=d["backend"],
        )


def make_cache_key(fragment_sequence: str, config, backend_name: str = "aer") -> FragmentCacheKey:
    """Build a FragmentCacheKey from a sequence and config.

    Args:
        fragment_sequence: AA sequence of the fragment.
        config: FoldConfig instance.
        backend_name: Name of the backend being used.

    Returns:
        FragmentCacheKey.
    """
    cfg_dict = {}
    for field in _CACHE_RELEVANT_FIELDS:
        val = getattr(config, field, None)
        if val is not None:
            cfg_dict[field] = val

    cfg_str = json.dumps(cfg_dict, sort_keys=True)
    cfg_hash = hashlib.sha256(cfg_str.encode()).hexdigest()
    encoding = getattr(config, "encoding", "turn")

    return FragmentCacheKey(
        sequence=fragment_sequence,
        encoding=encoding,
        config_hash=cfg_hash,
        backend=backend_name,
    )


def hamming_distance(seq_a: str, seq_b: str) -> int:
    """Compute Hamming distance between two equal-length sequences.

    For different-length sequences, returns max(len(a), len(b)) (no match).
    """
    if len(seq_a) != len(seq_b):
        return max(len(seq_a), len(seq_b))
    return sum(a != b for a, b in zip(seq_a, seq_b))


def approximate_match(
    key: FragmentCacheKey,
    candidate_key: FragmentCacheKey,
    tolerance: int = 2,
) -> bool:
    """Check if two keys are approximately equal (same backend+encoding, near sequence).

    Args:
        key: Query key.
        candidate_key: Stored key to compare against.
        tolerance: Maximum Hamming distance on sequence (default: 2).

    Returns:
        True if the keys are approximately equal.
    """
    if key.encoding != candidate_key.encoding:
        return False
    if key.backend != candidate_key.backend:
        return False
    # Config must match exactly for approximate match to be safe
    if key.config_hash != candidate_key.config_hash:
        return False
    return hamming_distance(key.sequence, candidate_key.sequence) <= tolerance
