"""Fragment manifest: serialisable record of how a protein was fragmented.

A manifest captures the protein identity, sequence, strategy used,
and the resulting fragment layout — useful for reproducibility, caching,
and multi-run comparisons.
"""

import json
import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

from ..types import FragmentSpec


@dataclass
class FragmentManifest:
    """Persistent record of a fragmentation run.

    Attributes:
        protein_id: Identifier for the parent protein (UniProt, custom, etc.).
        sequence: Full amino acid sequence.
        total_length: len(sequence).
        strategy_used: Name of the FragmentStrategy that generated this.
        fragments: Ordered list of FragmentSpec objects.
        overlap_map: Mapping of fragment index → set of overlapping fragment indices.
        creation_timestamp: ISO 8601 timestamp.
        metadata: Arbitrary extra info (config snapshot, version, etc.).
    """
    protein_id: str
    sequence: str
    total_length: int
    strategy_used: str
    fragments: List[FragmentSpec] = field(default_factory=list)
    overlap_map: Dict[int, List[int]] = field(default_factory=dict)
    creation_timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)

    # ── Serialisation ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convert to a JSON-compatible dict."""
        d = {
            "protein_id": self.protein_id,
            "sequence": self.sequence,
            "total_length": self.total_length,
            "strategy_used": self.strategy_used,
            "creation_timestamp": self.creation_timestamp,
            "metadata": self.metadata,
            "overlap_map": {str(k): v for k, v in self.overlap_map.items()},
            "fragments": [],
        }
        for f in self.fragments:
            d["fragments"].append({
                "start_res": f.start_res,
                "end_res": f.end_res,
                "sequence": f.sequence,
                "overlap_left_turns": f.overlap_left_turns,
                "overlap_right_turns": f.overlap_right_turns,
                "priority": f.priority,
                "parent_protein_id": f.parent_protein_id,
                "preprocessing_metadata": f.preprocessing_metadata,
            })
        return d

    def to_json(self, path: str) -> None:
        """Write manifest to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "FragmentManifest":
        """Load manifest from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        frags = []
        for fd in data.get("fragments", []):
            frags.append(FragmentSpec(
                start_res=fd["start_res"],
                end_res=fd["end_res"],
                sequence=fd["sequence"],
                overlap_left_turns=fd.get("overlap_left_turns", 0),
                overlap_right_turns=fd.get("overlap_right_turns", 0),
                priority=fd.get("priority", 0.0),
                parent_protein_id=fd.get("parent_protein_id"),
                preprocessing_metadata=fd.get("preprocessing_metadata", {}),
            ))
        overlap_map = {int(k): v for k, v in data.get("overlap_map", {}).items()}
        return cls(
            protein_id=data["protein_id"],
            sequence=data["sequence"],
            total_length=data["total_length"],
            strategy_used=data["strategy_used"],
            fragments=frags,
            overlap_map=overlap_map,
            creation_timestamp=data.get("creation_timestamp", ""),
            metadata=data.get("metadata", {}),
        )

    # ── Display ─────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary with ASCII fragment layout."""
        lines = [
            f"=== Fragment Manifest ===",
            f"  Protein:    {self.protein_id}",
            f"  Length:     {self.total_length} AA",
            f"  Strategy:  {self.strategy_used}",
            f"  Fragments: {len(self.fragments)}",
            f"  Created:   {self.creation_timestamp}",
            "",
        ]

        # ASCII layout: show each fragment as a bar on the sequence axis
        if self.fragments:
            width = min(80, self.total_length)
            scale = width / self.total_length if self.total_length > 0 else 1.0

            lines.append(f"  Layout (each row = one fragment):")
            lines.append(f"  {'0':<{width//2}}{'':>{width//2}}{self.total_length}")
            lines.append(f"  |{'─' * width}|")

            for i, f in enumerate(self.fragments):
                left = int(f.start_res * scale)
                right = int(f.end_res * scale)
                bar_len = max(1, right - left)
                label = f"F{i}({f.sequence[:4]}…)" if len(f.sequence) > 4 else f"F{i}({f.sequence})"
                bar = " " * left + "█" * bar_len
                pri = f" P={f.priority:.2f}" if f.priority else ""
                lines.append(f"  |{bar:<{width}}| {label}{pri}")

            lines.append(f"  |{'─' * width}|")

        # Fragment details
        lines.append("")
        lines.append("  Details:")
        for i, f in enumerate(self.fragments):
            lines.append(
                f"    F{i}: res {f.start_res}-{f.end_res} "
                f"({f.sequence}) "
                f"overlap L={f.overlap_left_turns} R={f.overlap_right_turns}"
            )

        return "\n".join(lines)


def build_manifest(
    sequence: str,
    fragments: List[FragmentSpec],
    strategy_used: str,
    protein_id: str = "unknown",
    metadata: Optional[Dict] = None,
) -> FragmentManifest:
    """Build a FragmentManifest from a list of fragments.

    Automatically computes the overlap_map from fragment ranges.
    """
    # Compute overlap map
    overlap_map: Dict[int, List[int]] = {}
    for i, fi in enumerate(fragments):
        overlaps = []
        for j, fj in enumerate(fragments):
            if i == j:
                continue
            # Two fragments overlap if their residue ranges intersect
            if fi.start_res < fj.end_res and fj.start_res < fi.end_res:
                overlaps.append(j)
        overlap_map[i] = overlaps

    return FragmentManifest(
        protein_id=protein_id,
        sequence=sequence,
        total_length=len(sequence),
        strategy_used=strategy_used,
        fragments=fragments,
        overlap_map=overlap_map,
        metadata=metadata or {},
    )
