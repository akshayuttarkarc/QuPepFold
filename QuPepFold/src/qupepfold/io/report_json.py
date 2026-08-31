"""JSON report generation for fold results."""

import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..types import FoldResult, FoldConfig
from .. import __version__


def build_report(
    result: FoldResult,
    config: FoldConfig,
    provenance: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable report dict.
    
    Args:
        result: FoldResult from pipeline.
        config: FoldConfig used.
        provenance: Optional provenance dict.
        
    Returns:
        Report dictionary.
    """
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "package_version": __version__,
        },
        "input": {
            "sequence": result.sequence,
            "sequence_length": len(result.sequence),
        },
        "config": {
            "fragment_length": config.fragment_length,
            "overlap_turns": config.overlap_turns,
            "shots": config.shots,
            "spsa_iterations": config.spsa_iterations,
            "ansatz_depth": config.ansatz_depth,
            "backend": config.backend,
            "ibm_backend": config.ibm_backend,
            "seed": config.seed,
        },
        "global_search": {
            "bitstring": result.global_bits,
            "energy": result.global_energy,
        },
        "fragments": [],
        "stitching": {
            "bitstring": result.stitched_bits,
            "energy": result.stitched_energy,
        },
        "output": {
            "pdb_path": result.pdb_path,
            "relaxed_pdb_path": result.relaxed_pdb_path,
        },
    }
    
    # Add fragment details
    for i, candidates in enumerate(result.fragment_candidates):
        frag_info = {
            "fragment_index": i,
            "num_candidates": len(candidates),
            "candidates": [
                {
                    "bits": c.bits,
                    "energy": c.energy,
                    "probability": c.probability,
                }
                for c in candidates[:5]  # Top 5 for report
            ],
        }
        report["fragments"].append(frag_info)
    
    # Add provenance
    if provenance:
        report["provenance"] = provenance
    elif result.provenance:
        report["provenance"] = result.provenance
    
    return report


def write_report(
    result: FoldResult,
    config: FoldConfig,
    output_path: str,
    provenance: Optional[Dict] = None,
) -> None:
    """Write JSON report to file.
    
    Args:
        result: FoldResult from pipeline.
        config: FoldConfig used.
        output_path: Path to output JSON file.
        provenance: Optional provenance dict.
    """
    report = build_report(result, config, provenance)
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def write_summary_csv(
    result: FoldResult,
    output_path: str,
) -> None:
    """Write CSV summary of candidates.
    
    Args:
        result: FoldResult.
        output_path: Path to CSV file.
    """
    import csv
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fragment", "rank", "bitstring", "energy", "probability"])
        
        for frag_idx, candidates in enumerate(result.fragment_candidates):
            for rank, c in enumerate(candidates):
                writer.writerow([frag_idx, rank, c.bits, c.energy, c.probability])
