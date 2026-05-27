"""CSV export utilities for QuPepFold results."""

import os
import csv
from typing import Dict, List, Optional

from ..types import FoldResult, FragmentCandidate


def write_bitstring_summary(
    states: List[str],
    probabilities: List[float],
    energies: List[float],
    output_path: str,
    export_threshold: float = 0.02,
) -> None:
    """Write comprehensive bitstring summary CSV.
    
    Args:
        states: List of bitstrings.
        probabilities: Corresponding probabilities.
        energies: Corresponding energies.
        output_path: Path to CSV file.
        export_threshold: Probability threshold for marking as exported.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bitstring", "probability", "energy", "exported_PDB"])
        
        for s, p, e in zip(states, probabilities, energies):
            exported = 1 if p >= export_threshold else 0
            writer.writerow([s, f"{p:.6f}", f"{e:.4f}", exported])


def write_fragment_candidates_csv(
    fragment_candidates: List[List[FragmentCandidate]],
    output_path: str,
) -> None:
    """Write fragment candidates to CSV.
    
    Args:
        fragment_candidates: List of candidate lists per fragment.
        output_path: Path to CSV file.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fragment_idx", "rank", "bitstring", "energy", "probability"])
        
        for frag_idx, candidates in enumerate(fragment_candidates):
            for rank, c in enumerate(candidates):
                writer.writerow([
                    frag_idx,
                    rank,
                    c.bits,
                    f"{c.energy:.4f}",
                    f"{c.probability:.6f}",
                ])


def write_energy_breakdown_csv(
    bitstring: str,
    probability: float,
    components: Dict[str, float],
    output_path: str,
) -> None:
    """Write energy breakdown for a single bitstring.
    
    Args:
        bitstring: The bitstring.
        probability: Sampling probability.
        components: Dict of energy components.
        output_path: Path to CSV file.
    """
    total = sum(components.values())
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header row
        header = ["bitstring", "probability", "total"] + list(components.keys())
        writer.writerow(header)
        
        # Data row
        row = [bitstring, f"{probability:.6f}", f"{total:.4f}"]
        row += [f"{v:.4f}" for v in components.values()]
        writer.writerow(row)


def write_output_summary(
    result: FoldResult,
    output_path: str,
    config_dict: Optional[Dict] = None,
) -> None:
    """Write text summary of folding results.
    
    Args:
        result: FoldResult object.
        output_path: Path to text file.
        config_dict: Optional configuration dict for additional info.
    """
    lines = [
        "--- QuPepFold Folding Summary ---",
        "",
        f"Protein Sequence: {result.sequence}",
        f"Sequence Length:  {len(result.sequence)}",
        "",
        "=== Global Search ===",
        f"Best Bitstring: {result.global_bits}",
        f"Best Energy:    {result.global_energy:.4f}",
        "",
        "=== Stitched Result ===",
        f"Stitched Bitstring: {result.stitched_bits}",
        f"Stitched Energy:    {result.stitched_energy:.4f}",
        "",
        "=== Output Files ===",
        f"PDB:         {result.pdb_path or 'N/A'}",
        f"Relaxed PDB: {result.relaxed_pdb_path or 'N/A'}",
    ]
    
    if config_dict:
        lines.append("")
        lines.append("=== Configuration ===")
        for key, val in config_dict.items():
            lines.append(f"{key}: {val}")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_all_csvs(
    result: FoldResult,
    output_dir: str,
    prob_dict: Optional[Dict[str, float]] = None,
    energies: Optional[Dict[str, float]] = None,
    energy_components: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Generate all standard CSV files.
    
    Args:
        result: FoldResult object.
        output_dir: Directory to save CSVs.
        prob_dict: Optional probability distribution.
        energies: Optional energy mapping.
        energy_components: Optional energy breakdown.
        
    Returns:
        List of generated CSV paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = []
    
    # Fragment candidates
    if result.fragment_candidates:
        frag_csv = os.path.join(output_dir, "fragment_candidates.csv")
        write_fragment_candidates_csv(result.fragment_candidates, frag_csv)
        generated.append(frag_csv)
    
    # Bitstring summary
    if prob_dict and energies:
        states = list(prob_dict.keys())
        probs = [prob_dict[s] for s in states]
        ens = [energies.get(s, 0.0) for s in states]
        
        summary_csv = os.path.join(output_dir, "bitstring_summary.csv")
        write_bitstring_summary(states, probs, ens, summary_csv)
        generated.append(summary_csv)
    
    # Energy breakdown
    if energy_components:
        breakdown_csv = os.path.join(output_dir, "energy_breakdown.csv")
        # Use stitched result
        write_energy_breakdown_csv(
            result.stitched_bits,
            1.0,  # Probability not applicable for stitched
            energy_components,
            breakdown_csv,
        )
        generated.append(breakdown_csv)
    
    # Text summary
    summary_txt = os.path.join(output_dir, "output_summary.txt")
    write_output_summary(result, summary_txt)
    generated.append(summary_txt)
    
    return generated
