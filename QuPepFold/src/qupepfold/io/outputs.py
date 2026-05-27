"""Comprehensive output generation for QuPepFold results.

Generates all output files matching original qupepfold 0.8.0:
- output_summary.txt - Text summary of run
- bitstring_summary.csv - All bitstrings with energies
- fragment_candidates.csv - All fragment candidates
- most_negative_energy_breakdown.csv - Energy components for best state
- optimal_circuit.png - Quantum circuit diagram
- bitstring_histogram.png - Probability histogram
- cvar_scatter.png - CVaR across iterations
- energy_breakdown.png - Energy component bar chart
- fragment_energies.png - Per-fragment energy distribution
- pdb3d.zip - ZIP of all PDB files
"""

import os
import csv
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

from ..types import FoldResult, FoldConfig, FragmentSpec, FragmentCandidate


def save_output_summary(
    result: FoldResult,
    config: FoldConfig,
    output_dir: str,
    sequence: str,
    verbose: bool = True,
) -> str:
    """Save comprehensive text summary of the folding run.
    
    Matches original `output_summary.txt` format.
    """
    summary_lines = [
        "=" * 60,
        "          QuPepFold - Quantum Protein Folding Summary",
        "=" * 60,
        "",
        f"Date/Time:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Protein Sequence:   {sequence}",
        f"Sequence Length:    {len(sequence)} amino acids",
        "",
        "--- Configuration ---",
        f"Backend:            {config.backend}",
        f"Shots:              {config.shots}",
        f"SPSA Iterations:    {config.spsa_iterations}",
        f"Ansatz Depth:       {config.ansatz_depth}",
        f"CVaR Alpha:         {config.cvar_alpha}",
        f"SA Steps:           {config.sa_steps}",
        f"SA Restarts:        {config.sa_restarts}",
        f"Fragment Length:    {config.fragment_length}",
        f"Overlap Turns:      {config.overlap_turns}",
        f"Random Seed:        {config.seed}",
        "",
        "--- Results ---",
        f"Global Search Energy:   {result.global_energy:.6f}",
        f"Stitched Energy:        {result.stitched_energy:.6f}",
        f"Final Energy:           {result.energy:.6f}",
        f"Final Bitstring:        {result.best_bits[:40]}...",
        "",
        "--- Fragment Summary ---",
    ]
    
    # Add fragment info
    if result.selected_candidates:
        for i, cand in enumerate(result.selected_candidates):
            summary_lines.append(
                f"  Fragment {i+1}: E={cand.energy:>8.2f}  bits={cand.bits[:20]}..."
            )
    
    summary_lines.extend([
        "",
        "--- Energy Components (Best State) ---",
    ])
    
    if result.energy_breakdown:
        for key, val in result.energy_breakdown.items():
            summary_lines.append(f"  {key.capitalize():15s}: {val:>10.4f}")
    
    summary_lines.extend([
        "",
        "=" * 60,
    ])
    
    summary = "\n".join(summary_lines)
    
    path = os.path.join(output_dir, "output_summary.txt")
    with open(path, "w") as f:
        f.write(summary + "\n")
    
    if verbose:
        print(f"  ✓ Saved summary: output_summary.txt")
    
    return path


def save_bitstring_summary_csv(
    candidates: List[Dict[str, Any]],
    output_dir: str,
    verbose: bool = True,
) -> str:
    """Save all bitstrings with energies to CSV.
    
    Format: bitstring, energy, probability, fragment_id, source, exported
    """
    path = os.path.join(output_dir, "bitstring_summary.csv")
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bitstring", "energy", "probability", "fragment_id", "source", "selected"
        ])
        for cand in candidates:
            writer.writerow([
                cand.get("bits", ""),
                cand.get("energy", 0.0),
                cand.get("probability", 0.0),
                cand.get("fragment_id", -1),
                cand.get("source", "unknown"),
                cand.get("selected", 0),
            ])
    
    if verbose:
        print(f"  ✓ Saved CSV: bitstring_summary.csv ({len(candidates)} rows)")
    
    return path


def save_fragment_candidates_csv(
    fragment_candidates: List[List[FragmentCandidate]],
    fragments: List[FragmentSpec],
    selected_indices: Optional[List[int]],
    output_dir: str,
    verbose: bool = True,
) -> str:
    """Save all fragment candidates to detailed CSV."""
    path = os.path.join(output_dir, "fragment_candidates.csv")
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fragment_id", "fragment_sequence", "start_res", "end_res", 
            "candidate_rank", "bitstring", "energy", "probability", "source", "selected"
        ])
        
        for frag_idx, (frag, cands) in enumerate(zip(fragments, fragment_candidates)):
            selected_bits = None
            if selected_indices and frag_idx < len(selected_indices):
                sel_idx = selected_indices[frag_idx]
                if sel_idx < len(cands):
                    selected_bits = cands[sel_idx].bits
            
            for rank, cand in enumerate(cands):
                is_selected = 1 if cand.bits == selected_bits else 0
                source = cand.meta.get("source", "unknown") if cand.meta else "unknown"
                
                writer.writerow([
                    frag_idx + 1,
                    frag.sequence,
                    frag.start_res,
                    frag.end_res,
                    rank + 1,
                    cand.bits,
                    f"{cand.energy:.4f}",
                    f"{cand.probability:.6f}",
                    source,
                    is_selected,
                ])
    
    total = sum(len(c) for c in fragment_candidates)
    if verbose:
        print(f"  ✓ Saved CSV: fragment_candidates.csv ({total} candidates)")
    
    return path


def save_energy_breakdown_csv(
    components: Dict[str, float],
    bitstring: str,
    output_dir: str,
    verbose: bool = True,
) -> str:
    """Save energy breakdown for best bitstring to CSV."""
    path = os.path.join(output_dir, "most_negative_energy_breakdown.csv")
    
    total = sum(components.values())
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bitstring", "total"] + list(components.keys()))
        writer.writerow([bitstring, f"{total:.4f}"] + [f"{v:.4f}" for v in components.values()])
    
    if verbose:
        print(f"  ✓ Saved CSV: most_negative_energy_breakdown.csv")
    
    return path


def save_cvar_trace_csv(
    cvar_trace: List[float],
    output_dir: str,
    verbose: bool = True,
) -> str:
    """Save CVaR trace across iterations to CSV."""
    path = os.path.join(output_dir, "cvar_trace.csv")
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "cvar_energy"])
        for i, val in enumerate(cvar_trace):
            writer.writerow([i + 1, f"{val:.6f}"])
    
    if verbose:
        print(f"  ✓ Saved CSV: cvar_trace.csv ({len(cvar_trace)} iterations)")
    
    return path


def save_sa_trace_csv(
    sa_energies: List[float],
    output_dir: str,
    verbose: bool = True,
) -> str:
    """Save simulated annealing trace to CSV."""
    path = os.path.join(output_dir, "sa_trace.csv")
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["restart", "best_energy"])
        for i, val in enumerate(sa_energies):
            writer.writerow([i + 1, f"{val:.6f}"])
    
    if verbose:
        print(f"  ✓ Saved CSV: sa_trace.csv ({len(sa_energies)} restarts)")
    
    return path


def save_spsa_trace_csv(
    spsa_traces: List[List[float]],
    output_dir: str,
    verbose: bool = True,
) -> str:
    """Save SPSA optimization traces per fragment to CSV."""
    path = os.path.join(output_dir, "spsa_traces.csv")
    
    # Find max length
    max_len = max(len(t) for t in spsa_traces) if spsa_traces else 0
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["iteration"] + [f"fragment_{i+1}" for i in range(len(spsa_traces))]
        writer.writerow(header)
        
        for i in range(max_len):
            row = [i + 1]
            for trace in spsa_traces:
                if i < len(trace):
                    row.append(f"{trace[i]:.4f}")
                else:
                    row.append("")
            writer.writerow(row)
    
    if verbose:
        print(f"  ✓ Saved CSV: spsa_traces.csv ({len(spsa_traces)} fragments)")
    
    return path


def create_pdb_zip(
    pdb_dir: str,
    output_dir: str,
    verbose: bool = True,
) -> Optional[str]:
    """Create a ZIP file of all PDB files."""
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')] if os.path.exists(pdb_dir) else []
    
    if not pdb_files:
        if verbose:
            print(f"  ⚠ No PDB files to zip")
        return None
    
    zip_path = os.path.join(output_dir, "pdb3d")
    shutil.make_archive(zip_path, 'zip', pdb_dir)
    
    if verbose:
        print(f"  ✓ Created ZIP: pdb3d.zip ({len(pdb_files)} PDB files)")
    
    return zip_path + ".zip"


def save_terminal_log(
    log_lines: List[str],
    output_dir: str,
    verbose: bool = True,
) -> str:
    """Save captured terminal output to text file."""
    path = os.path.join(output_dir, "terminal_output.txt")
    
    with open(path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    
    if verbose:
        print(f"  ✓ Saved terminal log: terminal_output.txt")
    
    return path


def save_mj_matrix_csv(
    mj_matrix: np.ndarray,
    sequence: str,
    output_dir: str,
    verbose: bool = True,
) -> str:
    """Save MJ interaction matrix to CSV."""
    path = os.path.join(output_dir, "mj_matrix.csv")
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header with residue indices
        header = [""] + [f"{i}:{sequence[i]}" for i in range(len(sequence))]
        writer.writerow(header)
        
        for i in range(len(sequence)):
            row = [f"{i}:{sequence[i]}"]
            for j in range(len(sequence)):
                row.append(f"{mj_matrix[i, j]:.4f}")
            writer.writerow(row)
    
    if verbose:
        print(f"  ✓ Saved MJ matrix: mj_matrix.csv ({len(sequence)}x{len(sequence)})")
    
    return path


def generate_all_outputs(
    result: FoldResult,
    config: FoldConfig,
    sequence: str,
    output_dir: str,
    fragments: Optional[List[FragmentSpec]] = None,
    mj_matrix: Optional[np.ndarray] = None,
    cvar_traces: Optional[List[List[float]]] = None,
    sa_energies: Optional[List[float]] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """Generate all comprehensive output files.
    
    Returns dict mapping output type to file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs = {}
    
    if verbose:
        print("\n[Output Generation]")
        print("-" * 40)
    
    # 1. Text summary
    outputs["summary"] = save_output_summary(result, config, output_dir, sequence, verbose)
    
    # 2. Fragment candidates CSV
    if result.fragment_candidates and fragments:
        outputs["fragment_candidates"] = save_fragment_candidates_csv(
            result.fragment_candidates, fragments, None, output_dir, verbose
        )
    
    # 3. Energy breakdown CSV (if available)
    if result.energy_breakdown:
        outputs["energy_breakdown_csv"] = save_energy_breakdown_csv(
            result.energy_breakdown, result.best_bits, output_dir, verbose
        )
    
    # 4. SA trace
    if sa_energies:
        outputs["sa_trace"] = save_sa_trace_csv(sa_energies, output_dir, verbose)
    
    # 5. SPSA traces
    if cvar_traces:
        outputs["spsa_traces"] = save_spsa_trace_csv(cvar_traces, output_dir, verbose)
    
    # 6. MJ matrix
    if mj_matrix is not None:
        outputs["mj_matrix"] = save_mj_matrix_csv(mj_matrix, sequence, output_dir, verbose)
    
    # 7. PDB ZIP
    if os.path.exists(os.path.join(output_dir, "backbone.pdb")):
        # Create pdb3d directory and copy backbone
        pdb_dir = os.path.join(output_dir, "pdb3d")
        os.makedirs(pdb_dir, exist_ok=True)
        shutil.copy(
            os.path.join(output_dir, "backbone.pdb"),
            os.path.join(pdb_dir, "best_structure.pdb")
        )
        zip_path = create_pdb_zip(pdb_dir, output_dir, verbose)
        if zip_path:
            outputs["pdb_zip"] = zip_path
    
    return outputs
