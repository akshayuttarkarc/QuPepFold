"""Main pipeline orchestration for QuPepFold.

Coordinates all stages: global search → fragmentation → quantum refinement →
stitching → 3D building → output.
"""

import os
import sys
import dataclasses
from typing import Optional
from pathlib import Path

from .types import FoldConfig, FoldResult
from .config import validate_config
from .metrics import Timer, RunMetrics
from .model.encoding import decode_turns
from .model.mj import build_mj_matrix
from .model.fragments import generate_fragments
from .model.energy_fragment import build_energy_table, compute_chain_energy
from .global_search.anneal_sa import simulated_annealing, sa_with_restarts
from .quantum.backends.aer_sampler import AerSamplerBackend
from .quantum.refine_fragment import refine_all_fragments
from .stitch.overlap_dp import stitch_fragments
from .geom.backbone_builder import build_from_turns
from .geom.pdb_writer import write_pdb
from .io.report_json import write_report
from .io.provenance import get_provenance


def _print_banner(sequence: str, config: FoldConfig):
    """Print QuPepFold banner and configuration."""
    print("=" * 60)
    print("       QuPepFold: Quantum-Classical Peptide Folding")
    print("=" * 60)
    print()
    print(f"  Sequence:       {sequence}")
    print(f"  Length:         {len(sequence)} amino acids")
    print(f"  Backend:        {config.backend}")
    print()
    print("  Configuration:")
    print(f"    Shots:          {config.shots}")
    print(f"    CVaR alpha:     {config.cvar_alpha}")
    print(f"    SA steps:       {config.sa_steps}")
    print(f"    SA restarts:    {config.sa_restarts}")
    print(f"    SPSA iters:     {config.spsa_iterations}")
    print(f"    Fragment len:   {config.fragment_length}")
    print(f"    Overlap turns:  {config.overlap_turns}")
    print()
    print("=" * 60)


def run_fold(
    sequence: str,
    config: FoldConfig,
    output_dir: str,
    verbose: bool = True,
) -> FoldResult:
    """Run the complete folding pipeline.
    
    Pipeline stages:
    1. Classical global search (SA/PT) for initial backbone
    2. Fragment generation with overlap
    3. Precompute energy tables for each fragment
    4. Quantum refinement of each fragment
    5. DP stitching of fragment candidates
    6. Build 3D backbone coordinates
    7. Write outputs (PDB, JSON report)
    
    Args:
        sequence: Amino acid sequence.
        config: FoldConfig with all parameters.
        output_dir: Directory for output files.
        verbose: Whether to print progress.
        
    Returns:
        FoldResult with all outputs.
    """
    # Validate
    sequence = sequence.upper()
    validate_config(config)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialise metrics collection
    run_metrics = RunMetrics(
        seed=config.seed,
        backend_name=config.ibm_backend if config.backend == "runtime" and config.ibm_backend else "aer",
        optimizer_name=config.optimizer,
        encoding_name=config.encoding,
    )
    pipeline_timer = Timer("pipeline_total")
    pipeline_timer.__enter__()
    
    if verbose:
        _print_banner(sequence, config)
    
    # Stage 1: Global search with restarts
    if verbose:
        print("\n[Stage 1/8] Global Simulated Annealing")
        print("-" * 40)
    
    sa_timer = Timer("global_sa")
    sa_timer.__enter__()
    
    # Use sa_with_restarts for multiple tries
    best_turns = None
    best_energy = float('inf')
    best_bits = ""
    sa_energies = []
    
    for restart in range(config.sa_restarts):
        if verbose:
            print(f"  Try {restart + 1}/{config.sa_restarts}...", end=" ", flush=True)
        
        # Create per-restart config with different seed
        restart_config = dataclasses.replace(config, seed=config.seed + restart * 1000)
        
        # Define progress callback
        def sa_callback(step: int, turns, energy: float, temp: float):
            if verbose and step % 1000 == 0:
                pct = 100 * step / config.sa_steps
                print(f"\r  Try {restart + 1}/{config.sa_restarts}: Step {step}/{config.sa_steps} ({pct:.0f}%) | E={energy:.2f} | T={temp:.2f}   ", end="", flush=True)
        
        turns, energy, bits = simulated_annealing(sequence, restart_config, callback=sa_callback if verbose else None)
        sa_energies.append(energy)
        
        if verbose:
            print(f"\r  Try {restart + 1}/{config.sa_restarts}: Energy = {energy:.2f}               ")
        
        if energy < best_energy:
            best_turns = turns
            best_energy = energy
            best_bits = bits
    
    global_turns, global_energy, global_bits = best_turns, best_energy, best_bits
    sa_timer.__exit__(None, None, None)
    run_metrics.add_stage_timing("global_sa", sa_timer.elapsed)
    run_metrics.convergence_trace_sa = sa_energies
    
    if verbose:
        print(f"\n  ✓ Best global energy: {global_energy:.2f}")
        print(f"  ✓ Best bitstring: {global_bits[:30]}...")
    
    # Stage 2: Fragmentation
    if verbose:
        print("\n[Stage 2/8] Fragment Generation")
        print("-" * 40)
    
    with Timer("fragmentation") as frag_timer:
        fragments = generate_fragments(sequence, config)
    run_metrics.add_stage_timing("fragmentation", frag_timer.elapsed)
    run_metrics.n_fragments = len(fragments)
    
    if verbose:
        print(f"  Generated {len(fragments)} overlapping fragments:")
        for i, frag in enumerate(fragments):
            print(f"    Fragment {i+1}: residues {frag.start_res}-{frag.end_res} ({frag.sequence})")
    
    # Stage 3: Energy tables
    if verbose:
        print("\n[Stage 3/8] Precomputing Energy Tables")
        print("-" * 40)
    
    mj_matrix = build_mj_matrix(sequence, seed=config.seed)
    energy_tables = []
    
    with Timer("energy_tables") as etable_timer:
        for i, frag in enumerate(fragments):
            if verbose:
                print(f"  Building table {i+1}/{len(fragments)}...", end=" ", flush=True)
            table = build_energy_table(frag, mj_matrix, config)
            energy_tables.append(table)
            if verbose:
                min_e = float(table.energies.min())
                max_e = float(table.energies.max())
                print(f"done (min={min_e:.2f}, max={max_e:.2f})")
    run_metrics.add_stage_timing("energy_tables", etable_timer.elapsed)
    
    if verbose:
        print(f"\n  ✓ Built {len(energy_tables)} energy tables (1024 states each)")
    
    # Stage 4: Quantum VQE refinement with SPSA optimization
    if verbose:
        print("\n[Stage 4/8] Quantum VQE Refinement")
        print("-" * 40)
        print(f"  Backend: {config.backend}")
        print(f"  Shots: {config.shots}")
        print(f"  SPSA iterations: {config.spsa_iterations}")
        print()
    
    # Select backend
    if config.backend == "aer":
        backend = AerSamplerBackend()
    else:
        from .quantum.backends.runtime_sampler import RuntimeSamplerBackend
        ibm_backend_name: str = config.ibm_backend or ""
        backend = RuntimeSamplerBackend(ibm_backend_name)
    
    from .quantum.refine_fragment import refine_all_fragments
    from .model.lattice import trace_positions
    from .model.encoding import decode_turns
    
    # Compute global positions from SA result to use as fixed context
    global_turns = decode_turns(global_bits)
    global_positions = trace_positions(global_turns)
    
    vqe_timer = Timer("vqe_refinement")
    vqe_timer.__enter__()
    try:
        fragment_candidates = refine_all_fragments(
            fragments=fragments,
            energy_tables=energy_tables,
            backend=backend,
            config=config,
            mj_matrix=mj_matrix,  # Pass interaction matrix for context energy
            global_bits=global_bits,
            global_positions=global_positions,  # Pass fixed context
            verbose=verbose,
        )
    finally:
        backend.close()
        vqe_timer.__exit__(None, None, None)
    run_metrics.add_stage_timing("vqe_refinement", vqe_timer.elapsed)
    run_metrics.n_qubits_per_fragment = fragments[0].n_bits if fragments else 0
    run_metrics.total_shots = config.shots * 2 * config.spsa_iterations * len(fragments)
    run_metrics.total_circuit_evaluations = 2 * config.spsa_iterations * len(fragments) + len(fragments)
    
    if verbose:
        total_cands = sum(len(c) for c in fragment_candidates)
        print(f"\n  ✓ Extracted {total_cands} total candidates")
    
    # Stage 4b: Find the best (most negative) bitstring from all VQE candidates
    if verbose:
        print("\n[Stage 4b/8] Comparing All VQE Candidates")
        print("-" * 40)
    
    from .model.energy_fragment import compute_chain_energy
    
    best_vqe_bits = None
    best_vqe_energy = float('inf')
    
    # Collect all unique bitstrings from all fragment candidates
    all_candidates = []
    for i, cands in enumerate(fragment_candidates):
        for cand in cands:
            all_candidates.append((i, cand.bits, cand.energy))
    
    # For fragments, we need to evaluate the STITCHED result to get global chain energy
    # But we can also look at which fragments have negative energies
    negative_fragments = []
    for i, cands in enumerate(fragment_candidates):
        for cand in cands:
            if cand.energy < 0:
                negative_fragments.append((i, cand.bits, cand.energy))
                if verbose:
                    print(f"  Fragment {i+1} candidate: E={cand.energy:.2f} (negative!)")
    
    if verbose:
        print(f"  Found {len(negative_fragments)} fragment candidates with negative energy")
    
    # Stage 5: Stitching
    if verbose:
        print("\n[Stage 5/8] Fragment Stitching (Dynamic Programming)")
        print("-" * 40)
    
    stitching_failed = False
    stitched_bits = None
    selected = []
    
    # Try 1: Strict stitching (exact overlap match, no overlaps)
    try:
        stitched_bits, _, selected = stitch_fragments(
            fragment_candidates,
            fragments,
            require_exact_match=True,
            overlap_penalty=config.overlap_penalty,
            mj_matrix=mj_matrix,
        )
        if verbose:
            print(f"  ✓ Strict stitching succeeded")
    except ValueError:
        # Try 2: Relaxed stitching (soft penalties, allow mismatches)
        if verbose:
            print(f"  ⚠ Strict stitching failed, trying relaxed mode...")
        try:
            stitched_bits, _, selected = stitch_fragments(
                fragment_candidates,
                fragments,
                require_exact_match=False,  # Allow mismatches
                mismatch_penalty=10.0,      # Soft penalty
                overlap_penalty=100.0,      # Lower penalty
                mj_matrix=mj_matrix,
            )
            if verbose:
                print(f"  ✓ Relaxed stitching succeeded")
        except ValueError as e:
            # Fallback to SA
            stitching_failed = True
            stitched_bits = global_bits
            if verbose:
                print(f"  ⚠ Relaxed stitching also failed: {e}")
                print(f"  → Using global SA result (E={global_energy:.2f})")
    
    # Compute energy for stitched chain
    stitched_energy = compute_chain_energy(stitched_bits, sequence, mj_matrix, config)
    
    if not stitching_failed and verbose:
        print(f"  Selected candidates:")
        for i, cand in enumerate(selected):
            print(f"    Fragment {i+1}: {cand.bits[:15]}... (E={cand.energy:.2f})")
        print(f"\n  Stitched chain energy: {stitched_energy:.2f}")
    
    # Stage 5b: Post-stitch local refinement
    # Ensures result is locally optimal under true global energy
    if verbose:
        print("\n[Stage 5b/8] Post-Stitch Local Refinement")
        print("-" * 40)
    
    from .global_search.local_refine import local_refine
    refined_bits, refined_energy, improved = local_refine(
        stitched_bits,
        sequence,
        config,
        steps=500,
        verbose=verbose,
    )
    
    if improved:
        stitched_bits = refined_bits
        stitched_energy = refined_energy
    
    # Compare: global SA result vs refined stitched result
    # Use the one with the MOST NEGATIVE (lowest) energy
    final_bits = stitched_bits
    final_energy = stitched_energy
    used_global = False
    
    if global_energy < stitched_energy:
        final_bits = global_bits
        final_energy = global_energy
        used_global = True
    
    if verbose:
        print(f"\n  Global search energy:  {global_energy:.2f}")
        print(f"  Refined stitch energy: {stitched_energy:.2f}")
        if used_global:
            print(f"  → Using global result (better energy)")
        else:
            print(f"  → Using stitched result")
        print(f"  ✓ Final energy: {final_energy:.2f}")
    
    # Stage 6: Build 3D structure
    if verbose:
        print("\n[Stage 6/7] Building 3D Backbone")
        print("-" * 40)
    
    turns = decode_turns(final_bits)  # Use the better result
    atoms = build_from_turns(sequence, turns)
    
    pdb_path = os.path.join(output_dir, "backbone.pdb")
    # Pass turns so write_pdb uses turn codes directly for HELIX/SHEET records
    write_pdb(atoms, pdb_path, title=f"QuPepFold: {sequence}", turns=turns)

    if verbose:
        # Report secondary structure from turn codes (same source as PDB header)
        from .geom.pdb_writer import detect_secondary_structure
        helices, sheets = detect_secondary_structure(turns)
        if helices:
            print(f"  Detected {len(helices)} helix region(s): " +
                  ", ".join(f"res {s}-{e}" for s, e in helices))
        if sheets:
            print(f"  Detected {len(sheets)} sheet region(s): " +
                  ", ".join(f"res {s}-{e}" for s, e in sheets))
        if not helices and not sheets:
            print(f"  No regular secondary structure (all coil/loop)")
        print(f"  Built {len(atoms)} atoms")
        print(f"  ✓ Wrote: {pdb_path}")
    
    # Build provenance, metrics, and result
    provenance = get_provenance(
        backend_name=config.ibm_backend if config.backend == "runtime" else "aer",
        seed=config.seed,
    )

    quantum_metrics = {
        "n_qubits": fragments[0].n_bits if fragments else 0,
        "shots_per_circuit": config.shots,
        "spsa_iterations": config.spsa_iterations,
        "n_fragments": len(fragments),
        "total_shots": config.shots * 2 * config.spsa_iterations * len(fragments),
        "total_circuit_evaluations": 2 * config.spsa_iterations * len(fragments) + len(fragments),
    }

    relaxed_pdb_path = None  # Structure relaxation removed from pipeline

    result = FoldResult(
        sequence=sequence,
        global_bits=global_bits,
        global_energy=global_energy,
        fragment_candidates=fragment_candidates,
        stitched_bits=final_bits,
        stitched_energy=final_energy,
        pdb_path=pdb_path,
        relaxed_pdb_path=relaxed_pdb_path,
        provenance=provenance,
        quantum_metrics=quantum_metrics,
        win_source="global_sa" if used_global else "stitched",
        convergence_trace=sa_energies,
    )

    # Stage 7: Write report and all outputs
    if verbose:
        print("\n[Stage 7/7] Writing Report")
        print("-" * 40)

    report_path = os.path.join(output_dir, "report.json")
    write_report(result, config, report_path, provenance)
    
    if verbose:
        print(f"  ✓ Wrote: {report_path}")
    
    # Generate comprehensive outputs (CSVs, summary, plots)
    from .io.outputs import generate_all_outputs
    from .io.plots import generate_all_plots
    
    try:
        # Generate all CSV and text outputs
        output_files = generate_all_outputs(
            result=result,
            config=config,
            sequence=sequence,
            output_dir=output_dir,
            fragments=fragments,
            mj_matrix=mj_matrix,
            sa_energies=[global_energy],  # SA restart energies
            verbose=verbose,
        )
        
        # Generate all plots
        plot_files = generate_all_plots(
            result=result,
            output_dir=output_dir,
            energy_components=result.energy_breakdown,
        )
        
        if verbose and plot_files:
            for p in plot_files:
                print(f"  ✓ Saved plot: {os.path.basename(p)}")
    except Exception as e:
        if verbose:
            print(f"  ⚠ Output generation warning: {e}")
    
    # Finalize metrics
    pipeline_timer.__exit__(None, None, None)
    run_metrics.wall_clock_seconds = pipeline_timer.elapsed
    result.run_metrics = run_metrics
    
    # Write metrics JSON
    metrics_path = os.path.join(output_dir, "run_metrics.json")
    run_metrics.to_json(metrics_path)
    
    # Final summary
    if verbose:
        print("\n" + "=" * 60)
        print("  QuPepFold Complete!")
        print("=" * 60)
        print(f"\n  Sequence:        {sequence}")
        print(f"  Global Energy:   {global_energy:.2f}")
        print(f"  Stitched Energy: {stitched_energy:.2f}")
        print(f"  Wall-clock:      {run_metrics.wall_clock_seconds:.2f}s")
        print(f"\n  Output Files:")
        print(f"    Backbone PDB:  {pdb_path}")
        print(f"    Report JSON:   {report_path}")
        print(f"    Run Metrics:   {metrics_path}")
        print(f"    Summary TXT:   {os.path.join(output_dir, 'output_summary.txt')}")
        print()
    
    return result


def run_fold_simple(sequence: str, output_dir: str = "./qupepfold_output") -> FoldResult:
    """Run folding with default configuration.
    
    Convenience function for quick testing.
    
    Args:
        sequence: Amino acid sequence.
        output_dir: Output directory.
        
    Returns:
        FoldResult.
    """
    from .config import get_default_config
    config = get_default_config()
    return run_fold(sequence, config, output_dir)
