"""CLI entry point for QuPepFold."""

import argparse
import sys
import os


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="qupepfold",
        description="QuPepFold: Quantum-Classical Hybrid Peptide Folding"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Fold command
    fold_parser = subparsers.add_parser("fold", help="Run folding pipeline")
    
    # Input options
    input_group = fold_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--seq", type=str, help="Amino acid sequence")
    input_group.add_argument("--fasta", type=str, help="Path to FASTA file")
    
    # Backend options
    fold_parser.add_argument("--backend", choices=["aer", "runtime"], default="aer",
                            help="Quantum backend (default: aer)")
    fold_parser.add_argument("--ibm-backend", type=str, default=None,
                            help="IBM backend name for runtime mode (e.g., ibm_brisbane)")
    fold_parser.add_argument("--ibm-token", type=str, default=None,
                            help="IBM Quantum API token (or set QISKIT_IBM_TOKEN env var)")
    
    # Quantum parameters
    fold_parser.add_argument("--shots", type=int, default=2000,
                            help="Shots per circuit (default: 2000)")
    fold_parser.add_argument("--spsa-iters", type=int, default=50,
                            help="SPSA iterations (default: 50)")
    fold_parser.add_argument("--ansatz-depth", type=int, default=2,
                            help="Ansatz depth (default: 2)")
    
    # CVaR parameters (from original)
    fold_parser.add_argument("--alpha", type=float, default=0.25,
                            help="CVaR alpha - tail probability (default: 0.25)")
    fold_parser.add_argument("--tries", type=int, default=50,
                            help="Optimization restarts/tries (default: 50)")
    
    # Fragment parameters
    fold_parser.add_argument("--frag-len", type=int, default=7,
                            help="Fragment length (default: 7)")
    fold_parser.add_argument("--overlap", type=int, default=2,
                            help="Overlap turns (default: 2)")
    
    # Global search parameters
    fold_parser.add_argument("--sa-steps", type=int, default=5000,
                            help="SA steps (default: 5000)")
    fold_parser.add_argument("--sa-restarts", type=int, default=3,
                            help="SA restarts (default: 3)")
    
    # Output options
    fold_parser.add_argument("--out", type=str, default="./qupepfold_output",
                            help="Output directory")
    fold_parser.add_argument("--export-prob", type=float, default=0.02,
                            help="Min probability for PDB export (default: 0.02)")
    fold_parser.add_argument("--seed", type=int, default=42,
                            help="Random seed (default: 42)")
    fold_parser.add_argument("--quiet", action="store_true",
                            help="Suppress output")
    fold_parser.add_argument("--no-plots", action="store_true",
                            help="Skip generating plots")
    fold_parser.add_argument("--no-csv", action="store_true",
                            help="Skip generating CSV files")
    
    # Version command
    subparsers.add_parser("version", help="Show version")

    # Preprocess command
    pre_parser = subparsers.add_parser("preprocess", help="Fragment a protein into ranked jobs")
    pre_in = pre_parser.add_mutually_exclusive_group(required=True)
    pre_in.add_argument("--seq", type=str, help="Amino acid sequence")
    pre_in.add_argument("--fasta", type=str, help="Path to FASTA file")
    pre_parser.add_argument("--strategy", default="fixed_window",
                            choices=["fixed_window", "disorder", "domain", "user_defined"],
                            help="Fragmentation strategy (default: fixed_window)")
    pre_parser.add_argument("--window", type=int, default=7,
                            help="Fragment window length in residues (default: 7)")
    pre_parser.add_argument("--overlap", type=int, default=2,
                            help="Overlap turns between fragments (default: 2)")
    pre_parser.add_argument("--protein-id", type=str, default="unknown",
                            help="Protein identifier for manifest")
    pre_parser.add_argument("--regions", type=str, default=None,
                            help="User-defined ranges as start:end pairs, e.g. 0:7,5:12")
    pre_parser.add_argument("--out", type=str, default="./manifest.json",
                            help="Output manifest JSON path (default: ./manifest.json)")
    pre_parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "version":
        from . import __version__
        print(f"qupepfold {__version__}")
        sys.exit(0)

    if args.command == "fold":
        _run_fold(args)

    if args.command == "preprocess":
        _run_preprocess(args)


def _run_fold(args):
    """Execute fold command."""
    from .pipeline import run_fold
    from .config import get_default_config
    from . import __version__
    
    print(f"\n[QuPepFold v{__version__}] Starting pipeline...")
    print(f" > Hybrid VQE Enabled: YES")
    print(f" > Relaxed Stitching: YES\n")
    
    # Get sequence
    if args.seq:
        sequence = args.seq.upper()
    else:
        sequence = _read_fasta(args.fasta)
    
    # Validate - only 20 standard amino acids allowed
    VALID_AMINO_ACIDS = set("ARNDCEQGHILKMFPSTWYV")
    if not all(aa in VALID_AMINO_ACIDS for aa in sequence):
        invalid = [aa for aa in sequence if aa not in VALID_AMINO_ACIDS]
        print(f"Error: Invalid character(s) in sequence: {set(invalid)}", file=sys.stderr)
        print(f"Only these 20 amino acid codes are allowed:", file=sys.stderr)
        print(f"  A, R, N, D, C, E, Q, G, H, I, L, K, M, F, P, S, T, W, Y, V", file=sys.stderr)
        sys.exit(1)
    
    if args.backend == "runtime" and not args.ibm_backend:
        print("Error: --ibm-backend required for runtime backend", file=sys.stderr)
        sys.exit(1)
    
    # Get IBM token from CLI or environment
    ibm_token = args.ibm_token or os.environ.get("QISKIT_IBM_TOKEN")
    
    # Build config
    config = get_default_config(
        shots=args.shots,
        spsa_iterations=args.spsa_iters,
        ansatz_depth=args.ansatz_depth,
        fragment_length=args.frag_len,
        overlap_turns=args.overlap,
        sa_steps=args.sa_steps,
        sa_restarts=args.sa_restarts,
        cvar_alpha=args.alpha,
        optimization_tries=args.tries,
        export_prob_threshold=args.export_prob,
        backend=args.backend,
        ibm_backend=args.ibm_backend,
        ibm_token=ibm_token,
        seed=args.seed,
    )
    
    try:
        result = run_fold(
            sequence=sequence,
            config=config,
            output_dir=args.out,
            verbose=not args.quiet,
            generate_csv=not args.no_csv,
            generate_plots=not args.no_plots,
        )
        
        if not args.quiet:
            print(f"\n=== Results ===")
            print(f"Sequence: {sequence}")
            print(f"Global search energy: {result.global_energy:.2f}")
            print(f"Stitched energy: {result.stitched_energy:.2f}")
            print(f"Output: {result.pdb_path}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _run_preprocess(args):
    """Execute preprocess command — fragment a protein and emit a manifest."""
    from .config import get_default_config
    from .model.fragment_strategies import get_strategy
    from .model.manifest import build_manifest

    sequence = args.seq.upper() if args.seq else _read_fasta(args.fasta)
    VALID_AA = set("ARNDCEQGHILKMFPSTWYV")
    bad = [aa for aa in sequence if aa not in VALID_AA]
    if bad:
        print(f"Error: Invalid residues: {set(bad)}", file=sys.stderr)
        sys.exit(1)

    config = get_default_config(
        fragment_length=args.window,
        overlap_turns=args.overlap,
        fragment_strategy=args.strategy,
    )

    strategy_kwargs = {}
    if args.strategy == "user_defined":
        if not args.regions:
            print("Error: --regions is required with --strategy user_defined", file=sys.stderr)
            sys.exit(1)
        try:
            strategy_kwargs["regions"] = _parse_regions(args.regions)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    strategy = get_strategy(args.strategy, **strategy_kwargs)
    fragments = strategy.generate(sequence, config)

    # Sort by priority (highest first) for display
    ranked = sorted(fragments, key=lambda f: -f.priority)

    manifest = build_manifest(
        sequence=sequence,
        fragments=fragments,       # preserve original order in manifest
        strategy_used=args.strategy,
        protein_id=args.protein_id,
        metadata={"window": args.window, "overlap": args.overlap},
    )

    manifest.to_json(args.out)

    if not args.quiet:
        print(manifest.summary())
        print(f"\n  Ranked fragment order (by priority):")
        for i, f in enumerate(ranked):
            print(f"    #{i+1}: res {f.start_res}–{f.end_res} ({f.sequence}) "
                  f"P={f.priority:.3f}")
        print(f"\n  ✓ Manifest written → {args.out}")


def _read_fasta(filepath: str) -> str:
    """Read FASTA file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    seq = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line.startswith(">"):
                seq.append(line.upper())
    return "".join(seq)


def _parse_regions(value: str):
    """Parse comma-separated start:end residue ranges."""
    regions = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid region '{part}'. Expected start:end")
        start_text, end_text = part.split(":", 1)
        try:
            start = int(start_text)
            end = int(end_text)
        except ValueError as exc:
            raise ValueError(f"Invalid region '{part}'. Bounds must be integers") from exc
        if start < 0 or end <= start:
            raise ValueError(f"Invalid region '{part}'. Need 0 <= start < end")
        regions.append((start, end))

    if not regions:
        raise ValueError("At least one region is required")
    return regions


if __name__ == "__main__":
    main()
