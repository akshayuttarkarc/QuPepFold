#!/usr/bin/env python3
"""QuPepFold CLI - Quantum-Classical Hybrid Peptide Folding.

Usage:
    qupepfold fold --seq SEQUENCE [options]
    qupepfold fold --fasta FILE [options]
    
Examples:
    qupepfold fold --seq ACDEFGHI --backend aer --shots 2000 --out ./output
    qupepfold fold --seq ACDEFGHIKLMN --backend runtime --ibm-backend ibm_brisbane
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        prog="qupepfold",
        description="QuPepFold: Quantum-Classical Hybrid Peptide Folding"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Fold command
    fold_parser = subparsers.add_parser("fold", help="Run folding pipeline")
    
    # Input options
    input_group = fold_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--seq", type=str, help="Amino acid sequence (e.g., ACDEFGHI)")
    input_group.add_argument("--fasta", type=str, help="Path to FASTA file")
    
    # Backend options
    fold_parser.add_argument("--backend", choices=["aer", "runtime"], default="aer",
                            help="Quantum backend: aer (local) or runtime (IBM QPU)")
    fold_parser.add_argument("--ibm-backend", type=str, default=None,
                            help="IBM backend name for runtime mode (e.g., ibm_brisbane)")
    
    # Quantum parameters
    fold_parser.add_argument("--shots", type=int, default=2000,
                            help="Number of shots per circuit (default: 2000)")
    fold_parser.add_argument("--spsa-iters", type=int, default=50,
                            help="SPSA optimizer iterations (default: 50)")
    fold_parser.add_argument("--ansatz-depth", type=int, default=2,
                            help="Ansatz circuit depth (default: 2)")
    
    # Fragment parameters
    fold_parser.add_argument("--frag-len", type=int, default=6,
                            help="Fragment length in residues (default: 6)")
    fold_parser.add_argument("--overlap", type=int, default=2,
                            help="Overlap turns between fragments (default: 2)")
    
    # Global search parameters
    fold_parser.add_argument("--sa-steps", type=int, default=5000,
                            help="Simulated annealing steps (default: 5000)")
    
    # Output options
    fold_parser.add_argument("--out", type=str, default="./qupepfold_output",
                            help="Output directory (default: ./qupepfold_output)")
    fold_parser.add_argument("--seed", type=int, default=42,
                            help="Random seed (default: 42)")
    fold_parser.add_argument("--quiet", action="store_true",
                            help="Suppress progress output")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "fold":
        run_fold_command(args)


def run_fold_command(args):
    """Execute the fold command."""
    # Import here to avoid slow startup for --help
    from qupepfold import run_fold
    from qupepfold.config import get_default_config
    
    # Get sequence
    if args.seq:
        sequence = args.seq.upper()
    else:
        sequence = read_fasta(args.fasta)
    
    # Validate sequence
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if not all(aa in valid_aa for aa in sequence):
        invalid = [aa for aa in sequence if aa not in valid_aa]
        print(f"Error: Invalid amino acids in sequence: {set(invalid)}", file=sys.stderr)
        sys.exit(1)
    
    # Validate runtime backend
    if args.backend == "runtime" and not args.ibm_backend:
        print("Error: --ibm-backend required when using --backend runtime", file=sys.stderr)
        sys.exit(1)
    
    # Build config
    config = get_default_config(
        shots=args.shots,
        spsa_iterations=args.spsa_iters,
        ansatz_depth=args.ansatz_depth,
        fragment_length=args.frag_len,
        overlap_turns=args.overlap,
        sa_steps=args.sa_steps,
        backend=args.backend,
        ibm_backend=args.ibm_backend,
        seed=args.seed,
    )
    
    # Run pipeline
    try:
        result = run_fold(
            sequence=sequence,
            config=config,
            output_dir=args.out,
            verbose=not args.quiet,
        )
        
        if not args.quiet:
            print("\n=== Results ===")
            print(f"Sequence: {result.sequence}")
            print(f"Global search energy: {result.global_energy:.2f}")
            print(f"Stitched energy: {result.stitched_energy:.2f}")
            print(f"PDB output: {result.pdb_path}")
            if result.relaxed_pdb_path:
                print(f"Relaxed PDB: {result.relaxed_pdb_path}")
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def read_fasta(filepath: str) -> str:
    """Read sequence from FASTA file."""
    if not os.path.exists(filepath):
        print(f"Error: FASTA file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    
    sequence = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                continue  # Skip header
            sequence.append(line.upper())
    
    return "".join(sequence)


if __name__ == "__main__":
    main()
