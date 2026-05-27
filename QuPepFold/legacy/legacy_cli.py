# qupepfold/cli.py
import os
import csv
import argparse

from . import (
    generate_turn2qubit,
    count_interaction_qubits,
    build_mj_interactions,
    optimize_cvar_multistart,
    build_scalable_ansatz,
    statevector_fold_probs,
    exact_hamiltonian,
)

__all__ = ["main"]

def main():
    parser = argparse.ArgumentParser(prog="qupepfold")
    parser.add_argument("--seq", required=True, help="Protein sequence (2–10 aa, e.g., APRLRFY)")
    parser.add_argument("--tries", type=int, default=50, help="Number of CVaR multi-start attempts")
    parser.add_argument("--alpha", type=float, default=0.025, help="CVaR tail mass (0<alpha<1)")
    parser.add_argument("--shots", type=int, default=1024, help="(Informational) shots to report")
    parser.add_argument("--out", default="./results", help="Output directory")
    parser.add_argument("--write-csv", action="store_true", help="Write bitstring_summary.csv")
    args = parser.parse_args()

    seq = args.seq.upper()
    if not (2 <= len(seq) <= 10) or any(c not in "ARNDCEQGHILKMFPSTWYV" for c in seq):
        raise SystemExit("ERROR: --seq must be 2–10 amino acids using standard one-letter codes.")

    # Build mapping & hyper (aligned with the core module)
    turn2qubit, fixed_bits, variable_bits = generate_turn2qubit(seq)
    num_q_cfg = turn2qubit.count("q")
    num_q_int = count_interaction_qubits(seq)
    hyper = {
        "protein": seq,
        "turn2qubit": turn2qubit,
        "numQubitsConfig": num_q_cfg,
        "numQubitsInteraction": num_q_int,
        "interactionEnergy": build_mj_interactions(seq),
        "numShots": int(args.shots),
    }

    print("=== Qubit mapping ===")
    print("turn2qubit:", turn2qubit)
    print("fixed bits:", fixed_bits)
    print("var bits:  ", variable_bits)
    print(f"cfg qubits: {num_q_cfg}  |  int qubits: {num_q_int}  |  total (incl. ancilla): {num_q_cfg+num_q_int+1}")

    # Optimize CVaR (multi-start)
    print(f"\n[CVaR-VQE] alpha={args.alpha}, tries={args.tries}")
    best_x, best_cvar, trace = optimize_cvar_multistart(hyper, args.tries, args.alpha)
    print(f"[CVaR-VQE] best CVaR energy: {best_cvar:.6f}")

    # Distribution at optimum (statevector)
    qc = build_scalable_ansatz(best_x, hyper, measure=False)
    probs = statevector_fold_probs(qc, hyper)          # dict: bitstring -> probability
    states = list(probs.keys())
    energies = exact_hamiltonian(states, hyper)

    # Report: most probable & most negative-energy bitstrings
    s_most_prob = max(states, key=lambda s: probs[s])
    s_min_idx = int(min(range(len(states)), key=lambda i: energies[i]))
    s_min_energy = states[s_min_idx]

    print("\n=== Results at optimum ===")
    print(f"Most probable bitstring : {s_most_prob} (P={probs[s_most_prob]:.6f})")
    print(f"Lowest-energy bitstring : {s_min_energy} (E={energies[s_min_idx]:.6f})")

    # Optional CSV dump
    if args.write_csv:
        os.makedirs(args.out, exist_ok=True)
        csv_path = os.path.join(args.out, "bitstring_summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bitstring", "cfg_bits", "probability", "energy"])
            for s, e in zip(states, energies):
                w.writerow([s, s[:num_q_cfg], float(probs[s]), float(e)])
        print(f"\nWrote CSV → {csv_path}")
