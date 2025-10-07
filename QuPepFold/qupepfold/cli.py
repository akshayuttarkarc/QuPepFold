# import argparse
# from . import generate_turn2qubit, protein_vqe_objective

# def main():
#     parser = argparse.ArgumentParser(prog="qupepfold")
#     parser.add_argument("--seq", required=True, help="Protein sequence")
#     parser.add_argument("--maxiter", type=int, default=200, help="VQE max iterations")
#     args = parser.parse_args()

#     fixed, fb, vb = generate_turn2qubit(args.seq)
#     print("Qubit mapping:", fixed, fb, vb)
#     res = protein_vqe_objective(
#         protein_sequence=args.seq,
#         hyperParams={"optimizer": "SLSQP", "maxiter": args.maxiter}
#     )
#     print("Energy:", res["energy"])

# if __name__ == "__main__":
#     main()
