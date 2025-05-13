# QuPepFold

Quantum peptide‐folding simulations built on Qiskit.

## Installation

```bash
pip install qupepfold

 OR

 git clone https://github.com/yourusername/qupepfold.git
cd qupepfold
pip install


from qupepfold import generate_turn2qubit, protein_vqe_objective

seq = "HHPHPH"
fixed, fb, vb = generate_turn2qubit(seq)
result = protein_vqe_objective(
    protein_sequence=seq,
    hyperParams={…}
)
print(result)



