from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Generate turn-to-qubit mapping
def generate_turn2qubit(protein_sequence):
    N = len(protein_sequence)
    if N < 2:
        raise ValueError("Protein sequence must have at least 2 beads.")
    num_turns = 2 * (N - 1)
    fixed_bits = '0100q1'
    variable_bits = 'q' * (num_turns - len(fixed_bits))
    return fixed_bits + variable_bits, fixed_bits, variable_bits

# Build MJ interaction matrix
def build_mj_interactions(protein):
    N = len(protein)
    mat = np.zeros((N, N))
    np.random.seed(29507)
    MJ = np.random.rand(20, 20) * -6
    MJ = np.triu(MJ) + np.triu(MJ, 1).T
    acids = ["C", "M", "F", "I", "L", "V", "W", "Y", "A", "G", "T", "S", "N", "Q", "D", "E", "H", "R", "K", "P"]
    acid2idx = {acid: idx for idx, acid in enumerate(acids)}
    for i in range(N):
        for j in range(N):
            mat[i, j] = MJ[acid2idx[protein[i]], acid2idx[protein[j]]]
    return mat

# Compute energy from bitstrings
def exact_hamiltonian(bitstrings, hyperParams):
    lambda_dis = 720
    lambda_loc = 20
    lambda_back = 50
    energies = np.zeros(len(bitstrings))
    num_beads = len(hyperParams["protein"])

    for idx, bitstring in enumerate(bitstrings):
        config = list(hyperParams["turn2qubit"])
        q_indices = [i for i, x in enumerate(config) if x == 'q']
        for i, q_idx in enumerate(q_indices):
            config[q_idx] = bitstring[i]
        config = ''.join(config)
        turns = [int(config[i:i+2], 2) for i in range(0, len(config), 2)]
        energies[idx] = lambda_back * sum(turns[i] == turns[i+1] for i in range(len(turns) - 1))
        curr_interaction_qubit = hyperParams["numQubitsConfig"]
        for i in range(num_beads - 4):
            for j in range(i + 5, num_beads, 2):
                if curr_interaction_qubit >= len(bitstring):
                    break
                if bitstring[curr_interaction_qubit] == '0':
                    curr_interaction_qubit += 1
                    continue
                energies[idx] += hyperParams["interactionEnergy"][i, j]
                delta_n_ij = np.zeros(4)
                delta_n_ir = np.zeros(4)
                delta_n_mj = np.zeros(4)
                for k in range(4):
                    delta_n_ij[k] = sum((-1)**m * (turns[m] == k) for m in range(i, j))
                    delta_n_ir[k] = sum((-1)**m * (turns[m] == k) for m in range(i, j - 1))
                    delta_n_mj[k] = sum((-1)**m * (turns[m] == k) for m in range(i + 1, j))
                d_ij = np.linalg.norm(delta_n_ij)**2
                d_ir = np.linalg.norm(delta_n_ir)**2
                d_mj = np.linalg.norm(delta_n_mj)**2
                energies[idx] += lambda_dis * (d_ij - 1)
                energies[idx] += lambda_loc * (2 - d_ir)
                energies[idx] += lambda_loc * (2 - d_mj)
                if i - 1 >= 0:
                    for k in range(4):
                        delta_n_mj[k] = sum((-1)**m * (turns[m] == k) for m in range(i - 1, j))
                    d_mj = np.linalg.norm(delta_n_mj)**2
                    energies[idx] += lambda_loc * (2 - d_mj)
                if j + 1 < num_beads:
                    for k in range(4):
                        delta_n_ir[k] = sum((-1)**m * (turns[m] == k) for m in range(i, j + 1))
                    d_ir = np.linalg.norm(delta_n_ir)**2
                    energies[idx] += lambda_loc * (2 - d_ir)
                curr_interaction_qubit += 1
    return energies

# Ansatz circuit
def protein_config_ansatz(parameters):
    num_qubits = len(parameters) // 3
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
        qc.ry(parameters[i], i)
        qc.rx(parameters[i + num_qubits], i)
        qc.rz(parameters[i + 2 * num_qubits], i)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    for i in range(num_qubits):
        qc.t(i)
    for i in range(num_qubits):
        qc.ry(parameters[i], i)
        qc.rx(parameters[i + num_qubits], i)
        qc.rz(parameters[i + 2 * num_qubits], i)
    qc.measure_all()
    return qc

# CVaR objective function
def protein_vqe_objective(parameters, hyperParams):
    ansatz = protein_config_ansatz(parameters)
    simulator = AerSimulator()
    compiled_circuit = transpile(ansatz, simulator)
    job = simulator.run(compiled_circuit, shots=hyperParams["numShots"])
    result = job.result()
    counts = result.get_counts()
    bitstrings = [format(int(k.replace(" ", ""), 2), f'0{len(parameters) // 3}b') for k in counts]
    probs = np.array(list(counts.values())) / hyperParams["numShots"]
    energies = exact_hamiltonian(bitstrings, hyperParams)
    sort_idx = np.argsort(energies)
    sorted_probs = probs[sort_idx]
    sorted_energies = energies[sort_idx]
    alpha = 0.025
    cut_idx = np.searchsorted(np.cumsum(sorted_probs), alpha)
    cvar_probs = sorted_probs[:cut_idx + 1]
    cvar_probs[-1] += alpha - np.sum(cvar_probs)
    cvar_energy = np.dot(cvar_probs, sorted_energies[:cut_idx + 1]) / alpha
    return cvar_energy

# --- MAIN EXECUTION ---
protein_sequence = input("Enter the protein sequence (e.g., APRLRFY): ").strip().upper()
turn2qubit, fixed_bits, variable_bits = generate_turn2qubit(protein_sequence)
num_qubits_config = turn2qubit.count('q')
num_qubits_interaction = 2
total_qubits = num_qubits_config + num_qubits_interaction

interaction_energy = build_mj_interactions(protein_sequence)

hyperParams = {
    "protein": protein_sequence,
    "turn2qubit": turn2qubit,
    "numQubitsConfig": num_qubits_config,
    "interactionEnergy": interaction_energy,
    "numShots": 1024
}

obj_fcn = lambda theta: protein_vqe_objective(theta, hyperParams)

# --- ITERATE CVaR VQE ---
cvar_results = []
optimal_params = None
min_energy = np.inf

for i in range(8):
    initial_parameters = np.random.uniform(-np.pi, np.pi, size=3 * total_qubits)
    result = minimize(obj_fcn, initial_parameters, method='COBYLA')
    cvar_results.append(result.fun)
    if result.fun < min_energy:
        min_energy = result.fun
        optimal_params = result.x

# --- OUTPUT TEXT SUMMARY ---
summary = f"""
--- Quantum Protein Folding Summary ---

Protein Sequence: {protein_sequence}
Fixed Bits:       {fixed_bits}
Variable Bits:    {variable_bits}

MJ Interaction Energy Matrix:
{interaction_energy}

Minimum CVaR Energy: {min_energy:.5f}
"""

print(summary)
with open("output_summary.txt", "w") as f:
    f.write(summary)

# --- SAVE OPTIMAL CIRCUIT ---
optimal_circuit = protein_config_ansatz(optimal_params)
circuit_drawer(optimal_circuit.remove_final_measurements(inplace=False), output='mpl', filename="optimal_circuit.png")
print("Saved: optimal_circuit.png")

# --- SCATTER PLOT ---
plt.figure()
plt.scatter(range(1, 9), cvar_results, color='blue', marker='o')
plt.title("CVaR Energies Across 8 Iterations")
plt.xlabel("Iteration")
plt.ylabel("CVaR Energy")
plt.grid(True)
plt.savefig("cvar_scatter.png")
plt.close()
print("Saved: cvar_scatter.png")

# --- BITSTRING HISTOGRAM ---
simulator = AerSimulator()
compiled_optimal = transpile(optimal_circuit, simulator)
job = simulator.run(compiled_optimal, shots=hyperParams["numShots"])
result = job.result()
counts = result.get_counts()

threshold = 0.02
total_shots = sum(counts.values())
filtered_counts = {k: v / total_shots for k, v in counts.items() if (v / total_shots) >= threshold}
sorted_counts = dict(sorted(filtered_counts.items(), key=lambda item: item[1], reverse=True))

plt.figure(figsize=(10, 5))
plt.bar(sorted_counts.keys(), sorted_counts.values(), color='skyblue', edgecolor='black')
plt.ylabel("Probability")
plt.xlabel("Bitstring")
plt.title("High-Probability Bitstring Outcomes (Threshold = 2%)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("bitstring_histogram.png")
plt.close()
print("Saved: bitstring_histogram.png")
