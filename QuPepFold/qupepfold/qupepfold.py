import os
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math, numpy as _np

_BOND = {"C-N": 1.329, "N-CA": 1.458, "CA-C": 1.525, "C=O": 1.229}
_ANGLE = {
    "C-N-CA": math.radians(121.7),
    "N-CA-C": math.radians(110.4),
    "CA-C-N": math.radians(116.2),
    "CA-C-O": math.radians(120.8),
}
OMEGA_TRANS = math.radians(180.0)
PHI_PSI = {
    "H": (math.radians(-60.0), math.radians(-45.0)),
    "E": (math.radians(-135.0), math.radians(135.0)),
    "C": (math.radians(-70.0), math.radians(140.0)),
}
_three_letter = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS",
    "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
}

def _decode_turns_from_bitstring(bitstring, turn2qubit):
    cfg = list(turn2qubit)
    q_idx = [i for i, ch in enumerate(cfg) if ch == 'q']
    for i, tpos in enumerate(q_idx):
        if i >= len(bitstring):
            break
        cfg[tpos] = bitstring[i]
    cfg = ''.join(cfg)
    turns = [int(cfg[i:i+2], 2) for i in range(0, len(cfg), 2)]
    return turns

def _infer_secondary_from_turns(turns):
    n = len(turns) + 1
    ss = ["C"] * n
    for i in range(n - 1):
        win = turns[max(0, i - 2):min(len(turns), i + 2)]
        if len(win) >= 3:
            if len(set(win)) == 1:
                ss[i] = "H"
            elif all((win[k] % 2) != (win[k + 1] % 2) for k in range(len(win) - 1)):
                ss[i] = "E"
    def expand(tag, minlen):
        i = 0
        while i < n:
            if ss[i] == tag:
                j = i
                while j < n and ss[j] == tag:
                    j += 1
                if j - i < minlen:
                    for k in range(i, j):
                        ss[k] = "C"
                i = j
            else:
                i += 1
    expand("H", 4)
    expand("E", 3)
    return ss

def _normalize(v):
    v = _np.asarray(v, float)
    n = _np.linalg.norm(v)
    if n < 1e-8:
        return v * 0.0
    return v / n

def _orthonormal_frame(a, b, c):
    cb = _normalize(_np.asarray(b) - _np.asarray(c))
    t = _np.asarray(b) - _np.asarray(a)
    n = _np.cross(t, cb)
    if _np.linalg.norm(n) < 1e-8:
        tmp = _np.array([1.0, 0.0, 0.0])
        if abs(_np.dot(tmp, cb)) > 0.9:
            tmp = _np.array([0.0, 1.0, 0.0])
        n = _np.cross(tmp, cb)
    n = _normalize(n)
    m = _normalize(_np.cross(n, cb))
    return m, n, cb

def _place_atom(pA, pB, pC, bond_len, angle_rad, dihedral_rad):
    m, n, cb = _orthonormal_frame(pA, pB, pC)
    x = -bond_len * math.cos(angle_rad)
    y = bond_len * math.cos(dihedral_rad) * math.sin(angle_rad)
    z = bond_len * math.sin(dihedral_rad) * math.sin(angle_rad)
    C = _np.asarray(pC, float)
    D = C + x * cb + y * m + z * n
    return tuple(D.tolist())

def _seed_first_residue(ss0):
    N1 = (0.0, 0.0, 0.0)
    CA1 = (_BOND["N-CA"], 0.0, 0.0)
    ang = _ANGLE["N-CA-C"]
    vx = -math.cos(ang)
    vy = math.sin(ang)
    C1 = (CA1[0] + _BOND["CA-C"] * vx, CA1[1] + _BOND["CA-C"] * vy, 0.0)
    O1 = _place_atom(N1, CA1, C1, _BOND["C=O"], _ANGLE["CA-C-O"], 0.0)
    return N1, CA1, C1, O1

def _build_backbone_3d(seq, ss):
    N = len(seq)
    N1, CA1, C1, O1 = _seed_first_residue(ss[0])
    atoms = [{"name": "N", "coords": N1}, {"name": "CA", "coords": CA1},
             {"name": "C", "coords": C1}, {"name": "O", "coords": O1}]
    prevA, prevB, prevC = N1, CA1, C1
    for i in range(1, N):
        phi, psi = PHI_PSI[ss[i]]
        Ni = _place_atom(prevA, prevB, prevC, _BOND["C-N"], _ANGLE["CA-C-N"], OMEGA_TRANS)
        CAi = _place_atom(prevB, prevC, Ni, _BOND["N-CA"], _ANGLE["C-N-CA"], phi)
        Ci = _place_atom(prevC, Ni, CAi, _BOND["CA-C"], _ANGLE["N-CA-C"], psi)
        Oi = _place_atom(Ni, CAi, Ci, _BOND["C=O"], _ANGLE["CA-C-O"], 0.0)
        atoms.extend([{"name": "N", "coords": Ni}, {"name": "CA", "coords": CAi}, {"name": "C", "coords": Ci}, {"name": "O", "coords": Oi}])
        prevA, prevB, prevC = Ni, CAi, Ci
    out = []
    for i in range(N):
        idx = i * 4
        Ni, CAi, Ci = atoms[idx]["coords"], atoms[idx + 1]["coords"], atoms[idx + 2]["coords"]
        out.extend([atoms[idx], atoms[idx + 1]])
        if seq[i] != "G":
            v1 = _normalize(_np.asarray(Ni) - _np.asarray(CAi))
            v2 = _normalize(_np.asarray(Ci) - _np.asarray(CAi))
            u = v1 + v2
            if _np.linalg.norm(u) < 1e-6:
                tmp = _np.array([1.0, 0.0, 0.0])
                if abs(_np.dot(tmp, v1)) > 0.9:
                    tmp = _np.array([0.0, 1.0, 0.0])
                u = _normalize(tmp)
            else:
                u = _normalize(u)
            n = _np.cross(v1, v2)
            if _np.linalg.norm(n) < 1e-6:
                tmp = _np.array([0.0, 0.0, 1.0])
                n = tmp
            n = _normalize(n)
            dir_cb = _normalize(0.943 * u + 0.333 * n)
            CB = (_np.asarray(CAi) + 1.53 * dir_cb).tolist()
            out.append({"name": "CB", "coords": tuple(CB)})
        out.extend([atoms[idx + 2], atoms[idx + 3]])
    return out

def _detect_ss_records(ss, seq):
    helices, sheets = [], []
    n = len(ss)
    i = 0
    while i < n:
        if ss[i] == "H":
            j = i
            while j < n and ss[j] == "H":
                j += 1
            if j - i >= 4:
                helices.append((i + 1, j))
            i = j
        else:
            i += 1
    i = 0
    while i < n:
        if ss[i] == "E":
            j = i
            while j < n and ss[j] == "E":
                j += 1
            if j - i >= 3:
                sheets.append((i + 1, j))
            i = j
        else:
            i += 1
    return helices, sheets

def _write_pdb_advanced(bitstring_label, seq, ss, atoms, out_path):
    lines = []
    helices, sheets = _detect_ss_records(ss, seq)
    for idx, (start, end) in enumerate(helices, start=1):
        lines.append(f"HELIX  {idx:>3d} H{idx:>2d} ALA A{start:4d}  ALA A{end:4d}  1 {end-start+1:5d}")
    for idx, (start, end) in enumerate(sheets, start=1):
        lines.append(f"SHEET  {idx:>3d} S{idx:>2d} 1 ALA A{start:4d}  ALA A{end:4d}  0")
    serial = 1
    resi = 1
    serial_map = []
    i = 0
    while resi <= len(seq):
        aa = seq[resi - 1]
        resn = _three_letter.get(aa, "GLY")
        want = ["N", "CA", "CB", "C", "O"] if aa != "G" else ["N", "CA", "C", "O"]
        k = i
        res_serials = {"N": None, "CA": None, "CB": None, "C": None, "O": None}
        for name in want:
            found = None
            for j in range(k, min(k + 12, len(atoms))):
                if atoms[j]["name"] == name:
                    found = atoms[j]
                    k = j + 1
                    break
            if found is None:
                continue
            x, y, z = found["coords"]
            lines.append(f"ATOM  {serial:5d} {name:>3s}  {resn} A{resi:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C")
            res_serials[name] = serial
            serial += 1
        serial_map.append(res_serials)
        i = max(i + 1, k)
        resi += 1
    def add_conect(a, b):
        if a is not None and b is not None:
            lines.append(f"CONECT{a:5d}{b:5d}")
    nres = len(serial_map)
    for r in range(nres):
        s = serial_map[r]
        add_conect(s["N"], s["CA"])
        if s["CB"] is not None:
            add_conect(s["CA"], s["CB"])
        add_conect(s["CA"], s["C"])
        add_conect(s["C"], s["O"])
        if r < nres - 1:
            t = serial_map[r + 1]
            add_conect(s["C"], t["N"])
    lines.append(f"REMARK BITSTRING {bitstring_label}")
    lines.append("END")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

def _make_pdbs_from_counts_3d(filtered_counts, hyperParams, protein_sequence, output_dir):
    pdb_dir = os.path.join(output_dir, "pdb3d")
    os.makedirs(pdb_dir, exist_ok=True)
    made = []
    for bitstring_key, prob in filtered_counts.items():
        bs = bitstring_key.replace(" ", "")
        cfg_bits = bs[:hyperParams["numQubitsConfig"]]
        turns = _decode_turns_from_bitstring(cfg_bits, hyperParams["turn2qubit"])
        ss = _infer_secondary_from_turns(turns)
        atoms = _build_backbone_3d(protein_sequence, ss)
        fname = f"fold3d_{cfg_bits}.pdb"
        fpath = os.path.join(pdb_dir, fname)
        _write_pdb_advanced(cfg_bits, protein_sequence, ss, atoms, fpath)
        made.append(fpath)
    zip_path = os.path.join(output_dir, "pdb3d_bitstrings_ge_2pct.zip")
    import zipfile
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in made:
            zf.write(fp, arcname=os.path.join("pdb3d", os.path.basename(fp)))
    print(f"Generated {len(made)} robust 3D PDBs → {pdb_dir}")
    print(f"Zipped bundle → {zip_path}")
    return pdb_dir, zip_path

def generate_turn2qubit(protein_sequence):
    N = len(protein_sequence)
    if N < 2:
        raise ValueError("Protein sequence must have at least 2 beads.")
    num_turns = 2 * (N - 1)
    fixed_bits = '0100q1'
    variable_bits = 'q' * (num_turns - len(fixed_bits))
    return fixed_bits + variable_bits, fixed_bits, variable_bits

def build_mj_interactions(protein):
    N = len(protein)
    mat = np.zeros((N, N))
    np.random.seed(29507)
    MJ = np.random.rand(20, 20) * -6
    MJ = np.triu(MJ) + np.triu(MJ, 1).T
    acids = ["C", "M", "F", "I", "L", "V", "W", "Y", "A", "G",
             "T", "S", "N", "Q", "D", "E", "H", "R", "K", "P"]
    acid2idx = {acid: idx for idx, acid in enumerate(acids)}
    for i in range(N):
        for j in range(N):
            mat[i, j] = MJ[acid2idx[protein[i]], acid2idx[protein[j]]]
    return mat

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
        energies[idx] = lambda_back * sum(turns[i] == turns[i + 1] for i in range(len(turns) - 1))
        curr_interaction_qubit = hyperParams["numQubitsConfig"]
        for i in range(num_beads - 4):
            for j in range(i + 5, num_beads, 2):
                if curr_interaction_qubit >= len(bitstring):
                    break
                if bitstring[curr_interaction_qubit] == '0':
                    curr_interaction_qubit += 1
                    continue
                energies[idx] += hyperParams["interactionEnergy"][i, j]
                curr_interaction_qubit += 1
    return energies

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
    return np.dot(cvar_probs, sorted_energies[:cut_idx + 1]) / alpha

protein_sequence = input("Enter the protein sequence (max 10 amino acids, e.g., APRLRFY): ").strip().upper()
if len(protein_sequence) < 2 or len(protein_sequence) > 10:
    raise ValueError("Protein sequence must have between 2 and 10 amino acids.")

max_iterations = input("Enter maximum iterations [default 50]: ").strip()
max_iterations = int(max_iterations) if max_iterations.isdigit() and int(max_iterations) > 0 else 50

num_shots_input = input("Enter number of shots [default 1024]: ").strip()
num_shots = int(num_shots_input) if num_shots_input.isdigit() and int(num_shots_input) > 0 else 1024

output_dir = input("Enter output directory [default './results']: ").strip() or "./results"
os.makedirs(output_dir, exist_ok=True)

turn2qubit, fixed_bits, variable_bits = generate_turn2qubit(protein_sequence)
num_qubits_config = turn2qubit.count('q')
interaction_energy = build_mj_interactions(protein_sequence)
hyperParams = {
    "protein": protein_sequence,
    "turn2qubit": turn2qubit,
    "numQubitsConfig": num_qubits_config,
    "interactionEnergy": interaction_energy,
    "numShots": num_shots
}

cvar_results = []
optimal_params = None
min_energy = np.inf

for i in range(max_iterations):
    initial_parameters = np.random.uniform(-np.pi, np.pi, size=3 * (num_qubits_config + 2))
    result = minimize(lambda θ: protein_vqe_objective(θ, hyperParams), initial_parameters, method='COBYLA')
    cvar_results.append(result.fun)
    if result.fun < min_energy:
        min_energy = result.fun
        optimal_params = result.x
    pct = (i + 1) / max_iterations * 100
    print(f"Iteration {i+1}/{max_iterations} completed — {pct:.1f}%")

summary = f"""
--- Quantum Protein Folding Summary ---

Protein Sequence: {protein_sequence}
Fixed Bits:       {fixed_bits}
Variable Bits:    {variable_bits}
Shots Used:       {num_shots}
Minimum CVaR Energy: {min_energy:.5f}
"""

summary_path = os.path.join(output_dir, "output_summary.txt")
with open(summary_path, "w") as f:
    f.write(summary)
print(f"Saved summary → {summary_path}")

optimal_circuit = protein_config_ansatz(optimal_params)
circuit_path = os.path.join(output_dir, "optimal_circuit.png")
circuit_drawer(optimal_circuit.remove_final_measurements(inplace=False), output='mpl', filename=circuit_path)
print(f"Saved circuit diagram → {circuit_path}")

scatter_path = os.path.join(output_dir, "cvar_scatter.png")
plt.figure()
plt.scatter(range(1, max_iterations + 1), cvar_results, marker='o')
plt.title("CVaR Energies Across Iterations")
plt.xlabel("Iteration")
plt.ylabel("CVaR Energy")
plt.grid(True)
plt.savefig(scatter_path)
plt.close()
print(f"Saved CVaR scatter → {scatter_path}")

simulator = AerSimulator()
compiled_optimal = transpile(optimal_circuit, simulator)
job = simulator.run(compiled_optimal, shots=hyperParams["numShots"])
result = job.result()
counts = result.get_counts()
total_shots = sum(counts.values())
filtered = {k: v / total_shots for k, v in counts.items() if v / total_shots >= 0.02}
hist_path = os.path.join(output_dir, "bitstring_histogram.png")
plt.figure(figsize=(10, 5))
plt.bar(filtered.keys(), filtered.values(), edgecolor='black')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Probability")
plt.xlabel("Bitstring")
plt.title("High-Probability Bitstrings (≥2%)")
plt.tight_layout()
plt.savefig(hist_path)
plt.close()
print(f"Saved bitstring histogram → {hist_path}")

try:
    _pdb3d_dir, _zip3d_path = _make_pdbs_from_counts_3d(filtered, hyperParams, protein_sequence, output_dir)
    print(f"3D PDBs created in: {_pdb3d_dir}")
    print(f"3D PDB ZIP: {_zip3d_path}")
except Exception as _e:
    import traceback as _tb
    print("3D PDB generation failed:", _e)
    _tb.print_exc()
