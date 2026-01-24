import os
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from scipy.optimize import minimize

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.visualization import circuit_drawer
    from qiskit.quantum_info import Statevector
except Exception as e:
    raise SystemExit("This script needs qiskit and qiskit-aer.\nTry: pip install qiskit qiskit-aer\n" + str(e))

# ----------------------- User knobs (defaults; configurable via prompts) ----------------
CVAR_ALPHA_DEFAULT   = 0.025
MAX_EVALS_PER_TRY    = 80        # per-iteration budget for Nelder–Mead (we do many tries)
EXPORT_P_MIN_DEFAULT = 0.02
RNG_SEED             = 29507

# Geometry (Å; degrees→radians)
_BOND = {"C-N": 1.329, "N-CA": 1.458, "CA-C": 1.525, "C=O": 1.229}
_ANGLE = {"C-N-CA": math.radians(121.7),
          "N-CA-C": math.radians(110.4),
          "CA-C-N": math.radians(116.2),
          "CA-C-O": math.radians(120.8)}
OMEGA_TRANS = math.radians(180.0)

AA3 = {"A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","E":"GLU","Q":"GLN","G":"GLY","H":"HIS",
       "I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE","P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL"}

# Turn → (φ,ψ) to ensure every bitstring gives distinct 3D
TURN_DIHEDRAL = {
    0: (math.radians(-60),  math.radians(-45)),   # helix-like
    1: (math.radians(-135), math.radians(135)),   # beta-like
    2: (math.radians(-75),  math.radians(145)),   # PPII-like
    3: (math.radians(-60),  math.radians(140)),   # extended/coil-ish
}

# ======================================================================================
# Terminal prompts (UX parity)
# ======================================================================================
def ask(prompt: str, default: str = "") -> str:
    s = input(prompt).strip()
    return s if s else default

def read_cli_inputs():
    protein_sequence = ask("Enter the protein sequence (max 10 amino acids, e.g., APRLRFY): ").upper()
    if len(protein_sequence) < 2 or len(protein_sequence) > 10 or any(c not in AA3 for c in protein_sequence):
        raise ValueError("Protein sequence must be 2–10 AAs using standard one-letter codes.")

    max_iterations_s = ask("Enter maximum iterations [default 50]: ", "50")
    try:
        max_iterations = int(max_iterations_s)
        if max_iterations <= 0: raise ValueError
    except Exception:
        max_iterations = 50

    num_shots_s = ask("Enter number of shots [default 1024]: ", "1024")
    try:
        num_shots = int(num_shots_s)
        if num_shots <= 0: raise ValueError
    except Exception:
        num_shots = 1024

    export_p_s = ask("Min probability to export PDBs [default 0.02]: ", f"{EXPORT_P_MIN_DEFAULT}")
    try:
        export_p = float(export_p_s)
        if export_p <= 0 or export_p >= 1: raise ValueError
    except Exception:
        export_p = EXPORT_P_MIN_DEFAULT

    cvar_alpha_s = ask("CVaR alpha (tail mass) [default 0.025]: ", f"{CVAR_ALPHA_DEFAULT}")
    try:
        cvar_alpha = float(cvar_alpha_s)
        if cvar_alpha <= 0 or cvar_alpha >= 1: raise ValueError
    except Exception:
        cvar_alpha = CVAR_ALPHA_DEFAULT

    output_dir = ask("Enter output directory [default './results']: ", "./results")
    os.makedirs(output_dir, exist_ok=True)

    return protein_sequence, max_iterations, num_shots, export_p, cvar_alpha, output_dir

# ======================================================================================
# Config mapping & interactions
# ======================================================================================
def generate_turn2qubit(sequence: str) -> Tuple[str, str, str]:
    """2*(N-1) bits template: small fixed prefix (contains a 'q') + 'q' suffix."""
    N = len(sequence)
    total_turn_bits = 2 * (N - 1)
    fixed_prefix = "0100q1"
    if len(fixed_prefix) > total_turn_bits:
        fixed_prefix = fixed_prefix[:total_turn_bits]
    variable_bits = 'q' * max(0, total_turn_bits - len(fixed_prefix))
    return fixed_prefix + variable_bits, fixed_prefix, variable_bits

def count_interaction_qubits(sequence: str) -> int:
    """1-NN interaction controls: for i in [0..N-5], j in [i+5..N-1] step 2."""
    N = len(sequence)
    cnt = 0
    for i in range(0, N - 4):
        for j in range(i + 5, N, 2):
            cnt += 1
    return cnt

def build_mj_interactions(protein: str) -> np.ndarray:
    """Random symmetric MJ-like matrix mapped to residues (seeded for reproducibility)."""
    rng = np.random.default_rng(RNG_SEED)
    MJ = rng.random((20, 20)) * -6.0
    MJ = np.triu(MJ) + np.triu(MJ, 1).T
    acids = ["C","M","F","I","L","V","W","Y","A","G","T","S","N","Q","D","E","H","R","K","P"]
    acid2idx = {a:i for i,a in enumerate(acids)}
    N = len(protein)
    mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            mat[i, j] = float(MJ[acid2idx[protein[i]], acid2idx[protein[j]]])
    return mat

def fill_config_bits(cfg_bits: str, turn2qubit: str) -> str:
    cfg = list(turn2qubit)
    qpos = [i for i, ch in enumerate(cfg) if ch == 'q']
    if len(cfg_bits) != len(qpos):
        raise ValueError(f"cfg_bits length {len(cfg_bits)} != # of 'q' in template {len(qpos)}")
    for i, pos in enumerate(qpos):
        cfg[pos] = cfg_bits[i]
    expanded = ''.join(cfg)
    assert 'q' not in expanded, "Unfilled 'q' in expanded config."
    return expanded

# ======================================================================================
# Energy calculations
# ======================================================================================
def exact_hamiltonian(bitstrings: List[str], hyper: Dict) -> List[float]:
    """H = H_gc + H_in(1-NN) with penalties lamDis=720, lamLoc=20, lamBack=50."""
    if isinstance(bitstrings, str):
        bitstrings = [bitstrings]
    lamDis, lamLoc, lamBack = 720.0, 20.0, 50.0
    N = len(hyper["protein"])
    energies = []

    for bs in bitstrings:
        cfg_bits = bs[:hyper["numQubitsConfig"]]
        config = fill_config_bits(cfg_bits, hyper["turn2qubit"])
        turns = [int(config[k:k+2], 2) for k in range(0, len(config), 2)]  # length N-1

        # H_gc: penalize adjacent equal turns
        E = lamBack * sum(1 for a, b in zip(turns[:-1], turns[1:]) if a == b)

        # H_in: 1-NN; interaction bits follow cfg segment
        q = hyper["numQubitsConfig"]
        for i in range(0, N - 4):
            for j in range(i + 5, N, 2):
                if q >= len(bs):
                    continue
                ctrl = bs[q]; q += 1
                if ctrl == '0':
                    continue

                E += float(hyper["interactionEnergy"][i, j])

                def delta_vec(a: int, b: int) -> np.ndarray:
                    seg = turns[a:b]
                    vec = np.zeros(4)
                    for k in range(4):
                        mask = np.array([1 if t == k else 0 for t in seg], float)
                        if mask.size:
                            vec[k] = np.sum(((-1) ** np.arange(mask.size)) * mask)
                    return vec

                dij = np.linalg.norm(delta_vec(i, j)) ** 2
                dir_ = np.linalg.norm(delta_vec(i, j - 1)) ** 2 if j - 1 > i else 0.0
                dmj = np.linalg.norm(delta_vec(i + 1, j)) ** 2 if j > i + 1 else 0.0

                E += lamDis * (dij - 1.0)
                E += lamLoc * (2.0 - dir_)
                E += lamLoc * (2.0 - dmj)

                if i - 1 >= 0:
                    dmj2 = np.linalg.norm(delta_vec(i - 1, j)) ** 2
                    E += lamLoc * (2.0 - dmj2)
                if j + 1 <= N - 1:
                    dir2 = np.linalg.norm(delta_vec(i, j)) ** 2
                    E += lamLoc * (2.0 - dir2)

        energies.append(float(E))
    return energies

# ======================================================================================
# Scalable ansatz (2×RY + CX loop) and prob extraction
# ======================================================================================
def qubit_layout(hyper: Dict):
    """[0..cfg-1]=config, [cfg..cfg+int-1]=interaction, [last]=ancilla; measure cfg+int."""
    num_cfg = int(hyper["numQubitsConfig"])
    num_int = int(hyper["numQubitsInteraction"])
    cfg_idx = list(range(0, num_cfg))
    int_idx = list(range(num_cfg, num_cfg + num_int))
    anc_idx = [num_cfg + num_int]
    all_idx = cfg_idx + int_idx + anc_idx
    measured_idx = cfg_idx + int_idx
    return cfg_idx, int_idx, anc_idx, all_idx, measured_idx

def target_lists_for_ry_layers(hyper: Dict):
    cfg_idx, int_idx, anc_idx, _, _ = qubit_layout(hyper)
    layer1 = cfg_idx + anc_idx + int_idx
    layer2 = cfg_idx + anc_idx + (int_idx[:-1] if int_idx else [])
    return layer1, layer2

def num_angles_for_ansatz(hyper: Dict) -> int:
    L1, L2 = target_lists_for_ry_layers(hyper)
    return len(L1) + len(L2)

def build_scalable_ansatz(parameters: np.ndarray, hyper: Dict, measure: bool = False) -> QuantumCircuit:
    """Two RY layers + CX loop (scalable). If measure=True, measure all for sampling."""
    cfg_idx, int_idx, anc_idx, all_idx, _ = qubit_layout(hyper)
    L1, L2 = target_lists_for_ry_layers(hyper)
    theta = np.asarray(parameters, float).ravel()
    need = len(L1) + len(L2)
    assert theta.size == need, f"Parameter length {theta.size} != required {need}"

    qc = QuantumCircuit(len(all_idx), 0 if not measure else len(all_idx))

    # Hadamards
    for q in all_idx:
        qc.h(q)

    # First RY
    off = 0
    for t in L1:
        qc.ry(float(theta[off]), t); off += 1

    # CX loop: cfg chain; cfg[-1]→anc; anc→int[-1]→…→int[0]→cfg[0]
    for a, b in zip(cfg_idx[:-1], cfg_idx[1:]):
        qc.cx(a, b)
    if cfg_idx and anc_idx:
        qc.cx(cfg_idx[-1], anc_idx[0])
    if anc_idx and int_idx:
        qc.cx(anc_idx[0], int_idx[-1])
        for a, b in zip(int_idx[::-1][:-1], int_idx[::-1][1:]):
            qc.cx(a, b)
        qc.cx(int_idx[0], cfg_idx[0])

    # Second RY
    for t in L2:
        qc.ry(float(theta[off]), t); off += 1

    if measure:
        qc.measure(range(qc.num_qubits), range(qc.num_clbits))
    return qc

def statevector_fold_probs(qc: QuantumCircuit, hyper: Dict) -> Dict[str, float]:
    """Exact probabilities for measured qubits (cfg+int), ancilla excluded."""
    sv = Statevector.from_instruction(qc)
    probs = np.abs(sv.data) ** 2
    Q = qc.num_qubits
    _, _, _, _, measured_idx = qubit_layout(hyper)

    out: Dict[str, float] = {}
    for i, p in enumerate(probs):
        b_le = format(i, f"0{Q}b")[::-1]   # little-endian -> q_{Q-1}..q_0
        fold = ''.join(b_le[j] for j in measured_idx)[::-1]
        out[fold] = out.get(fold, 0.0) + float(p)
    return out

# ======================================================================================
# CVaR objective + multi-start optimizer (console updates)
# ======================================================================================
def cvar_objective(parameters: np.ndarray, hyper: Dict, alpha: float) -> float:
    qc = build_scalable_ansatz(parameters, hyper, measure=False)
    probs = statevector_fold_probs(qc, hyper)
    states = list(probs.keys())
    pvals  = np.array([probs[s] for s in states], float)
    energies = np.array(exact_hamiltonian(states, hyper), float)

    order = np.argsort(energies)
    energies = energies[order]; pvals = pvals[order]
    c = np.cumsum(pvals)
    k = int(np.count_nonzero(c < alpha))
    tail = pvals[:k].tolist(); tail.append(alpha - sum(tail))
    return float(np.dot(tail, energies[:k+1]) / alpha)

def optimize_cvar_multistart(hyper: Dict, max_iterations: int, alpha: float):
    """Run many small-budget optimizations and log CVaR; print progress per try."""
    rng = np.random.default_rng(RNG_SEED)
    D = num_angles_for_ansatz(hyper)
    best_x, best_f = None, float("inf")
    trace = []
    for i in range(max_iterations):
        x0 = rng.uniform(-np.pi, np.pi, size=D)
        f = lambda x: cvar_objective(x, hyper, alpha)
        res = minimize(f, x0, method="Nelder-Mead",
                       options={"maxfev": MAX_EVALS_PER_TRY, "xatol":1e-3, "fatol":1e-3})
        trace.append(res.fun)
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x
        pct = (i+1)*100.0/max_iterations
        print(f"Iteration {i+1}/{max_iterations} completed — {pct:.1f}%")
    return best_x, best_f, trace

# ======================================================================================
# 3D builder + PDB writer with CONECT
# ======================================================================================
def _normalize(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v*0.0 if n < 1e-8 else v/n

def _orthonormal_frame(a, b, c):
    cb = _normalize(np.asarray(b) - np.asarray(c))
    t  = np.asarray(b) - np.asarray(a)
    n  = np.cross(t, cb)
    if np.linalg.norm(n) < 1e-8:
        tmp = np.array([1.0,0.0,0.0])
        if abs(np.dot(tmp, cb)) > 0.9: tmp = np.array([0.0,1.0,0.0])
        n = np.cross(tmp, cb)
    n = _normalize(n); m = _normalize(np.cross(n, cb))
    return m, n, cb

def _place_atom(pA,pB,pC,bond_len,angle_rad,dihedral_rad):
    m,n,cb = _orthonormal_frame(pA,pB,pC)
    x = -bond_len*math.cos(angle_rad)
    y =  bond_len*math.cos(dihedral_rad)*math.sin(angle_rad)
    z =  bond_len*math.sin(dihedral_rad)*math.sin(angle_rad)
    C = np.asarray(pC,float); D = C + x*cb + y*m + z*n
    return tuple(D.tolist())

def _seed_first_residue():
    N1=(0.0,0.0,0.0)
    CA1=(_BOND["N-CA"],0.0,0.0)
    ang=_ANGLE["N-CA-C"]; vx,vy=-math.cos(ang), math.sin(ang)
    C1=(CA1[0]+_BOND["CA-C"]*vx, CA1[1]+_BOND["CA-C"]*vy, 0.0)
    O1=_place_atom(N1,CA1,C1,_BOND["C=O"],_ANGLE["CA-C-O"],0.0)
    return N1,CA1,C1,O1

def turns_from_cfg_bits(cfg_bits: str, turn2qubit: str) -> List[int]:
    cfg = fill_config_bits(cfg_bits, turn2qubit)
    return [int(cfg[i:i+2],2) for i in range(0,len(cfg),2)]

def dihedrals_from_turns(turns: List[int], N: int) -> Tuple[List[float], List[float]]:
    phis=[None]*N; psis=[None]*N
    for i in range(N):
        t_prev = turns[i-1] if i-1>=0 and i-1<len(turns) else 1
        t_next = turns[i]   if i  <len(turns) else 2
        phis[i]=TURN_DIHEDRAL[t_prev][0]
        psis[i]=TURN_DIHEDRAL[t_next][1]
    return phis,psis

def build_backbone_3d(seq: str, phis: List[float], psis: List[float]) -> List[Dict]:
    N=len(seq)
    N1,CA1,C1,O1=_seed_first_residue()
    atoms=[{"name":"N","coords":N1},{"name":"CA","coords":CA1},{"name":"C","coords":C1},{"name":"O","coords":O1}]
    prevA,prevB,prevC=N1,CA1,C1
    for i in range(1,N):
        phi,psi=phis[i],psis[i]
        Ni = _place_atom(prevA,prevB,prevC,_BOND["C-N"], _ANGLE["CA-C-N"], OMEGA_TRANS)
        CAi= _place_atom(prevB,prevC,Ni,   _BOND["N-CA"], _ANGLE["C-N-CA"], phi)
        Ci = _place_atom(prevC,Ni,CAi,     _BOND["CA-C"], _ANGLE["N-CA-C"], psi)
        Oi = _place_atom(Ni,CAi,Ci,        _BOND["C=O"],  _ANGLE["CA-C-O"], 0.0)
        atoms.extend([{"name":"N","coords":Ni},{"name":"CA","coords":CAi},{"name":"C","coords":Ci},{"name":"O","coords":Oi}])
        prevA,prevB,prevC=Ni,CAi,Ci

    out=[]
    for i in range(N):
        idx=i*4
        Ni,CAi,Ci = atoms[idx]["coords"], atoms[idx+1]["coords"], atoms[idx+2]["coords"]
        out.extend([atoms[idx], atoms[idx+1]])
        if seq[i]!="G":
            v1=_normalize(np.asarray(Ni)-np.asarray(CAi))
            v2=_normalize(np.asarray(Ci)-np.asarray(CAi))
            u=v1+v2
            if np.linalg.norm(u)<1e-8:
                tmp=np.array([1.0,0.0,0.0]); 
                if abs(np.dot(tmp,v1))>0.9: tmp=np.array([0.0,1.0,0.0])
                u=_normalize(tmp)
            else:
                u=_normalize(u)
            n=np.cross(v1,v2); 
            if np.linalg.norm(n)<1e-8: n=np.array([0.0,0.0,1.0])
            n=_normalize(n)
            dir_cb=_normalize(0.943*u+0.333*n)
            CB=(np.asarray(CAi)+1.53*dir_cb).tolist()
            out.append({"name":"CB","coords":tuple(CB)})
        out.extend([atoms[idx+2], atoms[idx+3]])
    return out

def write_pdb_with_conect(bitstring_label: str, seq: str, atoms: List[Dict], out_path: str):
    lines=[]
    serial=1; resi=1; serial_map=[]; i=0
    while resi<=len(seq):
        aa=seq[resi-1]; resn=AA3.get(aa,"GLY")
        want=["N","CA","CB","C","O"] if aa!="G" else ["N","CA","C","O"]
        k=i; res_serials={"N":None,"CA":None,"CB":None,"C":None,"O":None}
        for name in want:
            found=None
            for j in range(k, min(k+12, len(atoms))):
                if atoms[j]["name"]==name:
                    found=atoms[j]; k=j+1; break
            if found is None: continue
            x,y,z=found["coords"]
            lines.append(f"ATOM  {serial:5d} {name:>3s}  {resn} A{resi:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C")
            res_serials[name]=serial; serial+=1
        serial_map.append(res_serials)
        i=max(i+1,k); resi+=1

    def add_conect(a,b):
        if a is not None and b is not None: lines.append(f"CONECT{a:5d}{b:5d}")

    nres=len(serial_map)
    for r in range(nres):
        s=serial_map[r]
        add_conect(s["N"], s["CA"])
        if s["CB"] is not None: add_conect(s["CA"], s["CB"])
        add_conect(s["CA"], s["C"])
        add_conect(s["C"], s["O"])
        if r<nres-1:
            t=serial_map[r+1]
            add_conect(s["C"], t["N"])
    lines.append(f"REMARK BITSTRING {bitstring_label}")
    lines.append("END")
    with open(out_path,"w") as f: f.write("\n".join(lines))

def make_pdbs_from_probs(filtered_probs: Dict[str,float], hyper: Dict, seq: str, out_dir: str):
    pdb_dir = os.path.join(out_dir, "pdb3d"); os.makedirs(pdb_dir, exist_ok=True)
    made=[]
    for bs, p in filtered_probs.items():
        cfg_bits = bs[:hyper["numQubitsConfig"]]
        turns   = turns_from_cfg_bits(cfg_bits, hyper["turn2qubit"])
        phis, psis = dihedrals_from_turns(turns, len(seq))
        atoms   = build_backbone_3d(seq, phis, psis)
        fname = f"fold3d_{cfg_bits}.pdb"
        fpath = os.path.join(pdb_dir, fname)
        write_pdb_with_conect(cfg_bits, seq, atoms, fpath)
        made.append(fpath)
    zip_path = os.path.join(out_dir, "pdb3d_bitstrings_ge_2pct.zip")
    if made:
        import zipfile
        with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as zf:
            for fp in made:
                zf.write(fp, arcname=os.path.join("pdb3d", os.path.basename(fp)))
        print(f"Generated {len(made)} robust 3D PDBs → {pdb_dir}")
        print(f"Zipped bundle → {zip_path}")
    else:
        print("No PDBs generated (no bitstrings above threshold).")
    return pdb_dir, zip_path if made else (pdb_dir, None)

# ======================================================================================
# Energy breakdown utilities (BEST FOLDED PEPTIDE BITSTRING)
# ======================================================================================
def energy_breakdown_components(bitstring: str, hyper: Dict):
    """
    Return (components_dict, total_energy) for a single bitstring.
    Components: backbone (gc), mj, distance (dis), locality (loc).
    """
    lamDis, lamLoc, lamBack = 720.0, 20.0, 50.0
    N = len(hyper["protein"])

    cfg_bits = bitstring[:hyper["numQubitsConfig"]]
    config = fill_config_bits(cfg_bits, hyper["turn2qubit"])
    turns = [int(config[k:k+2], 2) for k in range(0, len(config), 2)]  # length N-1

    comp = {"backbone": 0.0, "mj": 0.0, "distance": 0.0, "locality": 0.0}

    # backbone term
    comp["backbone"] = lamBack * sum(1 for a, b in zip(turns[:-1], turns[1:]) if a == b)

    # helpers for locality/distance vectors
    def delta_vec(a: int, b: int) -> np.ndarray:
        seg = turns[a:b]
        vec = np.zeros(4)
        for k in range(4):
            mask = np.array([1 if t == k else 0 for t in seg], float)
            if mask.size:
                vec[k] = np.sum(((-1) ** np.arange(mask.size)) * mask)
        return vec

    # interaction bits follow config segment
    q = hyper["numQubitsConfig"]
    for i in range(0, N - 4):
        for j in range(i + 5, N, 2):
            if q >= len(bitstring):
                continue
            ctrl = bitstring[q]; q += 1
            if ctrl == '0':
                continue

            # MJ contact
            comp["mj"] += float(hyper["interactionEnergy"][i, j])

            # Distance / locality penalties
            dij = np.linalg.norm(delta_vec(i, j)) ** 2
            dir_ = np.linalg.norm(delta_vec(i, j - 1)) ** 2 if j - 1 > i else 0.0
            dmj = np.linalg.norm(delta_vec(i + 1, j)) ** 2 if j > i + 1 else 0.0

            comp["distance"] += lamDis * (dij - 1.0)
            comp["locality"] += lamLoc * (2.0 - dir_)
            comp["locality"] += lamLoc * (2.0 - dmj)

            if i - 1 >= 0:
                dmj2 = np.linalg.norm(delta_vec(i - 1, j)) ** 2
                comp["locality"] += lamLoc * (2.0 - dmj2)
            if j + 1 <= N - 1:
                dir2 = np.linalg.norm(delta_vec(i, j)) ** 2
                comp["locality"] += lamLoc * (2.0 - dir2)

    total = comp["backbone"] + comp["mj"] + comp["distance"] + comp["locality"]
    return comp, float(total)

def plot_energy_breakdown_for_most_negative(probs: Dict[str, float], hyper: Dict, out_dir: str):
    """
    Identify the most negative-energy bitstring among keys(probs),
    plot a component breakdown, and save PNG + CSV.
    """
    states = list(probs.keys())
    energies = exact_hamiltonian(states, hyper)
    idx_min = int(np.argmin(energies))
    s_min   = states[idx_min]
    p_min   = float(probs[s_min])
    comp, total = energy_breakdown_components(s_min, hyper)

    # Bar chart
    labels = ["Backbone", "MJ contacts", "Distance", "Locality", "Total"]
    values = [comp["backbone"], comp["mj"], comp["distance"], comp["locality"], total]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, edgecolor="black")
    plt.ylabel("Energy")
    plt.title("Energy Breakdown — Most Negative Bitstring")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "most_negative_energy_breakdown.png")
    plt.savefig(out_path); plt.close()

    # Console summary
    cfg_bits = s_min[:hyper["numQubitsConfig"]]
    print("\n--- Best peptide folding Bitstring ---")
    print(f"Bitstring (cfg+int): {s_min}")
    print(f"Config bits only   : {cfg_bits}")
    print(f"Probability        : {p_min:.6f}")
    print(f"Energy (total)     : {total:.6f}")
    print(f"  - Backbone       : {comp['backbone']:.6f}")
    print(f"  - MJ contacts    : {comp['mj']:.6f}")
    print(f"  - Distance       : {comp['distance']:.6f}")
    print(f"  - Locality       : {comp['locality']:.6f}")
    print(f"Saved energy breakdown plot → {out_path}")

    # CSV
    csv_path = os.path.join(out_dir, "most_negative_energy_breakdown.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bitstring","probability","total","backbone","mj","distance","locality"])
        w.writerow([s_min, p_min, total, comp["backbone"], comp["mj"], comp["distance"], comp["locality"]])
    print(f"Saved component CSV → {csv_path}")

# ======================================================================================
# Main
# ======================================================================================
def main():
    # === Prompts/inputs (UX parity) ===
    seq, max_iterations, num_shots, export_p, cvar_alpha, out_dir = read_cli_inputs()

    # === Mapping & hyper ===
    turn2qubit, fixed_bits, variable_bits = generate_turn2qubit(seq)
    num_q_cfg = turn2qubit.count('q')                 # IMPORTANT: count all 'q' in template
    num_q_int = count_interaction_qubits(seq)
    hyper = {
        "protein": seq,
        "turn2qubit": turn2qubit,
        "numQubitsConfig": num_q_cfg,
        "numQubitsInteraction": num_q_int,
        "interactionEnergy": build_mj_interactions(seq),
        "numShots": int(num_shots),
    }

    total_qubits = num_q_cfg + num_q_int + 1
    print(f"[Init] sequence={seq} (N={len(seq)})")
    print(f"       turn2qubit='{turn2qubit}'  |  fixed='{fixed_bits}'  |  variable='{variable_bits}'")
    print(f"       numQubitsConfig={num_q_cfg}  numQubitsInteraction={num_q_int}   total(qubits)={total_qubits} (incl. ancilla)")
    os.makedirs(out_dir, exist_ok=True)

    # === Multi-start CVaR optimization with per-iteration prints ===
    print(f"[CVaR-VQE] alpha={cvar_alpha}, tries={max_iterations}, per-try budget={MAX_EVALS_PER_TRY}")
    best_x, best_cvar, cvar_trace = optimize_cvar_multistart(hyper, max_iterations, cvar_alpha)
    print(f"Minimum CVaR Energy: {best_cvar:.6f}")

    # === Save text summary ===
    summary = f"""
--- Quantum Protein Folding Summary ---

Protein Sequence: {seq}
Fixed Bits:       {fixed_bits}
Variable Bits:    {variable_bits}
Shots Used:       {num_shots}
CVaR alpha:       {cvar_alpha}
Tries:            {max_iterations} (each Nelder–Mead maxfev={MAX_EVALS_PER_TRY})
Minimum CVaR Energy: {best_cvar:.6f}
"""
    summary_path = os.path.join(out_dir, "output_summary.txt")
    with open(summary_path, "w") as f: f.write(summary.strip() + "\n")
    print(f"Saved summary → {summary_path}")

    # === Draw ansatz circuit (no measurements) ===
    try:
        qc_best = build_scalable_ansatz(best_x, hyper, measure=False)
        circuit_path = os.path.join(out_dir, "optimal_circuit.png")
        circuit_drawer(qc_best, output='mpl', filename=circuit_path)
        print(f"Saved circuit diagram → {circuit_path}")
    except Exception as e:
        print("Circuit drawing failed:", e)

    # === Final optimized distribution (statevector) ===
    probs = statevector_fold_probs(build_scalable_ansatz(best_x, hyper, measure=False), hyper)
    states = list(probs.keys())
    pvals  = np.array([probs[s] for s in states], float)

    # === (Optional) Aer sampling just to mirror UX feel; not used for decisions ===
    try:
        qc_meas = build_scalable_ansatz(best_x, hyper, measure=True)
        sim = AerSimulator()
        _ = sim.run(transpile(qc_meas, sim), shots=num_shots).result().get_counts()
    except Exception as e:
        print("Aer sampling skipped (simulator error):", e)

    # === Histogram of ≥ export_p using exact probs ===
    filtered = {k: float(v) for k, v in probs.items() if v >= export_p}
    hist_path = os.path.join(out_dir, "bitstring_histogram.png")
    plt.figure(figsize=(10,5))
    if filtered:
        xs = list(filtered.keys()); ys = [filtered[k] for k in xs]
        plt.bar(xs, ys, edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Probability")
    plt.xlabel("Bitstring (config+interaction)")
    plt.title(f"High-Probability Bitstrings (≥ {export_p:.2%})")
    plt.tight_layout()
    plt.savefig(hist_path); plt.close()
    print(f"Saved bitstring histogram → {hist_path}")

    # === CSV: all states with energy and export flag (statevector distribution) ===
    csv_all = os.path.join(out_dir, "bitstring_summary.csv")
    energies_all = exact_hamiltonian(states, hyper)
    with open(csv_all, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bitstring","cfg_bits","probability","energy","exported_PDB3D"])
        for s, p, e in zip(states, pvals, energies_all):
            cfg_bits = s[:hyper["numQubitsConfig"]]
            exported = int(filtered.get(s, 0.0) >= export_p)
            w.writerow([s, cfg_bits, float(p), float(e), exported])
    print(f"Saved CSV → {csv_all}")

    # === PDB3D export for ≥ threshold (with CONECT) ===
    try:
        pdb_dir, zip_path = make_pdbs_from_probs(filtered, hyper, seq, out_dir)
        print(f"3D PDBs created in: {pdb_dir}")
        if zip_path: print(f"3D PDB ZIP: {zip_path}")
    except Exception as e:
        import traceback as _tb
        print("3D PDB generation failed:", e)
        _tb.print_exc()

    # === CVaR-optimized distribution CSV (same as bitstring_summary_cvar.csv) ===
    csv_cvar = os.path.join(out_dir, "bitstring_summary_cvar.csv")
    with open(csv_cvar, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["bitstring","probability","energy"])
        eopt = exact_hamiltonian(states, hyper)
        for s, p, e in zip(states, pvals, eopt):
            w.writerow([s, float(p), float(e)])
    print(f"Saved CVaR distribution CSV → {csv_cvar}")

    # === Energy breakdown plot for the most negative bitstring ===
    plot_energy_breakdown_for_most_negative(probs, hyper, out_dir)

    # === CVaR scatter plot across iterations (added back) ===
    scatter_path = os.path.join(out_dir, "cvar_scatter.png")
    plt.figure()
    plt.scatter(range(1, len(cvar_trace)+1), cvar_trace, marker='o')
    plt.title("CVaR Energies Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("CVaR Energy")
    plt.grid(True)
    plt.savefig(scatter_path); plt.close()
    print(f"Saved CVaR scatter → {scatter_path}")

    print("Done.")

if __name__ == "__main__":
    main()
