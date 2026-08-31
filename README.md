[![Qiskit Ecosystem](https://qisk.it/e-ab89baa7)](https://qisk.it/e)
![Platform](https://img.shields.io/badge/Platform-Linux_%7C_MacOS_%7C_Windows-purple?style=flat&labelColor=blue)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/qupepfold?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BRIGHTGREEN&left_text=Total+Downloads)](https://pepy.tech/projects/qupepfold)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/qupepfold?period=monthly&units=NONE&left_color=GRAY&right_color=BRIGHTGREEN&left_text=Last+Month+Downloads)](https://pepy.tech/projects/qupepfold)
![License](https://img.shields.io/pypi/l/Qupepfold)
![Version](https://img.shields.io/pypi/v/Qupepfold?logo=pypi)
![Python](https://img.shields.io/pypi/pyversions/qupepfold)

---

# QuPepFold v1.3.2 — Production Release

**QuPepFold** is a quantum-classical hybrid peptide folding simulation toolkit built on [Qiskit](https://qiskit.org/). It integrates global lattice simulated annealing, quantum VQE fragment refinement (via Qiskit `SamplerV2`), and dynamic-programming stitching to predict accurate 3D backbone conformations — exporting standards-compliant PDB structures with secondary structure annotations.

---

## What's New in v1.3.2 (Production Release)

| Module | Enhancement | Impact |
|---|---|---|
| 📐 **NeRF 3D Geometry** | Fixed orthonormal frame vector `cb = (c - b)/\|c - b\|` and corrected $(\psi_{i-1}, \omega=180^\circ, \phi_i)$ bond mapping. | $C\alpha-C\alpha$ distances are strictly $3.80 \pm 0.01$ Å across all conformations; 0 steric clashes. |
| 🧬 **$C_\beta$ Stereochemistry** | Tetrahedral out-of-plane chiral placement for L-amino acids. | $C_\beta$ atoms project cleanly away from backbone with $1.53$ Å bond length and $>0.5$ Å out-of-plane separation. |
| 🧪 **Miyazawa-Jernigan Potential** | Replaced synthetic hydrophobicity with published 1996 Miyazawa-Jernigan Table 5 matrix ($20 \times 20$). | Authentic physical contact energetics with realistic hydrophobic stabilization and charge differentiation (salt bridges). |
| ⚡ **Energy Model Calibration** | Re-calibrated default $\lambda_{\text{back}} = 0.20$ (from 5.0) and unified Hamiltonian terms. | Enables hydrophobic collapse while retaining turn variety; consistent energy evaluation across SA, VQE, and DP. |
| 🧭 **Orientation-Aware Context** | Rotates fragment coordinate frames to match global incoming chain heading and detects environment clashes. | Preserves global spatial alignment during quantum fragment refinement. |
| ⚛️ **Quantum Optimization** | Active CVaR loss ($\alpha$), basis-state / biased-RY warmstart, initial Hadamard superposition, and 2-eval/iter SPSA. | True quantum variational refinement with self-avoiding post-selection filtering. |
| 🧩 **Markovian DP Stitching** | Penalizes incremental overlaps without double-counting history; prunes zero-DOF trailing fragments. | Fast, exact fragment assembly with fallback to global SA. |
| 📊 **Reporting & Metrics** | Exact circuit evaluation counts, complete candidate exports, populated summary headers. | Fully reproducible execution logs and production-ready outputs. |

---

## Architecture

```
Sequence (amino acids)
       │
       ▼
┌─────────────────────────────────────────────┐
│  Stage 1  │  Global Simulated Annealing      │  ← lattice model, SA/PT (3 restarts)
├─────────────────────────────────────────────┤
│  Stage 2  │  Fragment Generation             │  ← overlap sliding window (7-AA frags)
├─────────────────────────────────────────────┤
│  Stage 3  │  Energy Table Precomputation     │  ← 1996 MJ potential + context scoring
├─────────────────────────────────────────────┤
│  Stage 4  │  Quantum VQE Refinement (SPSA)   │  ← SamplerV2 / Aer / CVaR loss
├─────────────────────────────────────────────┤
│  Stage 5  │  Fragment Stitching (DP)         │  ← Markovian overlap-aware dynamic programming
├─────────────────────────────────────────────┤
│  Stage 5b │  Local Refinement                │  ← Full-chain lattice fine-tuning
├─────────────────────────────────────────────┤
│  Stage 6  │  3D Backbone Construction (NeRF) │  ← Cα-Cα 3.80 Å invariant, HELIX/SHEET PDB
├─────────────────────────────────────────────┤
│  Stage 7  │  Report & Metrics                │  ← JSON, CSV, plots, ZIP archive
└─────────────────────────────────────────────┘
       │
       ▼
  backbone.pdb  +  run_metrics.json  +  output_summary.txt  +  plots
```

---

## Installation

### From PyPI

```bash
pip install qupepfold
```

### From Source

```bash
git clone https://github.com/akshayuttarkarc/QuPepFold.git
cd QuPepFold/QuPepFold
pip install -e .
```

### Dependencies
- `python>=3.9`
- `qiskit>=1.0,<3`
- `qiskit-aer>=0.14`
- `numpy>=1.22`
- `matplotlib>=3.5`
- `scipy>=1.9`
- `pyyaml>=6.0`

---

## Quickstart

### Python API

```python
from qupepfold import run_fold, get_default_config

# Configure simulation
config = get_default_config(
    shots=1024,
    spsa_iterations=50,
    sa_steps=5000,
    sa_restarts=3,
    fragment_length=7,
    overlap_turns=2,
    cvar_alpha=0.25,
    seed=42,
)

# Run folding simulation
result = run_fold(
    sequence="APRLRFY",
    config=config,
    output_dir="./output_aprlrfy",
    verbose=True,
)

print(f"Final Energy: {result.energy:.4f} kcal/mol")
print(f"PDB output: {result.pdb_path}")
```

### Command Line Interface (CLI)

```bash
qupepfold --sequence APRLRFY --output-dir ./results --shots 1024 --spsa-iters 50
```

#### CLI Options:
```
Simulation Parameters:
  -s, --sequence TEXT      Amino acid sequence (e.g. APRLRFY)  [required]
  -o, --output-dir TEXT    Output directory  [default: ./qupepfold_output]
  -c, --config PATH        Path to custom YAML config file
  --backend TEXT           aer (default) | runtime
  --shots INT              Shots per circuit  [default: 2048]
  --spsa-iters INT         SPSA iterations per fragment  [default: 50]
  --sa-steps INT           SA steps per restart  [default: 5000]
  --sa-restarts INT        SA restarts  [default: 3]
  --fragment-len INT       Fragment window size (residues)  [default: 7]
  --alpha FLOAT            CVaR alpha parameter (0.0–1.0)  [default: 0.25]
  --seed INT               Random seed  [default: 42]
```

---

## Output Files

| File | Description |
|---|---|
| `backbone.pdb` | 3D backbone with HELIX/SHEET records and CONECT bond records. |
| `run_metrics.json` | Comprehensive execution metrics (timings, quantum circuit counts, shots, memory). |
| `report.json` | Machine-readable provenance and simulation metadata. |
| `output_summary.txt` | Formatted summary of energies, selected fragment candidates, and energy breakdown. |
| `fragment_candidates.csv` | All extracted VQE and table candidates across fragments. |
| `most_negative_energy_breakdown.csv` | Breakdown of contact, backbone penalty, and overlap energy terms. |
| `sa_trace.csv` | Global search restart trajectories and convergence history. |
| `mj_matrix.csv` | $N \times N$ Miyazawa-Jernigan contact interaction matrix for the target peptide. |
| `fragment_energies.png` | Per-fragment candidate energy distribution chart. |
| `energy_breakdown.png` | Contact vs backbone energy component plot. |
| `pdb3d.zip` | Bundled archive of generated PDB structures. |

---

## Secondary Structure & Turn Encodings

QuPepFold models backbone orientations on a discrete tetrahedral lattice mapped to canonical Ramachandran basins:

| Turn Code | $\phi$ | $\psi$ | $\omega$ | Region | PDB Record |
|---|---|---|---|---|---|
| `0` | $-60^\circ$ | $-45^\circ$ | $180^\circ$ | $\alpha$-helix | `HELIX` |
| `1` | $-135^\circ$ | $+135^\circ$ | $180^\circ$ | $\beta$-sheet | `SHEET` |
| `2` | $-75^\circ$ | $+145^\circ$ | $180^\circ$ | $\text{PP}_{\text{II}}$ / extended | Loop / Coil |
| `3` | $-60^\circ$ | $+140^\circ$ | $180^\circ$ | Turn / coil | Loop / Coil |

3D coordinates are reconstructed using the Natural Extension of Reference Frames (NeRF) algorithm with strict tetrahedral chiral placement of $C_\beta$ sidechains.

---

## Visualizing in Molecular Viewers

Open `backbone.pdb` in [PyMOL](https://pymol.org/), [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/), or [VMD](https://www.ks.uiuc.edu/Research/vmd/):

```python
# PyMOL commands
load backbone.pdb
show cartoon
color ss
show sticks, name CA+CB
```

---

## Verification & Testing

QuPepFold includes an automated test suite verifying geometry, quantum sampling, DP stitching, and regression baselines:

```bash
pytest -v
```

---

## Published Research

This tool is built upon peer-reviewed quantum computing and structural biology research:

1. Uttarkar, A., Niranjan, V. (2024). *Quantum synergy in peptide folding: A comparative study of CVaR-VQE and molecular dynamics simulation.* **International Journal of Biological Macromolecules**, 273, 133033. [doi:10.1016/j.ijbiomac.2024.133033](https://doi.org/10.1016/j.ijbiomac.2024.133033)

2. Uttarkar, A., Niranjan, V. (2024). *A comparative insight into peptide folding with quantum CVaR-VQE algorithm, MD simulations and structural alphabet analysis.* **Quantum Information Processing**, 23, 48. [doi:10.1007/s11128-024-04261-9](https://doi.org/10.1007/s11128-024-04261-9)

3. Uttarkar, A., Setlur, A. S., Niranjan, V. (2024). *T-Gate Enabled Fault-Tolerant Ansatz Circuit Design for VQE in Peptide Folding on Aria-1.* **Global AI Summit 2024**, IEEE. [doi:10.1109/GlobalAISummit62156.2024.10947993](https://doi.org/10.1109/GlobalAISummit62156.2024.10947993)

4. Uttarkar, A., Niranjan, V. (2025). *Quantum Enabled Protein Folding of Disordered Regions in Ubiquitin C Via Error Mitigated VQE Benchmarked on Tensor Network Simulator and Aria 1.* **IEEE Transactions on Molecular, Biological, and Multi-Scale Communications**. [doi:10.1109/TMBMC.2025.3600516](https://doi.org/10.1109/TMBMC.2025.3600516)

---

## Authors

- **Akshay Uttarkar** — [akshayuttarkar@gmail.com](mailto:akshayuttarkar@gmail.com)
- **Vinay Kumar**
- **Vidya Niranjan**

---

## License

MIT License — see [LICENSE](LICENSE) for details.
