[![Qiskit Ecosystem](https://qisk.it/e-ab89baa7)](https://qisk.it/e)
![Platform](https://img.shields.io/badge/Platform-Linux_%7C_MacOS_%7C_Windows-purple?style=flat&labelColor=blue)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/qupepfold?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BRIGHTGREEN&left_text=Total+Downloads)](https://pepy.tech/projects/qupepfold)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/qupepfold?period=monthly&units=NONE&left_color=GRAY&right_color=BRIGHTGREEN&left_text=Last+Month+Downloads)](https://pepy.tech/projects/qupepfold)
![License](https://img.shields.io/pypi/l/Qupepfold)
![Version](https://img.shields.io/pypi/v/Qupepfold?logo=pypi)
![Python](https://img.shields.io/pypi/pyversions/qupepfold)

---
**★ Primary citation for QuPepFold (v0.8.0):**

> Uttarkar A, Niranjan V, Saxena A, Kumar V (2026).  
> QuPepFold: A python package for hybrid quantum-classical protein folding simulations with CVaR-optimized VQE.  
> *PLoS One* **21**(2): e0342012.  
> https://doi.org/10.1371/journal.pone.0342012

---

# QuPepFold v1.3.1

**QuPepFold** is a quantum-classical hybrid peptide folding toolkit built on [Qiskit](https://qiskit.org/). It combines simulated annealing on lattice models, quantum VQE fragment refinement (via `SamplerV2`), and dynamic-programming stitching to predict 3D backbone conformations — then exports standards-compliant PDB files with full secondary structure annotation.

---

## What's New in v1.3.1 (QA-VQE)

| # | Change | Details |
|---|--------|---------|
| 1 | **Hybrid 7-stage pipeline** | SA global search → fragment generation → energy tables → VQE refinement (SPSA) → DP stitching → 3D backbone → report |
| 2 | **VQE via `SamplerV2`** | Uses Qiskit's modern `SamplerV2` API with SPSA optimiser and CVaR loss |
| 3 | **Overlap-aware DP stitching** | Dynamic-programming stitching with 2-turn overlap matching; falls back to best global SA result |
| 4 | **Correct secondary structure in PDB** | HELIX/SHEET records derived directly from the optimizer's turn codes — renders correctly in PyMOL, ChimeraX, VMD |

---

## Architecture

```
Sequence (amino acids)
       │
       ▼
┌─────────────────────────────────────────────┐
│  Stage 1  │  Global Simulated Annealing      │  ← lattice model, SA/PT
│           │  (3 restarts, best energy kept)  │
├─────────────────────────────────────────────┤
│  Stage 2  │  Fragment Generation             │  ← sliding window, 7-aa frags
├─────────────────────────────────────────────┤
│  Stage 3  │  Energy Table Precomputation     │  ← MJ potential, 1024 states
├─────────────────────────────────────────────┤
│  Stage 4  │  Quantum VQE Refinement (SPSA)   │  ← SamplerV2 / Aer / Runtime
├─────────────────────────────────────────────┤
│  Stage 5  │  Fragment Stitching (DP)         │  ← overlap-aware, 2-turn match
├─────────────────────────────────────────────┤
│  Stage 6  │  3D Backbone Construction        │  ← PDB with HELIX/SHEET records
├─────────────────────────────────────────────┤
│  Stage 7  │  Report & Metrics                │  ← JSON, CSV, plots, ZIP
└─────────────────────────────────────────────┘
       │
       ▼
  backbone.pdb  +  run_metrics.json  +  fragment_energies.png
```

---

## Installation

### From PyPI (stable)
```bash
pip install qupepfold
```

### From source (this branch)
```bash
git clone https://github.com/akshayuttarkarc/QuPepFold.git
cd QuPepFold
git checkout QA-VQE
cd QuPepFold                    # the Python package lives here
pip install -e .                # editable install
```

### Requirements
| Package | Version |
|---------|---------|
| Python | ≥ 3.9 |
| qiskit | ≥ 1.0 |
| qiskit-aer | ≥ 0.14 |
| numpy | ≥ 1.22 |
| matplotlib | ≥ 3.5 |
| scipy | ≥ 1.9 |
| pyyaml | ≥ 6.0 |

**Optional — IBM Quantum Runtime:**
```bash
pip install qupepfold[ibm]       # adds qiskit-ibm-runtime ≥ 0.20
```

---

## Quick Start

### CLI
```bash
# Local Aer simulator (default)
qupepfold fold \
  --seq ACDEFGHIKLMNPQRSTVWY \
  --backend aer \
  --shots 2048 \
  --out ./results

# IBM Quantum Runtime
qupepfold fold \
  --seq ACDEFGHIKLMNPQRSTVWY \
  --backend runtime \
  --ibm-token YOUR_TOKEN \
  --ibm-backend ibm_sherbrooke \
  --shots 4096 \
  --out ./results
```

### Python API
```python
from qupepfold import run_fold, get_default_config

config = get_default_config()
config.backend  = "aer"
config.shots    = 2048
config.sa_steps = 5000

result = run_fold(
    sequence="ACDEFGHIKLMNPQRSTVWY",
    config=config,
    output_dir="./results",
    verbose=True,
)

print(f"Global energy : {result.global_energy:.4f}")
print(f"Stitched energy: {result.stitched_energy:.4f}")
print(f"PDB            : {result.pdb_path}")
```

---

## CLI Reference

```
qupepfold fold [OPTIONS]

Required:
  --seq TEXT          Amino acid sequence (single-letter, 4–100 residues)
  --out PATH          Output directory (created if absent)

Quantum backend:
  --backend TEXT      aer (default) | runtime
  --shots INT         Shots per circuit  [default: 2048]
  --ibm-token TEXT    IBM Quantum API token (runtime only)
  --ibm-backend TEXT  IBM backend name  [default: ibm_fez]

Optimisation:
  --sa-steps INT      SA steps per restart  [default: 5000]
  --sa-restarts INT   SA restarts  [default: 3]
  --spsa-iters INT    SPSA iterations per fragment  [default: 50]
  --fragment-len INT  Fragment window size (residues)  [default: 7]
  --seed INT          Random seed  [default: 42]
  --alpha FLOAT       CVaR alpha (0–1)  [default: 0.25]
```

---

## Output Files

| File | Description |
|------|-------------|
| `backbone.pdb` | 3D backbone with HELIX/SHEET annotation — open in PyMOL / ChimeraX |
| `run_metrics.json` | Full runtime metrics (wall time, energies, quantum circuit stats) |
| `report.json` | Pipeline provenance and fold summary |
| `fragment_candidates.csv` | All 150 VQE candidate bitstrings with energies |
| `sa_trace.csv` | SA restart history |
| `mj_matrix.csv` | Miyazawa-Jernigan interaction matrix used |
| `fragment_energies.png` | Per-fragment best energy bar chart |
| `output_summary.txt` | Human-readable fold summary |
| `pdb3d.zip` | ZIP of all PDB files |

### Visualising the PDB

Open `backbone.pdb` in [PyMOL](https://pymol.org/), [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/), or [VMD](https://www.ks.uiuc.edu/Research/vmd/):

```python
# PyMOL commands
load backbone.pdb
show cartoon          # renders helices and strands from HELIX/SHEET records
color ss              # colour by secondary structure
```

---

## Turn Codes & Secondary Structure

QuPepFold encodes backbone conformation using 4 discrete turn types on a tetrahedral lattice:

| Turn | φ | ψ | Region | PDB record |
|------|---|---|--------|-----------|
| `0` | −60° | −45° | α-helix | `HELIX` |
| `1` | −135° | 135° | β-sheet | `SHEET` |
| `2` | −75° | 145° | PPII / extended | — |
| `3` | −60° | 140° | coil | — |

Each 2-bit turn code is encoded in the quantum bitstring. The final bitstring (best energy) drives both the 3D geometry and the PDB secondary-structure header records.

---

###  Publications

**Related quantum protein folding works from our group:**

1. Akshay Uttarkar, Vidya Niranjan (2024). Quantum synergy in peptide folding: A comparative study of CVaR-variational quantum eigensolver and molecular dynamics simulation. *International Journal of Biological Macromolecules*. Volume 273, Part 1, 133033. https://doi.org/10.1016/j.ijbiomac.2024.133033
2. Uttarkar, A., Niranjan, V. (2024). A comparative insight into peptide folding with quantum CVaR-VQE algorithm, MD simulations and structural alphabet analysis. *Quantum Inf Process* 23, 48. https://doi.org/10.1007/s11128-024-04261-9
3. A. Uttarkar and V. Niranjan, "Quantum Enabled Protein Folding of Disordered Regions in Ubiquitin C Via Error Mitigated VQE Benchmarked on Tensor Network Simulator and Aria 1," *IEEE Transactions on Molecular, Biological, and Multi-Scale Communications*, doi: 10.1109/TMBMC.2025.3600516. https://ieeexplore.ieee.org/document/11130538
4. A. Uttarkar, A. S. Setlur and V. Niranjan, "T-Gate Enabled Fault-Tolerant Ansatz Circuit Design for Variational Quantum Algorithms in Peptide Folding on Aria-1," *2024 Global AI Summit*, pp. 1271-1276, doi: 10.1109/GlobalAISummit62156.2024.10947993. https://ieeexplore.ieee.org/document/10947993
5. Rutwik S, A. Uttarkar, A. S. Setlur, A. B. H and V. Niranjan, "Exploring VQE for Ground State Energy Calculations of Small Molecules With Higher Bond Orders," *2024 Global AI Summit*, pp. 1182-1187, doi: 10.1109/GlobalAISummit62156.2024.10947806. https://ieeexplore.ieee.org/document/10947806



---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Uninstall

```bash
pip uninstall qupepfold
```
