[![Qiskit Ecosystem](https://qisk.it/e-ab89baa7)](https://qisk.it/e)
![Static Badge](https://img.shields.io/badge/Publication-PLoS_One-purple?style=flat&logo=pubmed&labelColor=blue)
![Static Badge](https://img.shields.io/badge/Platform-Linux_%7C_MacOS_%7C_Windows-purple?style=flat&labelColor=blue)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/qupepfold?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BRIGHTGREEN&left_text=Total+Downloads)](https://pepy.tech/projects/qupepfold)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/qupepfold?period=monthly&units=NONE&left_color=GRAY&right_color=BRIGHTGREEN&left_text=Last+Month+Downloads)](https://pepy.tech/projects/qupepfold)
![PyPI - License](https://img.shields.io/pypi/l/Qupepfold)
![PyPI - Version](https://img.shields.io/pypi/v/Qupepfold?logo=pypi)
![Libraries.io dependency status for GitHub repo](https://img.shields.io/librariesio/github/akshayuttarkarc/Qupepfold?style=flat&logo=dependencycheck)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/qupepfold?logo=cpython)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qupepfold)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/qupepfold)

---

> ### 📄 If you use QuPepFold in your research, please cite:
> **Uttarkar A, Niranjan V, Saxena A, Kumar V (2026)**  
> QuPepFold: A python package for hybrid quantum-classical protein folding simulations with CVaR-optimized VQE.  
> *PLoS One* **21**(2): e0342012.  
> 🔗 https://doi.org/10.1371/journal.pone.0342012

---

# QuPepFold

QupepFold is a small, research-oriented toolkit that turns short amino-acid sequences into quantum bitstring encodings, optimizes them with a CVaR-VQE routine, and exports 3D PDB files (with CONECT records) for high-probability folds. It's built to be easy to run, easy to inspect, and easy to tweak.

### Features

✨ Easy installation.<br>
✨ Simple CLI usage.<br>
✨ Get detailed outputs: Qubit mapping, best CVaR energy, probable bitstrings, and more!<br>
✨ Visualize your results with optimal_circuit.png, cvar_scatter.png, and bitstring_histogram.png.<br>
✨ Export 3D PDB files and summaries.<br>

## Installation and easy way to use in CLI

`pip3 install qupepfold`<br>
`pip3 install pylatexenc`<br>

`qupepfold --seq APRLFHG --tries 30 --shots 1000 --alpha 0.025 --write-csv --out /path/to/output/directory`

---

### What to expect in output in terminal

1. Qubit mapping (config/interaction/ancilla counts)
2. Best CVaR energy
3. Most probable bitstring (with probability)
3. Lowest-energy bitstring (with energy)

### Results in the output folder

1. output_summary.txt — quick result summary
2. optimal_circuit.png — ansatz diagram (no measurements)
2. cvar_scatter.png — CVaR value per multi-start iteration
2. bitstring_histogram.png — bar chart for states ≥ threshol
3. bitstring_summary.csv — [bitstring, cfg_bits, probability, energy, exported_PDB3D]
4. bitstring_summary_cvar.csv — same distribution (kept for continuity)
5. most_negative_energy_breakdown.png + .csv — component energies (backbone/MJ/distance/locality) for the lowest energy state
6. pdb3d/*.pdb — one PDB per exported bitstring, with CONECT bonds
7. pdb3d_bitstrings_ge_2pct.zip — bundle of those PDBs

### Uninstall

pip3 uninstall qupepfold

---

### 📚 Publications

**★ Primary citation for QuPepFold (v0.8.0):**

> Uttarkar A, Niranjan V, Saxena A, Kumar V (2026).  
> QuPepFold: A python package for hybrid quantum-classical protein folding simulations with CVaR-optimized VQE.  
> *PLoS One* **21**(2): e0342012.  
> https://doi.org/10.1371/journal.pone.0342012

**Related quantum protein folding works from our group:**

1. Akshay Uttarkar, Vidya Niranjan (2024). Quantum synergy in peptide folding: A comparative study of CVaR-variational quantum eigensolver and molecular dynamics simulation. *International Journal of Biological Macromolecules*. Volume 273, Part 1, 133033. https://doi.org/10.1016/j.ijbiomac.2024.133033
2. Uttarkar, A., Niranjan, V. (2024). A comparative insight into peptide folding with quantum CVaR-VQE algorithm, MD simulations and structural alphabet analysis. *Quantum Inf Process* 23, 48. https://doi.org/10.1007/s11128-024-04261-9
3. A. Uttarkar and V. Niranjan, "Quantum Enabled Protein Folding of Disordered Regions in Ubiquitin C Via Error Mitigated VQE Benchmarked on Tensor Network Simulator and Aria 1," *IEEE Transactions on Molecular, Biological, and Multi-Scale Communications*, doi: 10.1109/TMBMC.2025.3600516. https://ieeexplore.ieee.org/document/11130538
4. A. Uttarkar, A. S. Setlur and V. Niranjan, "T-Gate Enabled Fault-Tolerant Ansatz Circuit Design for Variational Quantum Algorithms in Peptide Folding on Aria-1," *2024 Global AI Summit*, pp. 1271-1276, doi: 10.1109/GlobalAISummit62156.2024.10947993. https://ieeexplore.ieee.org/document/10947993
5. Rutwik S, A. Uttarkar, A. S. Setlur, A. B. H and V. Niranjan, "Exploring VQE for Ground State Energy Calculations of Small Molecules With Higher Bond Orders," *2024 Global AI Summit*, pp. 1182-1187, doi: 10.1109/GlobalAISummit62156.2024.10947806. https://ieeexplore.ieee.org/document/10947806
   

---

## What's New in v0.9.0 — Performance & Correctness

v0.9.0 is a focused engineering release over v0.8.0. No algorithm or output format changes — the same CVaR-VQE method, same Hamiltonian, same PDB schema — but the same runs now complete significantly faster and without risk of RAM exhaustion on larger inputs.

### ⚡ Performance Improvements

| # | Where | What changed | Impact |
|---|-------|-------------|--------|
| 1 | `fill_config_bits` hot path | Pre-compiled numpy closure (`_make_fill_fn`) replaces per-call list comprehension | Eliminates O(N) string alloc × millions of calls |
| 2 | `delta_vec` inner loop | Vectorised with a single `np.arange` + broadcasting instead of 4× `np.array` + 4× `np.arange` per call | ~4× fewer array allocations per interaction |
| 3 | Interaction energy loop | `delta_vec(i,j)` result reused for the `dir2` term instead of being recomputed | Eliminates one redundant vector call per active pair |
| 4 | `exact_hamiltonian` | Module-level bounded LRU cache (`_BoundedCache`, 16 384 entries) — identical bitstrings computed only once per session | ~4 000 CVaR evaluations × ~100 states → cache hits after first pass |
| 5 | `build_mj_interactions` | 20×20 MJ matrix generated once and cached; subsequent calls slice the pre-built matrix | Eliminates RNG + matrix construction on every call |
| 6 | `cvar_objective` | `np.fromiter` avoids intermediate list; `np.searchsorted` replaces `count_nonzero`; pre-allocated weights array replaces `.tolist()` + append | Tighter inner loop for 4 000+ CVaR evaluations per run |
| 7 | `statevector_fold_probs` | Chunked numpy bit-extraction (8 192 rows/chunk) replaces Python loop over all 2^Q basis states | O(2^Q × Q) Python loop → O(2^Q / 8192 × M) numpy ops |
| 8 | Interaction pairs | `_interaction_pairs(N)` pre-computes the `(i,j)` list once; stored in `hyper["_pairs"]` | Eliminates repeated nested-loop overhead per bitstring |
| 9 | PDB writer | `io.StringIO` buffer replaces list-of-strings + final join | Fewer intermediate string allocations during PDB export |

---

### Future Version Update: Scalability for larger peptides
