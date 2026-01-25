[![Qiskit Ecosystem](https://qisk.it/e-ab89baa7)](https://qisk.it/e)
![Static Badge](https://img.shields.io/badge/Accepted_v_0.5.0-PLoS_ONE-green?style=flat&logo=pubmed&labelColor=blue)
![Static Badge](https://img.shields.io/badge/Platform-Linux_%7C_MacOS_%7C_Windows-purple?style=flat&labelColor=blue)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/Qupepfold)
![PyPI - License](https://img.shields.io/pypi/l/Qupepfold)
![PyPI - Version](https://img.shields.io/pypi/v/Qupepfold?logo=pypi)
![Libraries.io dependency status for GitHub repo](https://img.shields.io/librariesio/github/akshayuttarkarc/Qupepfold?style=flat&logo=dependencycheck)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/qupepfold?logo=cpython)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qupepfold)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/qupepfold)



# QuPepFold

QupepFold is a small, research-oriented toolkit that turns short amino-acid sequences into quantum bitstring encodings, optimizes them with a CVaR-VQE routine, and exports 3D PDB files (with CONECT records) for high-probability folds. It’s built to be easy to run, easy to inspect, and easy to tweak.

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

### Our previously published Quantum research works:

1. Akshay Uttarkar, Vidya Niranjan (2024). Quantum synergy in peptide folding: A comparative study of CVaR-variational quantum eigensolver and molecular dynamics simulation. International Journal of Biological Macromolecules. Volume 273, Part 1, 2024, 133033, ISSN 0141-8130, https://doi.org/10.1016/j.ijbiomac.2024.133033
2. Uttarkar, A., Niranjan, V. A comparative insight into peptide folding with quantum CVaR-VQE algorithm, MD simulations and structural alphabet analysis. Quantum Inf Process 23, 48 (2024). https://doi.org/10.1007/s11128-024-04261-9
3. A. Uttarkar, A. S. Setlur and V. Niranjan, "T-Gate Enabled Fault-Tolerant Ansatz Circuit Design for Variational Quantum Algorithms in Peptide Folding on Aria-1," 2024 International Conference on Artificial Intelligence and Emerging Technology (Global AI Summit), Greater Noida, India, 2024, pp. 1271-1276, doi: 10.1109/GlobalAISummit62156.2024.10947993
4. A. Uttarkar and V. Niranjan, "Quantum Enabled Protein Folding of Disordered Regions in Ubiquitin C Via Error Mitigated VQE Benchmarked on Tensor Network Simulator and Aria 1," in IEEE Transactions on Molecular, Biological, and Multi-Scale Communications, doi: 10.1109/TMBMC.2025.3600516

### 🚀 Future Version Update: GPU Acceleration

The next major planned update is integrating **GPU support** to accelerate the quantum circuit simulations and classical optimization loops at the core of QuPepFold.

---

### Why It Matters

* **Massive Parallelism**: GPUs excel at performing thousands of calculations simultaneously. This is ideal for the heavy matrix and vector math required to simulate quantum states, offering a significant speedup over sequential CPU processing. 

* **Accelerated VQE Loop**: The VQE (Variational Quantum Eigensolver) algorithm is iterative. By running the numerous circuit evaluations on a GPU, the entire optimization process to find the lowest energy state is drastically shortened.

* **Bigger, Bolder Simulations**: Faster computation makes it feasible to tackle more complex problems, such as longer amino acid sequences (which require more qubits) or running a higher number of optimization attempts (`--tries`) to ensure a better result.

* **Rapid Prototyping**: What currently takes hours can be reduced to minutes. This allows for more interactive research, enabling rapid testing of different parameters and sequences.

---

### Expected Impact on Runtime

We anticipate a **performance boost of 10x to over 50x**, depending on the specific hardware and problem complexity.

A typical simulation could see a runtime reduction like this:
* **On CPU**: ~1 hour
* **On GPU**: ~2-5 minutes



