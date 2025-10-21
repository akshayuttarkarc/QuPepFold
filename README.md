# QuPepFold

QupepFold is a small, research-oriented toolkit that turns short amino-acid sequences into quantum bitstring encodings, optimizes them with a CVaR-VQE routine, and exports 3D PDB files (with CONECT records) for high-probability folds. It’s built to be easy to run, easy to inspect, and easy to tweak.

### Features

✨ Easy installation.<br>
✨ Simple CLI usage.<br>
✨ Optional CUDA acceleration for the heavy statevector simulations.<br>
✨ Get detailed outputs: Qubit mapping, best CVaR energy, probable bitstrings, and more!<br>
✨ Visualize your results with optimal_circuit.png, cvar_scatter.png, and bitstring_histogram.png.<br>
✨ Export 3D PDB files and summaries.<br>

## Installation and easy way to use in CLI

`pip3 install qupepfold`<br>
`pip3 install pylatexenc`<br>

Optional CUDA build of Aer (Linux + NVIDIA GPU):<br>
`pip3 install "qupepfold[gpu]"`<br>
or install directly: `pip3 install "qiskit-aer-gpu>=0.12"`<br>

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

### 🚀 GPU Acceleration (CUDA)

QuPepFold now ships with **opt-in CUDA acceleration** for its statevector simulations. When an NVIDIA GPU with the CUDA toolkit is available, you can enable GPU execution in one of two ways:

* **Interactive mode** – answer `y` when prompted “Use CUDA GPU acceleration if available?”.
* **CLI mode** – append `--gpu` (optionally `--gpu-precision {single,double}`) to your command:  
  `qupepfold --seq APRLFHG --tries 30 --shots 1000 --alpha 0.025 --gpu`

If the CUDA-enabled Aer build is missing or the GPU is unavailable, QuPepFold automatically falls back to the high-precision CPU simulator and continues running.

**Additional dependencies:** install the GPU extras via `pip install "qupepfold[gpu]"` or install `qiskit-aer-gpu>=0.12` manually. The GPU build is currently published for Linux platforms with a recent NVIDIA driver.

**What accelerates?** The CVaR-VQE loop now evaluates statevectors on the GPU, significantly reducing the time spent in the inner optimisation. Histogram sampling also reuses the GPU simulator when possible.



