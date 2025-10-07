from setuptools import setup, find_packages
import io, os

here = os.path.abspath(os.path.dirname(__file__))

# ─── EDIT THESE LISTS ────────────────────────────────────────────────────────────
authors = [
    "Akshay Uttarkar",
    "Vinay Kumar",               # 2nd author
    "Vidya Niranjan",             # 3rd author
]
author_emails = [
    "akshayuttarkar@gmail.com",
]
# ────────────────────────────────────────────────────────────────────────────────

# Extended description pulled from the tool’s capabilities
extended_description = """
QuPepFold is a command-line quantum peptide folding simulator built on Qiskit.  
Given a protein sequence (2–10 amino acids), it:

  • Generates a turn-to-qubit mapping for fold degrees of freedom  
  • Builds a Miyazawa–Jernigan interaction matrix from the sequence  
  • Defines an ansatz circuit with parameterized single-qubit rotations and entangling CX gates  
  • Uses a CVaR-based Variational Quantum Eigensolver (VQE) with SciPy’s COBYLA optimizer  
    – Supports configurable shot count (default 1024) and iteration limit (default 50)  
  • Outputs:  
    1. A text summary of minimum CVaR energy and configuration bits  
    2. A PNG of the optimal circuit diagram  
    3. A convergence scatter plot of CVaR energies per iteration  
    4. A histogram of high-probability bitstrings (≥2%)
    5. A viewer-ready 3D peptide backbones consistent with the discrete folding cues captured by the quantum bitstrings.

Results are saved under a user-specified directory (default `./results`).  
Leverages Qiskit Aer for fast, local simulation and Matplotlib for visualization.
"""

# Read your existing README and prepend the extended part
readme_path = os.path.join(here, "README.md")
with io.open(readme_path, encoding="utf-8") as f:
    readme_md = f.read()

long_description = extended_description + "\n\n" + readme_md

setup(
    name="qupepfold",
    version="0.5.0",
    author=", ".join(authors),
    author_email=", ".join(author_emails),
    description="QuPepFold: Quantum peptide folding simulations with Qiskit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://vidyaniranjan.co.in/?i=1",
    packages=find_packages(),
    include_package_data=True,          
    install_requires=[
        "qiskit>=0.39",
        "qiskit-aer",
        "numpy",
        "matplotlib",
        "scipy",
    ],
        entry_points={
         "console_scripts": [
             "qupepfold=qupepfold.cli:main",  # normal setuptools stub
        ],
     },
    scripts=["scripts/qupepfold"],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Quantum Computing",
    ],
)
