from setuptools import setup, find_packages

setup(
    name="qupepfold",                   # pip package name, lowercase
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="QuPepFold: Quantum peptide folding simulations with Qiskit",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/qupepfold",
    packages=find_packages(),           # will include the qupepfold/ folder
    install_requires=[
        "qiskit>=0.39",
        "qiskit-aer",
        "numpy",
        "matplotlib",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Quantum Computing",
    ],
    python_requires=">=3.7",
)
