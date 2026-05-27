"""QuPepFold: Quantum-classical hybrid peptide folding simulations.

A scientifically rigorous package for protein folding combining:
- Classical global search on lattice models (SA/PT)
- Quantum fragment refinement via SamplerV2
- Overlap-aware dynamic programming stitching
- 3D backbone coordinate generation with secondary structure annotation
"""

__version__ = "1.3.1"


from .types import (
    FoldConfig,
    FragmentSpec,
    FragmentEnergyTable,
    FragmentCandidate,
    FoldResult,
)
from .config import get_default_config
from .pipeline import run_fold

__all__ = [
    "FoldConfig",
    "FragmentSpec",
    "FragmentEnergyTable",
    "FragmentCandidate",
    "FoldResult",
    "get_default_config",
    "run_fold",
]
