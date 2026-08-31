"""Backend abstraction for quantum samplers."""

from .base import SamplerBackend
from .aer_sampler import AerSamplerBackend

# RuntimeSamplerBackend requires qiskit-ibm-runtime (optional)
# Import lazily in pipeline.py when backend="runtime"

__all__ = ["SamplerBackend", "AerSamplerBackend"]
