"""IBM Runtime SamplerV2 backend for real QPU execution.

Uses IBM Runtime sessions for efficient iterative workloads.
"""

from typing import List, Dict, Optional
import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
except ImportError as e:
    raise ImportError(
        "qiskit-ibm-runtime is required for QPU execution. "
        "Install with: pip install qiskit-ibm-runtime"
    ) from e

from .base import SamplerBackend, reverse_bitstring_endianness


class RuntimeSamplerBackend(SamplerBackend):
    """IBM Runtime SamplerV2 backend with session management.
    
    Uses sessions for iterative workloads (optimizer loops) to get
    job prioritization and reduced queuing overhead.
    
    Example:
        >>> backend = RuntimeSamplerBackend("ibm_brisbane")
        >>> results = backend.run_sampler([circuit], [params], shots=4000)
        >>> backend.close()
    """
    
    def __init__(
        self,
        backend_name: str,
        instance: Optional[str] = None,
        channel: str = "ibm_quantum_platform",
    ):
        """Initialize IBM Runtime backend.
        
        Args:
            backend_name: IBM backend name (e.g., "ibm_brisbane").
            instance: Optional instance string (hub/group/project).
            channel: "ibm_quantum_platform" or "ibm_cloud".
        """
        self.backend_name = backend_name
        self._service = QiskitRuntimeService(channel=channel, instance=instance)
        self._backend = self._service.backend(backend_name)
        self._session: Optional[Session] = None
        self._sampler: Optional[SamplerV2] = None
    
    def _ensure_session(self) -> None:
        """Ensure a session is open."""
        if self._session is None:
            self._session = Session(backend=self._backend)
            self._session.__enter__()
            
            # Configure sampler options
            options = SamplerOptions()
            options.default_shots = 4000  # Default, overridden per call
            
            self._sampler = SamplerV2(mode=self._session, options=options)
    
    def run_sampler(
        self,
        circuits: List[QuantumCircuit],
        param_values: List[np.ndarray],
        shots: int,
    ) -> List[Dict[str, float]]:
        """Execute circuits on IBM QPU via SamplerV2.
        
        Args:
            circuits: List of parameterized circuits.
            param_values: Parameter values for each circuit.
            shots: Number of shots per circuit.
            
        Returns:
            List of probability dicts.
        """
        self._ensure_session()
        
        results = []
        
        for circuit, params in zip(circuits, param_values):
            # Bind parameters
            if circuit.num_parameters > 0:
                bound_circuit = circuit.assign_parameters(params)
            else:
                bound_circuit = circuit
            
            # Add measurements if not present
            if bound_circuit.num_clbits == 0:
                bound_circuit = bound_circuit.copy()
                bound_circuit.measure_all()
            
            # Transpile for target backend
            transpiled = transpile(bound_circuit, self._backend)
            
            # Run via sampler - V2 API
            job = self._sampler.run([transpiled], shots=shots)
            result = job.result()
            
            # Extract counts from result
            # SamplerV2 returns PrimitiveResult with data bundles
            pub_result = result[0]
            counts = pub_result.data.meas.get_counts()
            
            # Convert to probabilities with correct endianness
            probs = {}
            total_shots = sum(counts.values())
            for bitstring, count in counts.items():
                big_endian = reverse_bitstring_endianness(bitstring)
                probs[big_endian] = count / total_shots
            
            results.append(probs)
        
        return results
    
    def get_backend_info(self) -> Dict:
        """Get backend configuration info."""
        return {
            "name": self.backend_name,
            "num_qubits": self._backend.num_qubits,
            "basis_gates": list(self._backend.basis_gates),
        }
    
    def close(self) -> None:
        """Close the runtime session."""
        if self._session is not None:
            self._session.__exit__(None, None, None)
            self._session = None
            self._sampler = None
