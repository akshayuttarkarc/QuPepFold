"""Qiskit Aer SamplerV2 backend for local simulation.

Uses the Aer simulator for noiseless or noisy local execution.
"""

from typing import List, Dict
import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer.primitives import SamplerV2
    from qiskit_aer import AerSimulator
except ImportError as e:
    raise ImportError(
        "qiskit and qiskit-aer are required. "
        "Install with: pip install qiskit qiskit-aer"
    ) from e

from .base import SamplerBackend, counts_to_probs, reverse_bitstring_endianness


class AerSamplerBackend(SamplerBackend):
    """Local Aer SamplerV2 backend.
    
    Supports both statevector (exact) and shot-based sampling.
    
    Example:
        >>> backend = AerSamplerBackend()
        >>> results = backend.run_sampler([circuit], [params], shots=1000)
        >>> backend.close()
    """
    
    def __init__(self, use_gpu: bool = False, noise_model=None):
        """Initialize Aer sampler.
        
        Args:
            use_gpu: If True, attempt to use GPU acceleration.
            noise_model: Optional Qiskit noise model for noisy simulation.
        """
        self.use_gpu = use_gpu
        self.noise_model = noise_model
        
        # Configure simulator
        sim_options = {}
        if use_gpu:
            sim_options['device'] = 'GPU'
            sim_options['method'] = 'statevector_gpu'
        
        self._simulator = AerSimulator(**sim_options)
        if noise_model:
            self._simulator.set_options(noise_model=noise_model)
    
    def run_sampler(
        self,
        circuits: List[QuantumCircuit],
        param_values: List[np.ndarray],
        shots: int,
    ) -> List[Dict[str, float]]:
        """Execute circuits with Aer SamplerV2.
        
        Args:
            circuits: List of parameterized circuits.
            param_values: Parameter values for each circuit.
            shots: Number of shots.
            
        Returns:
            List of probability dicts.
        """
        results = []
        
        for circuit, params in zip(circuits, param_values):
            # Bind parameters if circuit is parameterized
            if circuit.num_parameters > 0:
                bound_circuit = circuit.assign_parameters(params)
            else:
                bound_circuit = circuit
            
            # Add measurements if not present
            if bound_circuit.num_clbits == 0:
                bound_circuit = bound_circuit.copy()
                bound_circuit.measure_all()
            
            # Transpile for simulator
            transpiled = transpile(bound_circuit, self._simulator)
            
            # Run
            job = self._simulator.run(transpiled, shots=shots)
            result = job.result()
            counts = result.get_counts(0)
            
            # Convert to probabilities with correct endianness
            probs = {}
            for bitstring, count in counts.items():
                # Qiskit returns little-endian, convert to big-endian
                big_endian = reverse_bitstring_endianness(bitstring)
                probs[big_endian] = count / shots
            
            results.append(probs)
        
        return results
    
    def run_statevector(self, circuit: QuantumCircuit, params: np.ndarray = None) -> Dict[str, float]:
        """Get exact probability distribution via statevector.
        
        Args:
            circuit: Quantum circuit (no measurements needed).
            params: Optional parameter values.
            
        Returns:
            Probability dict for all basis states.
        """
        from qiskit.quantum_info import Statevector
        
        if params is not None and circuit.num_parameters > 0:
            circuit = circuit.assign_parameters(params)
        
        sv = Statevector.from_instruction(circuit)
        probs_array = np.abs(sv.data) ** 2
        
        n_qubits = circuit.num_qubits
        probs = {}
        for idx, p in enumerate(probs_array):
            if p > 1e-10:  # Skip near-zero probabilities
                bitstring = format(idx, f'0{n_qubits}b')
                # Statevector uses little-endian, convert to big-endian
                probs[reverse_bitstring_endianness(bitstring)] = float(p)
        
        return probs
    
    def close(self) -> None:
        """No cleanup needed for Aer."""
        pass
