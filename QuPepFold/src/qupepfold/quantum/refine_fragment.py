"""Fragment refinement using variational quantum optimization.

Combines ansatz, warm-start, SPSA optimization, and candidate extraction.
"""

from typing import List, Optional, Dict
import numpy as np
import copy  # For deepcopy

from ..types import FoldConfig, FragmentSpec, FragmentEnergyTable, FragmentCandidate
from .ansatz import build_fragment_ansatz, get_initial_parameters
from .warmstart import basis_state_init, biased_ry_init
from .cost_expectation import expected_energy, cvar_energy, top_k_candidates
from .spsa import spsa_optimize, get_optimizer
from .backends.base import SamplerBackend
from ..model.energy_fragment import compute_context_energy
from ..model.encoding import decode_turns, index_to_bitstring


def refine_fragment(
    fragment: FragmentSpec,
    energy_table: FragmentEnergyTable,
    backend: SamplerBackend,
    config: FoldConfig,
    mj_matrix: np.ndarray,
    warmstart_bits: Optional[str] = None,
    global_positions: Optional[np.ndarray] = None,
    verbose: bool = False,
    fragment_idx: int = 0,
) -> List[FragmentCandidate]:
    """Refine a fragment using variational quantum optimization with optional context.
    
    Supports QA-Anchored optimization:
    - Fixed boundaries (overlap bits frozen to warm-start values)
    - Context energy (interactions with fixed global environment)
    """
    # Create a context-aware energy table valid for this refinement
    # We copy the original table so we don't pollute it if reused
    context_table = copy.deepcopy(energy_table)
    
    # QA-Anchored Logic:
    # 1. Constrain search space: Set energy to INF for states that 
    #    violate the fixed overlap boundaries from warm-start.
    # 2. Add context energy: Add E_interaction with global environment.
    
    if warmstart_bits and global_positions is not None:
        n_bits = fragment.n_bits
        
        # Identify fixed bit ranges
        # Left overlap: first 2 * overlap_left_turns bits
        n_left_bits = 2 * fragment.overlap_left_turns
        # Right overlap: last 2 * overlap_right_turns bits
        n_right_bits = 2 * fragment.overlap_right_turns
        
        left_fixed = warmstart_bits[:n_left_bits] if n_left_bits > 0 else ""
        right_fixed = warmstart_bits[-n_right_bits:] if n_right_bits > 0 else ""
        
        # Decode turns for context energy calc
        n_states = 2 ** n_bits
        start_res = fragment.start_res
        
        # Modify energy table in-place
        for idx in range(n_states):
            # Check fixed boundaries first (fast)
            bits = index_to_bitstring(idx, n_bits)
            
            valid_boundary = True
            if n_left_bits > 0 and not bits.startswith(left_fixed):
                valid_boundary = False
            if n_right_bits > 0 and not bits.endswith(right_fixed):
                valid_boundary = False
                
            if not valid_boundary:
                context_table.energies[idx] = np.inf
                continue
            
            # If valid, compute context energy
            turns = decode_turns(bits)
            e_context = compute_context_energy(
                fragment_turns=turns,
                global_positions=global_positions,
                fragment_start_res=start_res,
                mj_matrix=mj_matrix,
                contact_min_sep=config.contact_min_sep,
                contact_cutoff=config.contact_cutoff
            )
            
            # Update energy (Internal + Context)
            if context_table.energies[idx] != np.inf:
                context_table.energies[idx] += e_context
    
    # Use the context-aware table for VQE
    energy_table_to_use = context_table
    n_qubits = fragment.n_bits
    
    from .error_mitigation import PostSelectionFilter
    from qiskit import QuantumCircuit
    
    # Build ansatz
    with_warmstart = bool(warmstart_bits)
    
    # Check if warm-start is valid in new table
    if with_warmstart and warmstart_bits:
        ws_energy = energy_table_to_use.energy_of_bitstring(warmstart_bits)
        if ws_energy == np.inf:
            if verbose:
                print(f"  ⚠ Warm-start bitstring is invalid in context table! Disabling warm-start.")
            with_warmstart = False
            warmstart_bits = None
    
    # Construct circuit with appropriate initialization
    if not with_warmstart or not warmstart_bits:
        # No warm-start: build ansatz with initial Hadamard superposition layer
        circuit = build_fragment_ansatz(
            n_qubits=n_qubits,
            depth=config.ansatz_depth,
            with_warmstart=False,
        )
    elif config.warmstart_mode == "basis":
        # Basis state initialization (|warmstart_bits>)
        qc = QuantumCircuit(n_qubits)
        basis_state_init(qc, warmstart_bits)
        ansatz_body = build_fragment_ansatz(
            n_qubits=n_qubits,
            depth=config.ansatz_depth,
            with_warmstart=True,
        )
        circuit = qc.compose(ansatz_body)
    else:  # biased_ry mode
        qc = QuantumCircuit(n_qubits)
        biased_ry_init(qc, warmstart_bits, config.warmstart_bias)
        ansatz_body = build_fragment_ansatz(
            n_qubits=n_qubits,
            depth=config.ansatz_depth,
            with_warmstart=True,
        )
        circuit = qc.compose(ansatz_body)
    
    # Initial parameters
    n_params = circuit.num_parameters
    init_strategy = "small" if with_warmstart else "random"
    x0 = get_initial_parameters(n_params, strategy=init_strategy, seed=config.seed)
    
    # Track iteration progress
    iteration_count = [0]
    
    # Define cost function with optional verbose output
    def cost_fn(params: np.ndarray) -> float:
        """Cost function: CVaR or expected energy from sampler using CONTEXT table."""
        iteration_count[0] += 1
        
        # Run sampler
        prob_dicts = backend.run_sampler(
            circuits=[circuit],
            param_values=[params],
            shots=config.shots,
        )
        prob_dict = prob_dicts[0]
        
        # Compute energy with CONTEXT table (CVaR if alpha < 1.0, otherwise expected value)
        if config.cvar_alpha < 1.0:
            energy = cvar_energy(prob_dict, energy_table_to_use, alpha=config.cvar_alpha)
        else:
            energy = expected_energy(prob_dict, energy_table_to_use)
        
        # Progress update
        if verbose and iteration_count[0] % 10 == 0:
            iter_num = iteration_count[0] // 2
            pct = 100 * iter_num / config.spsa_iterations
            print(f"\r    SPSA iteration {iter_num}/{config.spsa_iterations} ({pct:.0f}%) | E={energy:.2f}   ", end="", flush=True)
        
        return float(energy)
    
    # Run optimization with the configured optimizer
    optimizer_name = getattr(config, 'optimizer', 'spsa')
    optimize_fn = get_optimizer(optimizer_name)
    
    best_params, best_cost, cost_history = optimize_fn(
        cost_fn=cost_fn,
        x0=x0,
        n_iterations=config.spsa_iterations,
        a=getattr(config, 'spsa_a', 0.1),
        c=getattr(config, 'spsa_c', 0.1),
        seed=config.seed,
    )
    
    if verbose:
        print(f"\r    SPSA complete: final E={best_cost:.2f}                    ")
    
    # Get final distribution
    final_probs = backend.run_sampler(
        circuits=[circuit],
        param_values=[best_params],
        shots=config.shots * 2,
    )[0]
    
    # Apply post-selection filter to discard self-intersections
    post_filter = PostSelectionFilter(fragment=fragment, warmstart_bits=warmstart_bits)
    mitigated_probs = post_filter(final_probs)
    
    # Extract candidates using CONTEXT table
    top_by_prob = top_k_candidates(
        mitigated_probs,
        energy_table_to_use,
        k=config.top_k_candidates,
        by="probability",
    )
    
    # Candidate selection intentionally excludes the exact best states from the
    # exhaustively enumerated energy table.  Injecting those states here would
    # allow them to displace every sampled state during the energy ranking below,
    # making the refinement result independent of the VQE distribution.
    unique_bits = set()
    candidates = []
    
    # Always include the warm-start bitstring if available
    if with_warmstart and warmstart_bits:
        ws_energy = energy_table_to_use.energy_of_bitstring(warmstart_bits)
        if ws_energy < 1e9:
            candidates.append(FragmentCandidate(
                bits=warmstart_bits,
                energy=ws_energy,
                probability=1.0,
                meta={"source": "warm_start_sa"}
            ))
            unique_bits.add(warmstart_bits)
    
    # Add VQE-sampled candidates
    for bits, energy, prob in sorted(top_by_prob, key=lambda x: x[1]):
        if bits not in unique_bits:
            unique_bits.add(bits)
            candidates.append(FragmentCandidate(
                bits=bits,
                energy=energy,
                probability=prob,
                meta={
                    "source": "vqe_sampling",
                    "backend": type(backend).__name__,
                    "shots": config.shots,
                    "optimizer": getattr(config, 'optimizer', 'spsa'),
                    "n_iterations": config.spsa_iterations,
                    "cvar_alpha": config.cvar_alpha,
                    "final_cost": best_cost,
                    "convergence_trace": cost_history,
                },
            ))
    
    # Rank only the warm start and states actually observed from the VQE sampler.
    candidates.sort(key=lambda c: c.energy)
    return candidates[:config.top_m_by_energy]


def refine_all_fragments(
    fragments: List[FragmentSpec],
    energy_tables: List[FragmentEnergyTable],
    backend: SamplerBackend,
    config: FoldConfig,
    mj_matrix: np.ndarray = None,
    global_bits: Optional[str] = None,
    global_positions: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> List[List[FragmentCandidate]]:
    """Refine all fragments in the list using QA-Anchored Optimization."""
    fragment_candidates = []
    
    for i, (frag, table) in enumerate(zip(fragments, energy_tables)):
        if verbose:
            print(f"  Fragment {i+1}/{len(fragments)}: {frag.sequence}")
        
        # Extract warm-start bits for this fragment from global solution
        warmstart_bits = None
        if global_bits:
            n_res_frag = frag.end_res - frag.start_res
            n_turn_frag = n_res_frag - 1
            bit_start = frag.start_res * 2
            bit_end = bit_start + (n_turn_frag * 2)
            
            if bit_end <= len(global_bits):
                warmstart_bits = global_bits[bit_start:bit_end]
        
        cands = refine_fragment(
            fragment=frag,
            energy_table=table,
            backend=backend,
            config=config,
            mj_matrix=mj_matrix, 
            warmstart_bits=warmstart_bits,
            global_positions=global_positions,
            verbose=verbose,
            fragment_idx=i,
        )
        fragment_candidates.append(cands)
        
        if verbose:
            best = cands[0]
            print(f"    → {len(cands)} candidates, best E={best.energy:.2f}")
            
    return fragment_candidates


def refine_all_fragments_from_table(
    fragments: List[FragmentSpec],
    energy_tables: List[FragmentEnergyTable],
    top_k: int = 10,
    verbose: bool = False,
) -> List[List[FragmentCandidate]]:
    """Get the best candidates for each fragment directly from energy tables."""
    all_candidates = []
    
    for i, (fragment, table) in enumerate(zip(fragments, energy_tables)):
        if verbose:
            print(f"  Fragment {i+1}/{len(fragments)}: {fragment.sequence}")
        
        candidates = table.get_best_candidates(top_k=top_k)
        
        if verbose:
            best_e = candidates[0].energy if candidates else float('inf')
            print(f"    → {len(candidates)} candidates from table, best E={best_e:.2f}")
        
        all_candidates.append(candidates)
    
    return all_candidates
