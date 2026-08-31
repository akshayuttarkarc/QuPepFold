from .qupepfold import (
    generate_turn2qubit,
    count_interaction_qubits,
    build_mj_interactions,
    exact_hamiltonian,
    build_scalable_ansatz,
    statevector_fold_probs,
    optimize_cvar_multistart,
    make_pdbs_from_probs,
    plot_energy_breakdown_for_most_negative,
    MAX_EVALS_PER_TRY,
    # GPU support
    check_gpu_available,
    get_simulator,
    USE_GPU_DEFAULT,
    # v0.9.0 performance helpers
    clear_energy_cache,
    _make_fill_fn,
    _interaction_pairs,
)

__version__ = "0.9.0"

__all__ = [
    "generate_turn2qubit",
    "count_interaction_qubits",
    "build_mj_interactions",
    "exact_hamiltonian",
    "build_scalable_ansatz",
    "statevector_fold_probs",
    "optimize_cvar_multistart",
    "make_pdbs_from_probs",
    "plot_energy_breakdown_for_most_negative",
    "MAX_EVALS_PER_TRY",
    # GPU support
    "check_gpu_available",
    "get_simulator",
    "USE_GPU_DEFAULT",
    # v0.9.0 performance helpers
    "clear_energy_cache",
    "_make_fill_fn",
    "_interaction_pairs",
    "__version__",
]
