"""Precompute fragment energy lookup tables.

For a 6-AA fragment (5 turns, 10 bits), there are 2^10 = 1024 possible states.
We enumerate all and compute their lattice-based energies once.

Energy terms from original Hamiltonian:
- H_gc (geometric constraint): lamBack, lamDis, lamLoc
- H_in (interaction): MJ contact energy
- Overlap penalty for self-intersection
"""

from typing import Optional, List, Tuple
import numpy as np

from ..types import FoldConfig, FragmentSpec, FragmentEnergyTable
from .encoding import decode_turns, index_to_bitstring
from .lattice import trace_positions, count_overlaps, contacts, radius_of_gyration


# Penalty weights from original Hamiltonian (exact_hamiltonian in qupepfold.py)
DEFAULT_LAM_DIS = 720.0   # Distance constraint penalty
DEFAULT_LAM_LOC = 20.0    # Locality constraint penalty
DEFAULT_LAM_BACK = 50.0   # Adjacent equal turns penalty


def compute_delta_vec(turns: List[int], a: int, b: int) -> np.ndarray:
    """Compute delta vector for turns from index a to b.
    
    From original: alternating sum of indicator vectors for each turn type.
    This encodes the geometric path displacement in turn-space.
    
    Args:
        turns: List of turn codes.
        a: Start index (inclusive).
        b: End index (exclusive).
        
    Returns:
        4-element vector representing the turn displacement.
    """
    seg = turns[a:b]
    vec = np.zeros(4)
    for k in range(4):
        mask = np.array([1 if t == k else 0 for t in seg], float)
        if mask.size > 0:
            vec[k] = np.sum(((-1) ** np.arange(mask.size)) * mask)
    return vec


def compute_backbone_penalty(turns: List[int], lam_back: float = DEFAULT_LAM_BACK) -> float:
    """Compute penalty for adjacent equal turns (H_gc backbone term).
    
    Penalizes consecutive identical turn codes to encourage structural variety.
    
    Args:
        turns: List of turn codes.
        lam_back: Penalty weight per violation.
        
    Returns:
        Total backbone penalty.
    """
    penalty = 0.0
    for i in range(len(turns) - 1):
        if turns[i] == turns[i + 1]:
            penalty += lam_back
    return penalty


def compute_geometric_constraints(
    turns: List[int],
    n_residues: int,
    lam_dis: float = DEFAULT_LAM_DIS,
    lam_loc: float = DEFAULT_LAM_LOC,
    positions: Optional[np.ndarray] = None,
    contact_pairs: Optional[List[Tuple[int, int]]] = None,
) -> float:
    """Compute geometric constraint penalties for CONTACTING residues only.
    
    Key insight: In the original VQE with control bits, lamDis/lamLoc were
    only applied when the control bit was '1' (interaction enabled). To mimic
    this without control bits, we apply constraints ONLY to residue pairs
    that are actually in contact on the lattice.
    
    This allows favorable MJ interactions (negative energy) to dominate
    while still penalizing bad geometries for actual contacts.
    
    Args:
        turns: List of turn codes.
        n_residues: Number of residues.
        lam_dis: Distance penalty weight.
        lam_loc: Locality penalty weight.
        positions: Pre-computed lattice positions (optional).
        contact_pairs: Pre-computed contact pairs (optional).
        
    Returns:
        Total geometric constraint energy.
    """
    if lam_dis == 0.0 and lam_loc == 0.0:
        return 0.0
    
    # If no contact pairs provided, we can't apply contact-based constraints
    if contact_pairs is None:
        return 0.0
    
    E = 0.0
    
    # Only apply constraints to residue pairs that are actually in contact
    for i, j in contact_pairs:
        # Skip if turns don't cover this range
        if j - 1 >= len(turns) or i >= len(turns):
            continue
        
        # Distance constraint: ||Δ_ij||² should be close to 1
        dij = np.linalg.norm(compute_delta_vec(turns, i, j)) ** 2
        E += lam_dis * abs(dij - 1.0)  # Use abs to ensure positive penalty
        
        # Locality constraint 1: direction from i to j-1
        if j - 1 > i:
            dir_ = np.linalg.norm(compute_delta_vec(turns, i, j - 1)) ** 2
            E += lam_loc * max(0, 2.0 - dir_)  # Ensure non-negative
        
        # Locality constraint 2: direction from i+1 to j
        if j > i + 1:
            dmj = np.linalg.norm(compute_delta_vec(turns, i + 1, j)) ** 2
            E += lam_loc * max(0, 2.0 - dmj)  # Ensure non-negative
    
    return E


def build_energy_table(
    fragment: FragmentSpec,
    mj_matrix: np.ndarray,
    config: FoldConfig,
    rg_weight: float = 0.0,
    target_rg: Optional[float] = None,
    include_geometric_constraints: bool = True,
) -> FragmentEnergyTable:
    """Precompute energy for all possible bitstrings of a fragment.
    
    Energy terms (matching original Hamiltonian):
    1. E_overlap = overlap_penalty * count_overlaps(positions)
    2. E_contact = Σ mj_matrix[i, j] for all (i, j) in contacts
    3. E_back = lamBack * (number of adjacent equal turns)
    4. E_geo = lamDis + lamLoc geometric constraints (if enabled)
    5. E_rg = rg_weight * (Rg - target_rg)² (optional compactness)
    
    Args:
        fragment: FragmentSpec defining the fragment.
        mj_matrix: MJ interaction matrix for the FULL sequence (use slice).
        config: FoldConfig with overlap_penalty, contact_cutoff, etc.
        rg_weight: Weight for radius of gyration term (0 = disabled).
        target_rg: Target Rg for compactness (default: sqrt(n_residues)).
        include_geometric_constraints: Include lamDis/lamLoc terms.
        
    Returns:
        FragmentEnergyTable with all energies precomputed.
        
    Example:
        >>> table = build_energy_table(fragment, mj, config)
        >>> table.energy_of_bitstring("0000000000")
        -2.5  # Example energy
    """
    n_bits = fragment.n_bits
    n_states = 2 ** n_bits
    n_residues = fragment.end_res - fragment.start_res
    
    # Default target Rg if not specified
    if target_rg is None:
        target_rg = np.sqrt(n_residues)
    
    # Extract MJ submatrix for this fragment
    mj_sub = mj_matrix[fragment.start_res:fragment.end_res,
                       fragment.start_res:fragment.end_res]
    
    # Get penalty weights from config
    lam_back = getattr(config, 'lam_back', DEFAULT_LAM_BACK)
    lam_dis = getattr(config, 'lam_dis', DEFAULT_LAM_DIS)
    lam_loc = getattr(config, 'lam_loc', DEFAULT_LAM_LOC)
    
    energies = np.zeros(n_states, dtype=np.float32)
    
    for idx in range(n_states):
        bitstring = index_to_bitstring(idx, n_bits)
        turns = decode_turns(bitstring)
        
        # Trace lattice positions
        positions = trace_positions(turns)
        
        # HARD CONSTRAINT: Self-intersection → forbidden state
        n_overlaps = count_overlaps(positions)
        if n_overlaps > 0:
            energies[idx] = np.inf  # FORBIDDEN - never select
            continue
        
        # Energy term 2: Contact interactions (MJ potential)
        contact_pairs = contacts(
            positions,
            min_sep=config.contact_min_sep,
            cutoff=config.contact_cutoff
        )
        e_contact = sum(mj_sub[i, j] for i, j in contact_pairs)
        
        # Energy term 3: Backbone turn variety penalty
        e_back = compute_backbone_penalty(turns, lam_back)
        
        # Energy term 4: Geometric constraints (only for contacting pairs)
        e_geo = 0.0
        if include_geometric_constraints and contact_pairs:
            e_geo = compute_geometric_constraints(
                turns, n_residues, lam_dis, lam_loc,
                positions=positions, contact_pairs=contact_pairs
            )
        
        # Energy term 5: Compactness (optional)
        e_rg = 0.0
        if rg_weight > 0:
            rg = radius_of_gyration(positions)
            e_rg = rg_weight * (rg - target_rg) ** 2
        
        energies[idx] = e_contact + e_back + e_geo + e_rg
    
    return FragmentEnergyTable(
        n_bits=n_bits,
        energies=energies,
        fragment_spec=fragment,
    )


def compute_single_energy(
    bitstring: str,
    fragment: FragmentSpec,
    mj_matrix: np.ndarray,
    config: FoldConfig,
    include_geometric_constraints: bool = True,
) -> float:
    """Compute energy for a single bitstring (not using table).
    
    Useful for verification or when table is not pre-built.
    
    Args:
        bitstring: Binary string encoding turns.
        fragment: FragmentSpec for this fragment.
        mj_matrix: Full MJ matrix.
        config: FoldConfig.
        include_geometric_constraints: Include lamDis/lamLoc terms.
        
    Returns:
        Energy value.
    """
    turns = decode_turns(bitstring)
    positions = trace_positions(turns)
    n_residues = fragment.end_res - fragment.start_res
    
    # Extract MJ submatrix
    mj_sub = mj_matrix[fragment.start_res:fragment.end_res,
                       fragment.start_res:fragment.end_res]
    
    # Get penalty weights from config
    lam_back = getattr(config, 'lam_back', DEFAULT_LAM_BACK)
    lam_dis = getattr(config, 'lam_dis', DEFAULT_LAM_DIS)
    lam_loc = getattr(config, 'lam_loc', DEFAULT_LAM_LOC)
    
    # Overlap penalty
    n_overlaps = count_overlaps(positions)
    e_overlap = config.overlap_penalty * n_overlaps
    
    # Contact energy
    contact_pairs = contacts(
        positions,
        min_sep=config.contact_min_sep,
        cutoff=config.contact_cutoff
    )
    e_contact = sum(mj_sub[i, j] for i, j in contact_pairs)
    
    # Backbone penalty
    e_back = compute_backbone_penalty(turns, lam_back)
    
    # Geometric constraints
    e_geo = 0.0
    if include_geometric_constraints:
        e_geo = compute_geometric_constraints(turns, n_residues, lam_dis, lam_loc)
    
    return e_overlap + e_contact + e_back + e_geo


def compute_chain_energy(
    bitstring: str,
    sequence: str,
    mj_matrix: np.ndarray,
    config: FoldConfig,
) -> float:
    """Compute energy for a full chain bitstring.
    
    This is used to compute the energy of the stitched chain,
    including proper overlap detection across ALL residues.
    
    Args:
        bitstring: Binary string encoding turns for full chain.
        sequence: Full amino acid sequence.
        mj_matrix: Full MJ matrix.
        config: FoldConfig.
        
    Returns:
        Total energy value.
    """
    turns = decode_turns(bitstring)
    n_residues = len(sequence)
    
    # Pad turns if needed (sequence length = turns + 1)
    while len(turns) < n_residues - 1:
        turns.append(0)  # Default straight
    
    # Trace positions for the full chain
    positions = trace_positions(turns)
    
    # Get penalty weights from config
    lam_back = getattr(config, 'lam_back', DEFAULT_LAM_BACK)
    
    # Overlap penalty - for the FULL chain
    n_overlaps = count_overlaps(positions)
    e_overlap = config.overlap_penalty * n_overlaps
    
    # Contact energy - MJ interactions
    contact_pairs = contacts(
        positions,
        min_sep=config.contact_min_sep,
        cutoff=config.contact_cutoff
    )
    e_contact = sum(mj_matrix[i, j] for i, j in contact_pairs)
    
    # Backbone penalty - adjacent equal turns
    e_back = compute_backbone_penalty(turns, lam_back)
    
    return e_overlap + e_contact + e_back

def compute_context_energy(
    fragment_turns: List[int],
    global_positions: np.ndarray,
    fragment_start_res: int,
    mj_matrix: np.ndarray,
    contact_min_sep: int = 2,
    contact_cutoff: float = 1.5
) -> float:
    """Compute interaction energy between fragment and fixed global environment.
    
    This anchors the fragment optimization to the global structure (QA/SA solution).
    
    Args:
        fragment_turns: List of turns for the candidate fragment.
        global_positions: Fixed positions of the ENTIRE chain (from SA/QA).
        fragment_start_res: Global index of the fragment's first residue.
        mj_matrix: Interaction matrix.
        
    Returns:
        Interaction energy (negative is favorable).
    """
    if len(global_positions) == 0:
        return 0.0

    # 1. Trace local positions (starts at 0,0,0)
    local_pos = trace_positions(turns=fragment_turns) 
    
    # 2. Align to global frame
    # We assume the start residue position is fixed to the global start residue
    # because we fix the left boundary (overlap) to match the global structure.
    if fragment_start_res >= len(global_positions):
        return 0.0
        
    offset = global_positions[fragment_start_res]
    aligned_pos = local_pos + offset
    
    # 3. Compute interactions with ENVIRONMENT
    # Environment = all residues NOT in this fragment
    n_frag = len(local_pos)
    frag_end_res = fragment_start_res + n_frag
    n_global = len(global_positions)
    
    E_context = 0.0
    
    # Optimization: Pre-calculate indices to avoid repeated checks
    frag_indices = set(range(fragment_start_res, frag_end_res))
    
    for i_local, pos_i in enumerate(aligned_pos):
        i_global = fragment_start_res + i_local
        
        # Check against all global residues j
        for j in range(n_global):
            # Skip if j is part of the fragment itself
            if j in frag_indices:
                continue
                
            # Skip neighbors in sequence (min_sep)
            if abs(i_global - j) < contact_min_sep:
                continue
            
            # Check distance
            dist = np.linalg.norm(pos_i - global_positions[j])
            if dist <= contact_cutoff:
                E_context += mj_matrix[i_global, j]
                
    return E_context
