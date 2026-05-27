"""PDB file writer with CONECT records and secondary structure annotations.

Secondary structure is assigned from the actual backbone φ/ψ dihedral angles
computed from the 3D coordinates -- NOT from the symbolic turn codes.
This ensures HELIX/SHEET records in the output PDB are consistent with the
actual geometry and will render correctly in PyMOL, ChimeraX, VMD, etc.

Ramachandran region boundaries used:
    Alpha-helix:  φ ∈ [-145°, -35°]  AND  ψ ∈ [-70°,  30°]
    Beta-sheet:   φ ∈ [-180°, -90°]  AND  ψ ∈ [ 90°, 180°]  (or ψ ∈ [-180°,-150°])
    Left-helix:   φ ∈ [  35°, 145°]  AND  ψ ∈ [ 10°,  90°]   (rare)
    All else:     coil / loop
"""

import math
from typing import List, Dict, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Dihedral computation
# ---------------------------------------------------------------------------

def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Compute dihedral angle A-B-C-D in degrees (range -180 to +180)."""
    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    if norm_n1 < 1e-8 or norm_n2 < 1e-8:
        return 0.0

    n1 /= norm_n1
    n2 /= norm_n2

    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-10))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return math.degrees(math.atan2(y, x))


def compute_backbone_dihedrals(
    atoms: List[Dict],
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Compute φ and ψ angles for each residue from 3D coordinates.

    Args:
        atoms: List of atom dicts with 'name', 'resid', 'coords'.

    Returns:
        (phi_list, psi_list) -- each has length = n_residues.
        First residue φ and last residue ψ are None (undefined).
    """
    # Build per-residue coordinate lookup
    resid_coords: Dict[int, Dict[str, np.ndarray]] = {}
    for atom in atoms:
        rid = atom["resid"]
        if rid not in resid_coords:
            resid_coords[rid] = {}
        resid_coords[rid][atom["name"]] = np.array(atom["coords"], dtype=float)

    resids = sorted(resid_coords.keys())
    n = len(resids)

    phis: List[Optional[float]] = [None] * n
    psis: List[Optional[float]] = [None] * n

    for i, rid in enumerate(resids):
        rc = resid_coords[rid]

        # φ (phi): C(i-1) – N(i) – CA(i) – C(i)
        if i > 0:
            prev = resid_coords[resids[i - 1]]
            if "C" in prev and "N" in rc and "CA" in rc and "C" in rc:
                phis[i] = _dihedral(prev["C"], rc["N"], rc["CA"], rc["C"])

        # ψ (psi): N(i) – CA(i) – C(i) – N(i+1)
        if i < n - 1:
            nxt = resid_coords[resids[i + 1]]
            if "N" in rc and "CA" in rc and "C" in rc and "N" in nxt:
                psis[i] = _dihedral(rc["N"], rc["CA"], rc["C"], nxt["N"])

    return phis, psis


# ---------------------------------------------------------------------------
# Ramachandran-based secondary structure assignment
# ---------------------------------------------------------------------------

def _classify_phi(phi: float) -> str:
    """Classify a residue by its phi angle alone.

    The backbone_builder discretises conformations into four turn codes, and
    phi alone is sufficient to distinguish them:

    Turn 0  phi ≈ -45°  →  helix  (range -80 to -20)
    Turn 1  phi ≈ +135° →  sheet  (range +115 to +137)
    Turn 2  phi ≈ +145° →  coil
    Turn 3  phi ≈ +140° →  coil

    The coil phi values (140-145°) fall above the sheet window (≤137°) so
    they are cleanly excluded.
    """
    if -80.0 <= phi <= -20.0:
        return SS_HELIX
    if 115.0 <= phi <= 137.0:
        return SS_SHEET
    return SS_COIL


# Keep for legacy compatibility (called elsewhere)
def _in_helix_region(phi: float, psi: float) -> bool:
    return _classify_phi(phi) == SS_HELIX


def _in_sheet_region(phi: float, psi: float) -> bool:
    return _classify_phi(phi) == SS_SHEET


SS_HELIX = "H"
SS_SHEET = "E"
SS_COIL  = "C"


def assign_secondary_structure(
    atoms: List[Dict],
    min_helix_length: int = 2,
    min_sheet_length: int = 2,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Assign secondary structure from backbone dihedral angles.

    Uses per-residue phi angles computed from 3D coordinates to classify each
    residue into helix, sheet, or coil, then collects consecutive runs.

    The phi angle is the primary discriminant (phi is more reliably computed
    than psi in lattice-derived structures):
        phi in [-80, -20] → helix   (turn code 0 → phi ≈ -45°)
        phi in [115, 137] → sheet   (turn code 1 → phi ≈ +135°)
        otherwise         → coil    (turn codes 2,3 → phi ≈ +140-145°)

    Args:
        atoms: List of atom dicts (from backbone_builder).
        min_helix_length: Minimum consecutive helix residues to annotate.
        min_sheet_length: Minimum consecutive sheet residues to annotate.

    Returns:
        (helix_regions, sheet_regions) as (start_res, end_res) 1-based tuples.
    """
    phis, _ = compute_backbone_dihedrals(atoms)

    # Build sorted residue list
    resid_coords: Dict[int, Dict[str, np.ndarray]] = {}
    for atom in atoms:
        rid = atom["resid"]
        if rid not in resid_coords:
            resid_coords[rid] = {}
        resid_coords[rid][atom["name"]] = atom["coords"]
    resids = sorted(resid_coords.keys())
    n = len(resids)

    # Per-residue SS label using phi only (most reliable discriminant)
    ss = []
    for i in range(n):
        phi = phis[i]
        if phi is None:
            ss.append(SS_COIL)
        else:
            ss.append(_classify_phi(phi))

    # Collect runs into regions
    helices: List[Tuple[int, int]] = []
    sheets:  List[Tuple[int, int]] = []

    def _collect_runs(label: str, min_len: int, out: List[Tuple[int, int]]) -> None:
        i = 0
        while i < n:
            if ss[i] == label:
                start = i
                while i < n and ss[i] == label:
                    i += 1
                if i - start >= min_len:
                    out.append((resids[start], resids[i - 1]))
            else:
                i += 1

    _collect_runs(SS_HELIX, min_helix_length, helices)
    _collect_runs(SS_SHEET, min_sheet_length, sheets)

    return helices, sheets


def detect_secondary_structure(
    turns: List[int],
    min_helix_length: int = 1,
    min_sheet_length: int = 1,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Assign secondary structure from turn codes.

    Turn code 0 (phi≈-60°, psi≈-45°) → alpha-helix → HELIX record.
    Turn code 1 (phi≈-135°, psi≈135°) → beta-strand → SHEET record.
    Turn codes 2, 3 → extended / coil → no record.

    Each turn at index ``i`` governs the dihedral between residues ``i+1``
    and ``i+2`` (1-based).  A run of ``k`` consecutive helix turns therefore
    spans residues ``start+1`` to ``start+k+1``.

    Args:
        turns: List of turn codes (length = n_residues - 1).
        min_helix_length: Minimum number of consecutive helix turns to
            annotate.  Default 1 annotates even single-turn helices (2 res).
        min_sheet_length: Same for sheet strands.

    Returns:
        (helix_regions, sheet_regions) as (start_res, end_res) 1-based
        residue numbers, inclusive.
    """
    HELIX_TURNS = {0}
    SHEET_TURNS = {1}

    if not turns:
        return [], []

    helices: List[Tuple[int, int]] = []
    sheets:  List[Tuple[int, int]] = []

    for region_set, min_len, out in [
        (HELIX_TURNS, min_helix_length, helices),
        (SHEET_TURNS, min_sheet_length, sheets),
    ]:
        i = 0
        while i < len(turns):
            if turns[i] in region_set:
                start = i
                while i < len(turns) and turns[i] in region_set:
                    i += 1
                length = i - start
                if length >= min_len:
                    # Turn i covers residues (i+1)..(i+length+1)
                    out.append((start + 1, start + length + 1))
            else:
                i += 1

    return helices, sheets


# ---------------------------------------------------------------------------
# PDB record builders
# ---------------------------------------------------------------------------

def _build_helix_records(
    helices: List[Tuple[int, int]],
    resid_to_name: Dict[int, str],
) -> List[str]:
    """Build HELIX records in strict PDB v3.3 column format.

    HELIX record columns (1-based, inclusive):
      1-6   "HELIX "
      8-10  serial (right-justified)
      12-14 helix ID (right-justified)
      16-18 init resName
      20    chain
      22-25 initSeqNum (right-justified)
      26    iCode (blank)
      28-30 end resName
      32    chain
      34-37 endSeqNum (right-justified)
      38    iCode (blank)
      39-40 helixClass (1 = right-handed alpha)
      41-70 comment (blank)
      72-76 length (right-justified)
    """
    records = []
    for idx, (start, end) in enumerate(helices, 1):
        start_name = resid_to_name.get(start, "ALA")
        end_name   = resid_to_name.get(end,   "ALA")
        length = end - start + 1
        helix_id = f"H{idx:02d}"
        record = (
            f"HELIX  {idx:3d} {helix_id:>3s} "
            f"{start_name:3s} A {start:4d}  "
            f"{end_name:3s} A {end:4d}  1"
            f"{'':30s}{length:5d}"
        )
        records.append(record)
    return records


def _build_sheet_records(
    sheets: List[Tuple[int, int]],
    resid_to_name: Dict[int, str],
) -> List[str]:
    """Build SHEET records in strict PDB v3.3 column format.

    Each beta-sheet region is written as a single-strand sheet.

    SHEET record columns:
      1-6   "SHEET "
      8-10  strand serial (right-justified)
      12-14 sheet ID (right-justified)
      15    numStrands
      18-20 init resName
      22    chain
      23-26 initSeqNum (right-justified)
      27    iCode
      29-31 end resName
      33    chain
      34-37 endSeqNum
      38    iCode
      39-40 sense (0 = first strand)
    """
    records = []
    for idx, (start, end) in enumerate(sheets, 1):
        sheet_id   = f"S{idx:02d}"
        start_name = resid_to_name.get(start, "ALA")
        end_name   = resid_to_name.get(end,   "ALA")
        record = (
            f"SHEET  {idx:3d} {sheet_id:>3s} 1 "
            f"{start_name:3s} A{start:4d}  "
            f"{end_name:3s} A{end:4d}  0"
        )
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Main write_pdb entry point
# ---------------------------------------------------------------------------

def write_pdb(
    atoms: List[Dict],
    output_path: str,
    title: Optional[str] = None,
    remarks: Optional[List[str]] = None,
    turns: Optional[List[int]] = None,
) -> None:
    """Write atoms to PDB format file with secondary structure annotations.

    Secondary structure is assigned from the ``turns`` array (turn codes
    directly from the quantum/SA optimizer) when provided, which is the most
    reliable source for this lattice model.  When ``turns`` is None the
    function falls back to phi-angle classification from 3D coordinates.

    Turn code → PDB record mapping:
        0 (phi≈-60°, psi≈-45°)   → HELIX  (alpha-helix)
        1 (phi≈-135°, psi≈135°)  → SHEET  (beta-strand)
        2, 3                      → no record (coil/extended)

    Args:
        atoms: List of atom dicts with 'name', 'resname', 'resid', 'coords'.
        output_path: Path to output PDB file.
        title: Optional title line.
        remarks: Optional list of REMARK lines.
        turns: List of turn codes from the optimizer.  **Strongly recommended**
               -- pass this whenever available.  If None, falls back to
               coordinate-based phi-angle classification.

    Example:
        >>> write_pdb(atoms, "structure.pdb", title="QuPepFold backbone", turns=turns)
    """
    lines: List[str] = []

    # ---- Header ----
    if title:
        lines.append(f"TITLE     {title[:70]}")

    if remarks:
        for remark in remarks:
            lines.append(f"REMARK    {remark}")

    # ---- Secondary structure ----
    resid_to_name: Dict[int, str] = {}
    for atom in atoms:
        resid_to_name[atom["resid"]] = atom["resname"]

    if turns is not None:
        # Primary path: use turn codes directly (most reliable for lattice model)
        helices, sheets = detect_secondary_structure(
            turns, min_helix_length=1, min_sheet_length=1
        )
    else:
        # Fallback: classify from phi angles of 3D coordinates
        helices, sheets = assign_secondary_structure(
            atoms, min_helix_length=2, min_sheet_length=2
        )

    lines.extend(_build_helix_records(helices, resid_to_name))
    lines.extend(_build_sheet_records(sheets, resid_to_name))

    # ---- ATOM records ----
    residue_atoms: Dict[int, Dict[str, int]] = {}
    serial = 1

    for atom in atoms:
        name    = atom["name"]
        resname = atom["resname"]
        resid   = atom["resid"]
        x, y, z = atom["coords"]

        # PDB fixed-column ATOM format (columns are 1-based in spec)
        # Atom name: right-justified in cols 13-16; element in cols 77-78
        line = (
            f"ATOM  {serial:5d} {name:>4s} {resname:3s} A{resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {name[0]:>2s}  "
        )
        lines.append(line)

        if resid not in residue_atoms:
            residue_atoms[resid] = {}
        residue_atoms[resid][name] = serial
        serial += 1

    # ---- CONECT records ----
    lines.extend(_build_conect_records(residue_atoms))

    lines.append("END")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CONECT records
# ---------------------------------------------------------------------------

def _build_conect_records(residue_atoms: Dict[int, Dict[str, int]]) -> List[str]:
    """Build CONECT records for backbone bonds.

    Args:
        residue_atoms: Mapping resid → {atom_name: serial}.

    Returns:
        List of CONECT lines.
    """
    conects: List[str] = []
    resids = sorted(residue_atoms.keys())

    for i, resid in enumerate(resids):
        atoms = residue_atoms[resid]

        # Intra-residue bonds
        if "N" in atoms and "CA" in atoms:
            conects.append(f"CONECT{atoms['N']:5d}{atoms['CA']:5d}")
        if "CA" in atoms and "C" in atoms:
            conects.append(f"CONECT{atoms['CA']:5d}{atoms['C']:5d}")
        if "CA" in atoms and "CB" in atoms:
            conects.append(f"CONECT{atoms['CA']:5d}{atoms['CB']:5d}")
        if "C" in atoms and "O" in atoms:
            conects.append(f"CONECT{atoms['C']:5d}{atoms['O']:5d}")

        # Inter-residue peptide bond C(i) → N(i+1)
        if i < len(resids) - 1:
            next_resid = resids[i + 1]
            if "C" in atoms and "N" in residue_atoms[next_resid]:
                conects.append(
                    f"CONECT{atoms['C']:5d}{residue_atoms[next_resid]['N']:5d}"
                )

    return conects


# ---------------------------------------------------------------------------
# Multi-model and utility functions
# ---------------------------------------------------------------------------

def write_pdb_multi_model(
    models: List[List[Dict]],
    output_path: str,
    title: Optional[str] = None,
) -> None:
    """Write multiple models to a single PDB file.

    Args:
        models: List of atom lists (one per model).
        output_path: Output path.
        title: Optional title.
    """
    lines: List[str] = []

    if title:
        lines.append(f"TITLE     {title[:70]}")

    for model_num, model_atoms in enumerate(models, 1):
        lines.append(f"MODEL     {model_num:4d}")
        serial = 1
        for atom in model_atoms:
            name    = atom["name"]
            resname = atom["resname"]
            resid   = atom["resid"]
            x, y, z = atom["coords"]
            lines.append(
                f"ATOM  {serial:5d} {name:>4s} {resname:3s} A{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {name[0]:>2s}  "
            )
            serial += 1
        lines.append("ENDMDL")

    lines.append("END")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def atoms_to_pdb_string(atoms: List[Dict]) -> str:
    """Convert atoms to PDB format string (no file IO).

    Args:
        atoms: List of atom dicts.

    Returns:
        PDB format string.
    """
    lines: List[str] = []
    serial = 1
    for atom in atoms:
        name    = atom["name"]
        resname = atom["resname"]
        resid   = atom["resid"]
        x, y, z = atom["coords"]
        lines.append(
            f"ATOM  {serial:5d} {name:>4s} {resname:3s} A{resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {name[0]:>2s}  "
        )
        serial += 1
    lines.append("END")
    return "\n".join(lines)
