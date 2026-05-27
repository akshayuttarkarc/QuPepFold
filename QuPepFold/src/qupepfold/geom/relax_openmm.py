"""OpenMM-based structure relaxation.

Uses PDBFixer to add missing atoms and OpenMM for energy minimization.
"""

from typing import Optional, Tuple
import os


def relax_structure(
    input_pdb: str,
    output_pdb: str,
    force_field: str = "amber14-all",
    max_iterations: int = 500,
    add_hydrogens: bool = True,
    add_missing_atoms: bool = True,
) -> Tuple[float, float]:
    """Relax structure using OpenMM energy minimization.
    
    Args:
        input_pdb: Path to input PDB file.
        output_pdb: Path to output relaxed PDB file.
        force_field: Force field name ('amber14-all' or 'charmm36').
        max_iterations: Maximum minimization iterations.
        add_hydrogens: Whether to add missing hydrogens.
        add_missing_atoms: Whether to add missing heavy atoms.
        
    Returns:
        Tuple of (initial_energy, final_energy) in kJ/mol.
        
    Raises:
        ImportError: If OpenMM or PDBFixer not installed.
    """
    try:
        from openmm import app, unit, LangevinMiddleIntegrator
        from openmm import Platform
        import openmm
    except ImportError as e:
        raise ImportError(
            "OpenMM is required for structure relaxation. "
            "Install with: conda install -c conda-forge openmm"
        ) from e
    
    try:
        from pdbfixer import PDBFixer
    except ImportError as e:
        raise ImportError(
            "PDBFixer is required for structure relaxation. "
            "Install with: conda install -c conda-forge pdbfixer"
        ) from e
    
    # Fix structure
    fixer = PDBFixer(filename=input_pdb)
    
    if add_missing_atoms:
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
    
    if add_hydrogens:
        fixer.addMissingHydrogens(7.0)  # pH 7.0
    
    # Select force field
    if force_field == "amber14-all":
        ff = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    elif force_field == "charmm36":
        ff = app.ForceField('charmm36.xml', 'charmm36/water.xml')
    else:
        ff = app.ForceField(force_field + '.xml')
    
    # Build system (implicit solvent for speed)
    modeller = app.Modeller(fixer.topology, fixer.positions)
    
    try:
        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
        )
    except Exception:
        # Fallback without constraints
        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
        )
    
    # Integrator (not used, but required)
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picoseconds,
        0.002 * unit.picoseconds,
    )
    
    # Create simulation
    platform = Platform.getPlatformByName('CPU')
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    
    # Get initial energy
    state = simulation.context.getState(getEnergy=True)
    initial_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    
    # Minimize
    simulation.minimizeEnergy(maxIterations=max_iterations)
    
    # Get final energy
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    final_positions = state.getPositions()
    
    # Write output
    with open(output_pdb, 'w') as f:
        app.PDBFile.writeFile(simulation.topology, final_positions, f)
    
    return initial_energy, final_energy


def relax_structure_simple(
    input_pdb: str,
    output_pdb: str,
    max_iterations: int = 100,
) -> bool:
    """Simplified relaxation without OpenMM dependencies.
    
    Falls back to just copying the file if OpenMM is not available.
    
    Args:
        input_pdb: Input PDB path.
        output_pdb: Output PDB path.
        max_iterations: Minimization iterations (if available).
        
    Returns:
        True if relaxation was performed, False if fallback copy.
    """
    try:
        relax_structure(input_pdb, output_pdb, max_iterations=max_iterations)
        return True
    except ImportError:
        # Fallback: just copy
        import shutil
        shutil.copy(input_pdb, output_pdb)
        return False


def is_openmm_available() -> bool:
    """Check if OpenMM is available."""
    try:
        import openmm  # noqa
        from pdbfixer import PDBFixer  # noqa
        return True
    except ImportError:
        return False
