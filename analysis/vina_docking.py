"""
AutoDock Vina docking utilities for evaluating generated ligands.

This module provides clean interfaces for:
1. Scoring ligands against receptors (score_only mode)
2. Full docking with pose generation
3. Batch processing of multiple ligands

Requirements:
- AutoDock Vina binary available in PATH (tested with v1.2.7)
- OpenBabel for file format conversions (via RDKit)
"""

import os
import re
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
from rdkit import Chem


def prepare_receptor(pdb_file: Path, output_pdbqt: Optional[Path] = None) -> Path:
    """
    Prepare receptor PDB file for docking by converting to PDBQT format.

    Args:
        pdb_file: Path to receptor PDB file
        output_pdbqt: Optional output path. If None, creates temp file.

    Returns:
        Path to prepared PDBQT file

    Note:
        This is a placeholder. For production use, implement proper receptor
        preparation using MGLTools' prepare_receptor4.py or similar.
        Currently assumes receptor is already in PDBQT format.
    """
    if pdb_file.suffix == '.pdbqt':
        return pdb_file

    if output_pdbqt is None:
        output_pdbqt = pdb_file.with_suffix('.pdbqt')

    # TODO: Implement proper receptor preparation
    # For now, assume receptor is already prepared
    raise NotImplementedError(
        "Receptor preparation not implemented. "
        "Please provide receptor in PDBQT format or prepare using MGLTools."
    )


def sdf_to_pdbqt(sdf_file: Path, pdbqt_file: Path, mol_id: int = 0) -> Path:
    """
    Convert SDF molecule to PDBQT format using OpenBabel.

    Args:
        sdf_file: Input SDF file
        pdbqt_file: Output PDBQT file
        mol_id: Molecule index in SDF file (0-based)

    Returns:
        Path to created PDBQT file
    """
    # OpenBabel uses 1-based indexing for -f and -l flags
    cmd = f'obabel {sdf_file} -O {pdbqt_file} -f {mol_id + 1} -l {mol_id + 1}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"OpenBabel conversion failed: {result.stderr}")

    return pdbqt_file


def rdmol_to_pdbqt(rdmol: Chem.Mol, pdbqt_file: Path) -> Path:
    """
    Convert RDKit molecule to PDBQT format via temporary SDF.

    Args:
        rdmol: RDKit molecule object
        pdbqt_file: Output PDBQT file path

    Returns:
        Path to created PDBQT file
    """
    with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp:
        tmp_sdf = Path(tmp.name)

    try:
        # Write RDKit mol to SDF
        writer = Chem.SDWriter(str(tmp_sdf))
        writer.write(rdmol)
        writer.close()

        # Convert SDF to PDBQT
        sdf_to_pdbqt(tmp_sdf, pdbqt_file, mol_id=0)
    finally:
        tmp_sdf.unlink(missing_ok=True)

    return pdbqt_file


def parse_vina_output(output: str) -> List[float]:
    """
    Parse AutoDock Vina output to extract binding affinity scores.

    Args:
        output: Raw stdout from vina command

    Returns:
        List of binding affinity scores in kcal/mol (lower is better)

    Example output formats:
        1. Docking mode:
            mode |   affinity | dist from best mode
                 | (kcal/mol) | rmsd l.b.| rmsd u.b.
            -----+------------+----------+----------
               1       -8.9      0.000      0.000
               2       -8.7      1.234      2.345

        2. Score-only mode:
            Estimated Free Energy of Binding   : -8.397 (kcal/mol)
    """
    scores = []
    lines = output.split('\n')

    # Try to parse score_only format first
    for line in lines:
        if 'Estimated Free Energy of Binding' in line:
            # Line format: "Estimated Free Energy of Binding   : -8.397 (kcal/mol)"
            match = re.search(r':\s*([+-]?\d+\.?\d*)\s*\(kcal/mol\)', line)
            if match:
                scores.append(float(match.group(1)))
                return scores

    # Try to parse docking table format
    in_table = False
    for line in lines:
        if '-----+------------+----------+----------' in line:
            in_table = True
            continue

        if in_table:
            # Parse lines like: "   1       -8.9      0.000      0.000"
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mode = int(parts[0])
                    affinity = float(parts[1])
                    scores.append(affinity)
                except (ValueError, IndexError):
                    # End of table or invalid line
                    break

    return scores


def calculate_vina_score(
    receptor_file: Path,
    ligand_file: Path,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float] = (20, 20, 20),
    exhaustiveness: int = 8,
    num_modes: int = 1,
    score_only: bool = False,
    output_file: Optional[Path] = None
) -> Union[float, List[float]]:
    """
    Calculate binding affinity using AutoDock Vina.

    Args:
        receptor_file: Receptor in PDBQT format
        ligand_file: Ligand in PDBQT format
        center: (x, y, z) coordinates for box center in Angstroms
        box_size: (x, y, z) dimensions of search box in Angstroms
        exhaustiveness: Search exhaustiveness (higher = more thorough but slower)
        num_modes: Number of binding modes to generate
        score_only: If True, only score without docking (much faster)
        output_file: Optional path to save docked poses (PDBQT format)

    Returns:
        If score_only: Single binding affinity score (kcal/mol)
        Otherwise: List of scores for all modes (sorted by affinity)

    Note:
        Scores are in kcal/mol. More negative = better binding.
        Typical good scores: -8 to -12 kcal/mol for drug-like molecules.
    """
    # Build Vina command
    cmd = [
        'vina',
        '--receptor', str(receptor_file),
        '--ligand', str(ligand_file),
        '--center_x', f'{center[0]:.4f}',
        '--center_y', f'{center[1]:.4f}',
        '--center_z', f'{center[2]:.4f}',
        '--size_x', f'{box_size[0]:.1f}',
        '--size_y', f'{box_size[1]:.1f}',
        '--size_z', f'{box_size[2]:.1f}',
    ]

    if score_only:
        cmd.append('--score_only')
    else:
        cmd.extend([
            '--exhaustiveness', str(exhaustiveness),
            '--num_modes', str(num_modes),
        ])

        if output_file is not None:
            cmd.extend(['--out', str(output_file)])

    # Run Vina
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Vina failed with return code {result.returncode}\n"
            f"STDERR: {result.stderr}\n"
            f"STDOUT: {result.stdout}"
        )

    # Parse scores from output
    scores = parse_vina_output(result.stdout)

    if not scores:
        # Fallback: try to parse from PDBQT output file if it exists
        if output_file and output_file.exists():
            scores = parse_pdbqt_scores(output_file)

    if not scores:
        raise ValueError(f"Could not parse scores from Vina output:\n{result.stdout}")

    # Return single score for score_only mode, list otherwise
    return scores[0] if score_only else scores


def parse_pdbqt_scores(pdbqt_file: Path) -> List[float]:
    """
    Extract binding affinity scores from PDBQT output file.

    Args:
        pdbqt_file: PDBQT file with docked poses

    Returns:
        List of binding affinity scores
    """
    scores = []

    with open(pdbqt_file, 'r') as f:
        for line in f:
            # Look for lines like: "REMARK VINA RESULT:   -8.9      0.000      0.000"
            if line.startswith('REMARK VINA RESULT:'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        score = float(parts[3])
                        scores.append(score)
                    except ValueError:
                        continue

    return scores


def dock_molecules(
    rdmols: List[Chem.Mol],
    receptor_file: Path,
    center: Optional[Tuple[float, float, float]] = None,
    box_size: Tuple[float, float, float] = (20, 20, 20),
    exhaustiveness: int = 8,
    score_only: bool = True,
    cleanup: bool = True
) -> List[float]:
    """
    Dock multiple RDKit molecules and return binding affinities.

    Args:
        rdmols: List of RDKit molecule objects
        receptor_file: Receptor PDBQT file
        center: Box center coordinates. If None, use centroid of first molecule.
        box_size: Search box dimensions
        exhaustiveness: Docking thoroughness
        score_only: If True, only score (faster, no conformational search)
        cleanup: Remove temporary files

    Returns:
        List of binding affinity scores (kcal/mol), one per molecule
        Returns np.nan for molecules that fail docking.
    """
    scores = []

    for i, mol in enumerate(rdmols):
        # Determine box center from molecule if not provided
        if center is None:
            conf = mol.GetConformer()
            positions = conf.GetPositions()
            mol_center = tuple(positions.mean(axis=0))
        else:
            mol_center = center

        # Create temporary PDBQT file for ligand
        with tempfile.NamedTemporaryFile(
            suffix='.pdbqt', delete=False
        ) as tmp:
            ligand_pdbqt = Path(tmp.name)

        try:
            # Convert molecule to PDBQT
            rdmol_to_pdbqt(mol, ligand_pdbqt)

            # Run docking
            score = calculate_vina_score(
                receptor_file=receptor_file,
                ligand_file=ligand_pdbqt,
                center=mol_center,
                box_size=box_size,
                exhaustiveness=exhaustiveness,
                score_only=score_only
            )

            scores.append(score)

        except Exception as e:
            print(f"Warning: Docking failed for molecule {i}: {e}")
            scores.append(np.nan)

        finally:
            if cleanup:
                ligand_pdbqt.unlink(missing_ok=True)

    return scores


# Convenience function matching existing API pattern
def vina_score(
    rdmols: Union[Chem.Mol, List[Chem.Mol]],
    receptor_file: Union[str, Path],
    **kwargs
) -> Union[float, List[float]]:
    """
    Calculate Vina scores for RDKit molecules (convenience wrapper).

    Args:
        rdmols: Single RDKit molecule or list of molecules
        receptor_file: Path to receptor PDBQT file
        **kwargs: Additional arguments passed to dock_molecules()

    Returns:
        Single score or list of scores

    Example:
        >>> mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')  # Aspirin
        >>> score = vina_score(mol, 'receptor.pdbqt')
        >>> print(f"Binding affinity: {score:.2f} kcal/mol")
    """
    receptor_file = Path(receptor_file)

    # Handle single molecule
    if isinstance(rdmols, Chem.Mol):
        rdmols = [rdmols]
        single_mol = True
    else:
        single_mol = False

    scores = dock_molecules(rdmols, receptor_file, **kwargs)

    return scores[0] if single_mol else scores


if __name__ == '__main__':
    """
    Test the docking implementation with example files.
    """
    import sys

    # Test with example files if available
    example_dir = Path(__file__).parent.parent / 'example'

    if not example_dir.exists():
        print("Example directory not found. Skipping tests.")
        sys.exit(0)

    # Look for test files
    receptor = example_dir / '3rfm.pdbqt'
    ligand_sdf = example_dir / '3rfm_mol.sdf'

    if not receptor.exists():
        print(f"Receptor not found: {receptor}")
        sys.exit(1)

    if not ligand_sdf.exists():
        print(f"Ligand not found: {ligand_sdf}")
        sys.exit(1)

    print(f"Testing Vina docking with:")
    print(f"  Receptor: {receptor}")
    print(f"  Ligand: {ligand_sdf}")
    print()

    # Load ligand
    suppl = Chem.SDMolSupplier(str(ligand_sdf), sanitize=False)
    mols = [mol for mol in suppl if mol is not None]

    if not mols:
        print("No valid molecules in SDF file")
        sys.exit(1)

    print(f"Loaded {len(mols)} molecule(s)")

    # Test score_only mode (fast)
    print("\nTesting score_only mode...")
    scores = vina_score(
        mols[0],
        receptor,
        center=(0, 0, 0),
        box_size=(25, 25, 25),
        score_only=True
    )
    print(f"Score: {scores:.3f} kcal/mol")

    print("\nTests completed successfully!")
