"""
Example: How to modify process_crossdock.py to track residue IDs

This demonstrates the required changes to track atom→residue mappings.
"""

import numpy as np
from typing import List, Tuple, Dict


class MockResidue:
    """Mock Bio.PDB.Residue for demonstration"""
    def __init__(self, resname, atoms):
        self.resname = resname
        self.atoms = atoms
        self.id = (None, 1, None)

    def get_atoms(self):
        return self.atoms

    def get_resname(self):
        return self.resname


class MockAtom:
    """Mock Bio.PDB.Atom for demonstration"""
    def __init__(self, element, coord):
        self.element = element
        self.coord = np.array(coord)

    def get_coord(self):
        return self.coord


def extract_pocket_current(pocket_residues, amino_acid_dict):
    """
    CURRENT implementation (NO residue ID tracking)
    This is what process_crossdock.py does now.
    """
    print("="*80)
    print("CURRENT IMPLEMENTATION (NO RESIDUE TRACKING)")
    print("="*80)

    # Current code from process_crossdock.py lines 105-118
    full_atoms = np.concatenate([
        np.array([atom.element for atom in res.get_atoms()])
        for res in pocket_residues
    ], axis=0)

    full_coords = np.concatenate([
        np.array([atom.coord for atom in res.get_atoms()])
        for res in pocket_residues
    ], axis=0)

    print(f"Extracted atoms: {full_atoms}")
    print(f"Coordinates shape: {full_coords.shape}")
    print(f"\n⚠️ PROBLEM: We lost track of which atoms belong to which residues!")
    print(f"   Cannot map ESM-C residue embeddings to atoms!\n")

    return full_atoms, full_coords, None


def extract_pocket_modified(pocket_residues, amino_acid_dict):
    """
    MODIFIED implementation (WITH residue ID tracking)
    This is what we need to implement.
    """
    print("="*80)
    print("MODIFIED IMPLEMENTATION (WITH RESIDUE TRACKING)")
    print("="*80)

    # Modified approach: track residue IDs per atom
    full_atoms = []
    full_coords = []
    residue_ids = []

    for res_idx, res in enumerate(pocket_residues):
        for atom in res.get_atoms():
            full_atoms.append(atom.element)
            full_coords.append(atom.coord)
            residue_ids.append(res_idx)  # Track which residue this atom belongs to

    full_atoms = np.array(full_atoms)
    full_coords = np.array(full_coords)
    residue_ids = np.array(residue_ids, dtype=np.int32)

    print(f"Extracted atoms: {full_atoms}")
    print(f"Residue IDs: {residue_ids}")
    print(f"Coordinates shape: {full_coords.shape}")
    print(f"\n✓ SUCCESS: We know which residue each atom belongs to!")
    print(f"   Can now broadcast ESM-C embeddings to atoms!\n")

    # Verify residue ID mapping
    print("Verification:")
    for res_idx, res in enumerate(pocket_residues):
        atom_count = np.sum(residue_ids == res_idx)
        print(f"  Residue {res_idx} ({res.resname}): {atom_count} atoms → "
              f"IDs {np.where(residue_ids == res_idx)[0].tolist()}")

    return full_atoms, full_coords, residue_ids


def broadcast_esmc_embeddings(esmc_residue_embeddings, residue_ids):
    """
    Demonstrate ESM-C broadcasting with tracked residue IDs
    """
    print("\n" + "="*80)
    print("ESM-C BROADCASTING DEMONSTRATION")
    print("="*80)

    n_residues = esmc_residue_embeddings.shape[0]
    esmc_dim = esmc_residue_embeddings.shape[1]
    n_atoms = len(residue_ids)

    print(f"\nInput:")
    print(f"  ESM-C embeddings: {esmc_residue_embeddings.shape} ({n_residues} residues × {esmc_dim} features)")
    print(f"  Residue IDs: {residue_ids.shape} ({n_atoms} atoms)")

    # Simple broadcasting via indexing
    esmc_atom_embeddings = esmc_residue_embeddings[residue_ids]

    print(f"\nOutput:")
    print(f"  Broadcasted embeddings: {esmc_atom_embeddings.shape} ({n_atoms} atoms × {esmc_dim} features)")

    # Verify broadcasting
    print(f"\nVerification:")
    for res_idx in range(n_residues):
        atom_indices = np.where(residue_ids == res_idx)[0]
        if len(atom_indices) > 0:
            # Check all atoms from same residue have identical embeddings
            embeddings_equal = np.allclose(
                esmc_atom_embeddings[atom_indices],
                esmc_residue_embeddings[res_idx]
            )
            print(f"  Residue {res_idx}: {len(atom_indices)} atoms, "
                  f"embeddings equal: {embeddings_equal}")

    return esmc_atom_embeddings


def demonstrate_full_pipeline():
    """
    Full demonstration of the workflow
    """
    print("\n" + "="*80)
    print("FULL ESM-C INTEGRATION WORKFLOW")
    print("="*80 + "\n")

    # Mock data: 3 residues (GLY, ALA, SER)
    amino_acid_dict = {'C': 0, 'N': 1, 'O': 2, 'S': 3}

    pocket_residues = [
        # Residue 0: GLY (4 atoms: N, CA, C, O)
        MockResidue('GLY', [
            MockAtom('N', [0.0, 0.0, 0.0]),
            MockAtom('C', [1.0, 0.0, 0.0]),  # CA
            MockAtom('C', [2.0, 0.0, 0.0]),  # C
            MockAtom('O', [3.0, 0.0, 0.0]),
        ]),
        # Residue 1: ALA (5 atoms: N, CA, C, O, CB)
        MockResidue('ALA', [
            MockAtom('N', [4.0, 0.0, 0.0]),
            MockAtom('C', [5.0, 0.0, 0.0]),  # CA
            MockAtom('C', [6.0, 0.0, 0.0]),  # C
            MockAtom('O', [7.0, 0.0, 0.0]),
            MockAtom('C', [5.0, 1.0, 0.0]),  # CB
        ]),
        # Residue 2: SER (6 atoms: N, CA, C, O, CB, OG)
        MockResidue('SER', [
            MockAtom('N', [8.0, 0.0, 0.0]),
            MockAtom('C', [9.0, 0.0, 0.0]),  # CA
            MockAtom('C', [10.0, 0.0, 0.0]),  # C
            MockAtom('O', [11.0, 0.0, 0.0]),
            MockAtom('C', [9.0, 1.0, 0.0]),  # CB
            MockAtom('O', [9.0, 2.0, 0.0]),  # OG
        ]),
    ]

    print("Input: 3 residues")
    for i, res in enumerate(pocket_residues):
        print(f"  Residue {i}: {res.resname} ({len(list(res.get_atoms()))} atoms)")
    print()

    # Step 1: Current approach (fails)
    atoms_curr, coords_curr, ids_curr = extract_pocket_current(pocket_residues, amino_acid_dict)

    # Step 2: Modified approach (succeeds)
    atoms_mod, coords_mod, ids_mod = extract_pocket_modified(pocket_residues, amino_acid_dict)

    # Step 3: Generate mock ESM-C embeddings
    n_residues = len(pocket_residues)
    esmc_dim = 960
    esmc_embeddings = np.random.randn(n_residues, esmc_dim).astype(np.float32)

    print("\n" + "="*80)
    print("ESM-C EMBEDDING GENERATION (MOCKED)")
    print("="*80)
    print(f"Generated {n_residues} residue embeddings (shape: {esmc_embeddings.shape})")

    # Step 4: Broadcast to atoms
    esmc_atom = broadcast_esmc_embeddings(esmc_embeddings, ids_mod)

    # Step 5: Show final data structure
    print("\n" + "="*80)
    print("FINAL NPZ STRUCTURE")
    print("="*80)
    print("""
Would save to NPZ as:
{
    'pocket_coords': (n_atoms, 3),           # Atom coordinates
    'pocket_one_hot': (n_atoms, 11),         # Atom type one-hot
    'pocket_mask': (n_atoms,),               # Batch indices
    'pocket_residue_ids': (n_atoms,),        # NEW: atom→residue mapping
    'pocket_esmc': (n_atoms, 960),           # NEW: broadcasted ESM-C
    'pocket_ids': List[str],                 # Residue identifiers (e.g., "A:123")
    ...
}
""")

    print(f"Example for first sample:")
    print(f"  pocket_coords: {coords_mod.shape}")
    print(f"  pocket_one_hot: ({len(atoms_mod)}, 11)  # Would be one-hot encoded")
    print(f"  pocket_residue_ids: {ids_mod}")
    print(f"  pocket_esmc: {esmc_atom.shape}")


def show_code_diff():
    """
    Show the exact code changes needed in process_crossdock.py
    """
    print("\n" + "="*80)
    print("CODE CHANGES FOR process_crossdock.py")
    print("="*80 + "\n")

    print("LOCATION: Lines 105-138 (full-atom branch)")
    print("\n--- BEFORE (current) ---\n")
    print("""
    full_atoms = np.concatenate([
        np.array([atom.element for atom in res.get_atoms()])
        for res in pocket_residues
    ], axis=0)

    full_coords = np.concatenate([
        np.array([atom.coord for atom in res.get_atoms()])
        for res in pocket_residues
    ], axis=0)
""")

    print("\n+++ AFTER (modified) +++\n")
    print("""
    # Track residue IDs per atom
    full_atoms = []
    full_coords = []
    residue_ids = []  # NEW

    for res_idx, res in enumerate(pocket_residues):
        for atom in res.get_atoms():
            full_atoms.append(atom.element)
            full_coords.append(atom.coord)
            residue_ids.append(res_idx)  # NEW

    full_atoms = np.array(full_atoms)
    full_coords = np.array(full_coords)
    residue_ids = np.array(residue_ids, dtype=np.int32)  # NEW
""")

    print("\n--- BEFORE (pocket_data dict) ---\n")
    print("""
    pocket_data = {
        "pocket_coords": full_coords,
        "pocket_one_hot": pocket_one_hot,
        "pocket_ids": pocket_ids,
    }
""")

    print("\n+++ AFTER (pocket_data dict) +++\n")
    print("""
    pocket_data = {
        "pocket_coords": full_coords,
        "pocket_one_hot": pocket_one_hot,
        "pocket_ids": pocket_ids,
        "pocket_residue_ids": residue_ids,  # NEW
    }
""")


if __name__ == "__main__":
    demonstrate_full_pipeline()
    show_code_diff()

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Apply the code changes to process_crossdock.py
2. Test on a single PDB file:
   python process_crossdock.py /path/to/test.pdb --no_H --full_atom

3. Verify residue IDs are correct:
   data = np.load('output.npz')
   print(data['pocket_residue_ids'])

4. Re-process full dataset (may take several hours)

5. Proceed with ESM-C embedding pre-computation
""")
    print("="*80 + "\n")
