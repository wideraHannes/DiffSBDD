## Preprocessing error

`uv run process_crossdock.py data`

current error:
34029/99927 successful: 100%|█████████████████████████████████████████████████████████████████████████▉| 99927/99929 [01:50<00:00, 906.06it/s]
Traceback (most recent call last):
File "/Users/hanneswidera/Uni/Master/thesis/DiffSBDD/process_crossdock.py", line 457, in <module>
train_smiles = compute_smiles(lig_coords, lig_one_hot, lig_mask)
File "/Users/hanneswidera/Uni/Master/thesis/DiffSBDD/process_crossdock.py", line 155, in compute_smiles
mol = build_molecule(
File "/Users/hanneswidera/Uni/Master/thesis/DiffSBDD/analysis/molecule_builder.py", line 157, in build_molecule
mol = make_mol_edm(positions, atom_types, dataset_info, add_coords)
File "/Users/hanneswidera/Uni/Master/thesis/DiffSBDD/analysis/molecule_builder.py", line 114, in make_mol_edm
E_full = get_bond_order_batch(atoms1, atoms2, dists, dataset_info).view(n, n)
File "/Users/hanneswidera/Uni/Master/thesis/DiffSBDD/analysis/molecule_builder.py", line 47, in get_bond_order_batch
bond_types[distances < bonds1[atoms1, atoms2] + margin1] = 1
RuntimeError: The size of tensor a (400) must match the size of tensor b (169) at non-singleton dimension 0

-> train folder is empty in noH_full_temp???
