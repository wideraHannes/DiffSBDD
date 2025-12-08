import argparse
import warnings
from pathlib import Path
from time import time
import logging

import torch
from rdkit import Chem
from tqdm import tqdm

from lightning_modules import LigandPocketDDPM
from analysis.molecule_builder import process_molecule
import utils

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MAXITER = 10
MAXNTRIES = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--test_dir", type=Path)
    parser.add_argument("--test_list", type=Path, default=None)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--n_samples", type=int, default=100)  # before 100
    parser.add_argument("--all_frags", action="store_true")
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--relax", action="store_true")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--resamplings", type=int, default=10)
    parser.add_argument("--jump_length", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--fix_n_nodes", action="store_true")
    parser.add_argument("--n_nodes_bias", type=int, default=0)
    parser.add_argument("--n_nodes_min", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--init_film_identity",
        action="store_true",
        help="Initialize FiLM to identity (gamma=1, beta=0). Use for baseline checkpoints without FiLM training.",
    )
    parser.add_argument(
        "--use_film",
        action="store_true",
        help="Enable FiLM conditioning (for baseline experiments)",
    )
    parser.add_argument(
        "--film_mode",
        type=str,
        choices=["identity", "random"],
        default="identity",
        help="FiLM initialization mode: 'identity' (γ=1, β=0) or 'random' (Kaiming uniform)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    args.outdir.mkdir(exist_ok=args.skip_existing)
    raw_sdf_dir = Path(args.outdir, "raw")
    raw_sdf_dir.mkdir(exist_ok=args.skip_existing)
    processed_sdf_dir = Path(args.outdir, "processed")
    processed_sdf_dir.mkdir(exist_ok=args.skip_existing)
    times_dir = Path(args.outdir, "pocket_times")
    times_dir.mkdir(exist_ok=args.skip_existing)

    # Load model - use new loading method if FiLM control requested
    logging.info(f"Loading model from checkpoint: {args.checkpoint}")

    if args.use_film or not args.init_film_identity:
        # Use new loading method with FiLM control
        use_film = args.use_film  # Explicit FiLM flag
        film_mode = args.film_mode if args.use_film else "identity"

        logging.info(f"Loading with FiLM control: use_film={use_film}, film_mode={film_mode}")
        model = LigandPocketDDPM.load_pretrained_with_esmc(
            args.checkpoint,
            device=device,
            film_only_training=False,
            use_film=use_film,
            film_mode=film_mode,
        )
    else:
        # Legacy path: load normally and init FiLM to identity
        model = LigandPocketDDPM.load_from_checkpoint(
            args.checkpoint, map_location=device, strict=False
        )
        model = model.to(device)

        # Initialize FiLM to identity if requested (for baseline checkpoint without FiLM training)
        # This ensures baseline behaves as: h' = 1*h + 0 = h (FiLM has no effect)
        if args.init_film_identity:
            film = model.ddpm.dynamics.film_network
            joint_nf = film[-1].out_features // 2
            with torch.no_grad():
                film[-1].weight.zero_()
                film[-1].bias.zero_()
                film[-1].bias[:joint_nf] = 1.0  # gamma = 1, beta = 0
            logging.info("FiLM initialized to identity (gamma=1, beta=0)")

    logging.info("Model loaded successfully")

    test_files = list(args.test_dir.glob("[!.]*.sdf"))
    logging.info(f"Found {len(test_files)} test files")
    if args.test_list is not None:
        with open(args.test_list, "r") as f:
            test_list = set(f.read().split(","))
        test_files = [x for x in test_files if x.stem in test_list]
        logging.info(f"Filtered to {len(test_files)} files based on test list")

    pbar = tqdm(test_files)
    time_per_pocket = {}
    for i, sdf_file in enumerate(pbar):
        ligand_name = sdf_file.stem
        logging.info(f"Processing pocket {i + 1}/{len(test_files)}: {ligand_name}")

        pdb_name, pocket_id, *suffix = ligand_name.split("_")
        pdb_file = Path(sdf_file.parent, f"{pdb_name}.pdb")
        txt_file = Path(sdf_file.parent, f"{ligand_name}.txt")
        sdf_out_file_raw = Path(raw_sdf_dir, f"{ligand_name}_gen.sdf")
        sdf_out_file_processed = Path(processed_sdf_dir, f"{ligand_name}_gen.sdf")
        time_file = Path(times_dir, f"{ligand_name}.txt")

        logging.info(
            f"Input files - PDB: {pdb_file.exists()}, TXT: {txt_file.exists()}"
        )

        if (
            args.skip_existing
            and time_file.exists()
            and sdf_out_file_processed.exists()
            and sdf_out_file_raw.exists()
        ):
            logging.info(f"Skipping {ligand_name} - already processed")
            with open(time_file, "r") as f:
                time_per_pocket[str(sdf_file)] = float(f.read().split()[1])

            continue

        for n_try in range(MAXNTRIES):
            logging.info(f"Attempt {n_try + 1}/{MAXNTRIES} for {ligand_name}")

            try:
                t_pocket_start = time()
                logging.info(f"Starting molecule generation for {ligand_name}")

                # Use reference ligand (SDF) to define pocket instead of residue list
                # This avoids issues with residue ID mismatches (insertion codes, etc.)
                ref_ligand = str(sdf_file)
                logging.info(
                    f"Using reference ligand to define pocket: {sdf_file.name}"
                )

                if args.fix_n_nodes:
                    # some ligands (e.g. 6JWS_bio1_PT1:A:801) could not be read with sanitize=True
                    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
                    num_nodes_lig = suppl[0].GetNumAtoms()
                    logging.info(f"Fixed number of nodes: {num_nodes_lig}")
                else:
                    num_nodes_lig = None
                    logging.info("Using variable number of nodes")

                all_molecules = []
                valid_molecules = []
                processed_molecules = []  # only used as temporary variable
                iter = 0
                n_generated = 0
                n_valid = 0
                while len(valid_molecules) < args.n_samples:
                    iter += 1
                    logging.info(
                        f"Generation iteration {iter}/{MAXITER}, need {args.n_samples - len(valid_molecules)} more valid molecules"
                    )
                    if iter > MAXITER:
                        raise RuntimeError(
                            "Maximum number of iterations has been exceeded."
                        )

                    num_nodes_lig_inflated = (
                        None
                        if num_nodes_lig is None
                        else torch.ones(args.batch_size, dtype=torch.int)
                        * num_nodes_lig
                    )

                    # Turn all filters off first
                    logging.info(f"Generating batch of {args.batch_size} molecules...")
                    batch_start_time = time()
                    mols_batch = model.generate_ligands(
                        pdb_file,
                        args.batch_size,
                        ref_ligand=ref_ligand,
                        num_nodes_lig=num_nodes_lig_inflated,
                        timesteps=args.timesteps,
                        sanitize=False,
                        largest_frag=False,
                        relax_iter=0,
                        n_nodes_bias=args.n_nodes_bias,
                        n_nodes_min=args.n_nodes_min,
                        resamplings=args.resamplings,
                        jump_length=args.jump_length,
                    )
                    batch_time = time() - batch_start_time
                    logging.info(
                        f"Generated {len(mols_batch)} molecules in {batch_time:.2f} seconds"
                    )

                    all_molecules.extend(mols_batch)

                    # Filter to find valid molecules
                    logging.info("Processing and filtering molecules...")
                    process_start_time = time()
                    mols_batch_processed = [
                        process_molecule(
                            m,
                            sanitize=args.sanitize,
                            relax_iter=(200 if args.relax else 0),
                            largest_frag=not args.all_frags,
                        )
                        for m in mols_batch
                    ]
                    process_time = time() - process_start_time
                    logging.info(f"Processed molecules in {process_time:.2f} seconds")

                    processed_molecules.extend(mols_batch_processed)
                    valid_mols_batch = [
                        m for m in mols_batch_processed if m is not None
                    ]

                    n_generated += args.batch_size
                    n_valid += len(valid_mols_batch)
                    valid_molecules.extend(valid_mols_batch)

                    logging.info(
                        f"Batch results: {len(valid_mols_batch)}/{args.batch_size} valid. Total: {len(valid_molecules)}/{args.n_samples}"
                    )

                # Remove excess molecules from list
                valid_molecules = valid_molecules[: args.n_samples]
                logging.info(f"Final: {len(valid_molecules)} valid molecules generated")

                # Reorder raw files
                all_molecules = [
                    all_molecules[i]
                    for i, m in enumerate(processed_molecules)
                    if m is not None
                ] + [
                    all_molecules[i]
                    for i, m in enumerate(processed_molecules)
                    if m is None
                ]

                # Write SDF files
                logging.info(f"Writing SDF files...")
                utils.write_sdf_file(sdf_out_file_raw, all_molecules)
                utils.write_sdf_file(sdf_out_file_processed, valid_molecules)

                # Time the sampling process
                time_per_pocket[str(sdf_file)] = time() - t_pocket_start
                with open(time_file, "w") as f:
                    f.write(f"{str(sdf_file)} {time_per_pocket[str(sdf_file)]}")

                logging.info(
                    f"Completed {ligand_name} in {time_per_pocket[str(sdf_file)]:.2f} seconds"
                )
                logging.info(f"Final validity: {n_valid / n_generated * 100:.2f}%")

                pbar.set_description(
                    f"Last processed: {ligand_name}. "
                    f"Validity: {n_valid / n_generated * 100:.2f}%. "
                    f"{(time() - t_pocket_start) / len(valid_molecules):.2f} "
                    f"sec/mol."
                )

                break  # no more tries needed

            except (RuntimeError, ValueError) as e:
                logging.warning(
                    f"Attempt {n_try + 1}/{MAXNTRIES} failed with error: {e}"
                )
                if n_try >= MAXNTRIES - 1:
                    raise RuntimeError("Maximum number of retries exceeded")
                warnings.warn(
                    f"Attempt {n_try + 1}/{MAXNTRIES} failed with "
                    f"error: '{e}'. Trying again..."
                )

    logging.info("Writing summary statistics...")
    with open(Path(args.outdir, "pocket_times.txt"), "w") as f:
        for k, v in time_per_pocket.items():
            f.write(f"{k} {v}\n")

    times_arr = torch.tensor([x for x in time_per_pocket.values()])
    print(
        f"Time per pocket: {times_arr.mean():.3f} ± {times_arr.std(unbiased=False):.2f}"
    )
    logging.info("Evaluation completed successfully!")
