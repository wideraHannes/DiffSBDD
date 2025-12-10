#!/usr/bin/env python3
"""
Enhanced analysis script with AutoDock Vina docking scores
Evaluates molecular quality, validity, drug-likeness, AND binding affinity
"""

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, Lipinski
from tqdm import tqdm
import argparse
import logging
from scipy.stats import wasserstein_distance
import subprocess

from analysis.metrics import BasicMolecularMetrics, MoleculeProperties
from analysis.vina_docking import vina_score
from constants import dataset_params

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def find_receptor_for_pocket(pocket_name: str, test_data_dir: Path) -> Path:
    """
    Find the receptor PDB file for a given pocket name.

    Args:
        pocket_name: Pocket identifier (e.g., "3daf-A-rec-3daf-feg-lig-tt-docked-0-pocket10")
        test_data_dir: Directory containing test dataset receptors

    Returns:
        Path to receptor PDB file
    """
    # Try to find PDB file with exact match
    pdb_file = test_data_dir / f"{pocket_name}.pdb"
    if pdb_file.exists():
        return pdb_file

    # If not found, log warning
    logging.warning(f"Receptor PDB not found for pocket: {pocket_name}")
    return None


def prepare_receptor_pdbqt(pdb_file: Path, output_dir: Path) -> Path:
    """
    Convert PDB receptor to PDBQT format if not already done.

    Args:
        pdb_file: Input PDB file
        output_dir: Directory for PDBQT outputs

    Returns:
        Path to PDBQT file
    """
    pdbqt_file = output_dir / f"{pdb_file.stem}.pdbqt"

    # Skip if already exists
    if pdbqt_file.exists():
        return pdbqt_file

    # Prepare receptor
    cmd = f'obabel {pdb_file} -O {pdbqt_file} -xr'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        logging.warning(f"Receptor preparation failed for {pdb_file.name}: {result.stderr}")
        return None

    return pdbqt_file


def calculate_vina_scores(
    molecules,
    sdf_file: Path,
    test_data_dir: Path,
    receptor_pdbqt_dir: Path,
    exhaustiveness: int = 8,
    score_only: bool = False
):
    """
    Calculate Vina docking scores for generated molecules.

    Args:
        molecules: List of RDKit molecules
        sdf_file: SDF file path (contains pocket name)
        test_data_dir: Directory with test dataset receptors
        receptor_pdbqt_dir: Directory for prepared receptors
        exhaustiveness: Vina exhaustiveness parameter
        score_only: If True, use fast score_only mode

    Returns:
        List of Vina scores (np.nan for failed molecules)
    """
    # Extract pocket name from SDF filename
    # Format: {pocket_name}_{pocket_name}_gen.sdf
    filename = sdf_file.stem
    if "_gen" in filename:
        # Remove _gen suffix and the repeated pocket name
        parts = filename.replace("_gen", "").split("_")
        # Find where the pocket name repeats
        # e.g., "3daf-A-rec-3daf-feg-lig-tt-docked-0-pocket10_3daf-A-rec-3daf-feg-lig-tt-docked-0"
        # Split by underscore and take the first part (pocket name)
        pocket_name = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]

        # Find last occurrence of pocket name pattern
        # The pocket name ends with -pocket10
        if "-pocket10" in pocket_name:
            pocket_name = pocket_name.split("-pocket10")[0] + "-pocket10"
    else:
        pocket_name = filename

    # Find receptor
    receptor_pdb = find_receptor_for_pocket(pocket_name, test_data_dir)
    if receptor_pdb is None:
        logging.warning(f"Skipping Vina scoring for {sdf_file.name}: no receptor found")
        return [np.nan] * len(molecules)

    # Prepare receptor PDBQT
    try:
        receptor_pdbqt = prepare_receptor_pdbqt(receptor_pdb, receptor_pdbqt_dir)
        if receptor_pdbqt is None:
            return [np.nan] * len(molecules)
    except Exception as e:
        logging.warning(f"Receptor preparation failed: {e}")
        return [np.nan] * len(molecules)

    # Calculate Vina scores for each molecule
    scores = []
    for mol in tqdm(molecules, desc=f"Docking {sdf_file.name}", leave=False):
        if mol is None:
            scores.append(np.nan)
            continue

        try:
            # Get ligand center for box placement
            conf = mol.GetConformer()
            center = tuple(conf.GetPositions().mean(axis=0))

            # Run Vina
            score = vina_score(
                mol,
                receptor_pdbqt,
                center=center,
                box_size=(25, 25, 25),
                exhaustiveness=exhaustiveness,
                score_only=score_only
            )

            # If full docking (returns list), take best score
            if isinstance(score, list):
                score = score[0] if score else np.nan

            scores.append(score)

        except Exception as e:
            logging.debug(f"Vina scoring failed for molecule: {e}")
            scores.append(np.nan)

    return scores


def calculate_basic_properties(molecules):
    """Calculate basic molecular properties"""
    if not molecules:
        return {}

    properties = {
        "molecular_weights": [],
        "logp_values": [],
        "hbd_counts": [],
        "hba_counts": [],
        "rotatable_bonds": [],
        "num_atoms": [],
        "num_heavy_atoms": [],
        "qed_scores": [],
        "sa_scores": [],
    }

    for mol in molecules:
        if mol is None:
            continue

        try:
            # Try to sanitize the molecule first
            Chem.SanitizeMol(mol)

            properties["molecular_weights"].append(Descriptors.MolWt(mol))
            properties["logp_values"].append(Crippen.MolLogP(mol))
            properties["hbd_counts"].append(Descriptors.NumHDonors(mol))
            properties["hba_counts"].append(Descriptors.NumHAcceptors(mol))
            properties["rotatable_bonds"].append(Descriptors.NumRotatableBonds(mol))
            properties["num_atoms"].append(mol.GetNumAtoms())
            properties["num_heavy_atoms"].append(mol.GetNumHeavyAtoms())

            # QED calculation can fail for invalid molecules
            try:
                properties["qed_scores"].append(QED.qed(mol))
            except:
                properties["qed_scores"].append(0.0)

            # SA Score (if available)
            try:
                from analysis.SA_Score.sascorer import calculateScore
                properties["sa_scores"].append(calculateScore(mol))
            except:
                properties["sa_scores"].append(None)

        except Exception as e:
            # Skip invalid molecules entirely
            logging.debug(f"Skipping invalid molecule: {e}")
            continue

    return properties


def lipinski_compliance(mol):
    """Check Lipinski Rule of Five compliance"""
    if mol is None:
        return 0

    try:
        Chem.SanitizeMol(mol)

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1

        return 4 - violations  # Return number of rules satisfied
    except:
        return 0


def compute_wasserstein_distances(generated_props, ground_truth_df):
    """
    Compute Wasserstein distance between generated and ground truth distributions

    Now includes Vina scores!

    Args:
        generated_props: dict with property lists from calculate_basic_properties
        ground_truth_df: DataFrame with ground truth properties

    Returns:
        dict of Wasserstein distances for each property
    """
    distances = {}

    # QED - check both naming conventions
    if len(generated_props["qed_scores"]) > 0:
        gt_qed_col = None
        if "qed_scores" in ground_truth_df.columns:
            gt_qed_col = "qed_scores"
        elif "qed" in ground_truth_df.columns:
            gt_qed_col = "qed"

        if gt_qed_col is not None:
            distances["qed_wasserstein"] = wasserstein_distance(
                generated_props["qed_scores"], ground_truth_df[gt_qed_col].dropna()
            )

    # SA Score - check both naming conventions
    if any(generated_props["sa_scores"]):
        gen_sa = [s for s in generated_props["sa_scores"] if s is not None]
        gt_sa_col = None
        if "sa_scores" in ground_truth_df.columns:
            gt_sa_col = "sa_scores"
        elif "sa_score" in ground_truth_df.columns:
            gt_sa_col = "sa_score"

        if gt_sa_col is not None and len(gen_sa) > 0:
            gt_sa = ground_truth_df[gt_sa_col].dropna()
            if len(gt_sa) > 0:
                distances["sa_wasserstein"] = wasserstein_distance(gen_sa, gt_sa)

    # LogP - check both naming conventions
    if len(generated_props["logp_values"]) > 0:
        gt_logp_col = None
        if "logp_values" in ground_truth_df.columns:
            gt_logp_col = "logp_values"
        elif "logp" in ground_truth_df.columns:
            gt_logp_col = "logp"

        if gt_logp_col is not None:
            distances["logp_wasserstein"] = wasserstein_distance(
                generated_props["logp_values"], ground_truth_df[gt_logp_col].dropna()
            )

    # Molecular Weight - check both naming conventions
    if len(generated_props["molecular_weights"]) > 0:
        gt_mw_col = None
        if "molecular_weights" in ground_truth_df.columns:
            gt_mw_col = "molecular_weights"
        elif "molecular_weight" in ground_truth_df.columns:
            gt_mw_col = "molecular_weight"

        if gt_mw_col is not None:
            distances["molwt_wasserstein"] = wasserstein_distance(
                generated_props["molecular_weights"],
                ground_truth_df[gt_mw_col].dropna(),
            )

    # Number of atoms
    if len(generated_props["num_atoms"]) > 0 and "num_atoms" in ground_truth_df.columns:
        distances["numatoms_wasserstein"] = wasserstein_distance(
            generated_props["num_atoms"], ground_truth_df["num_atoms"].dropna()
        )

    # VINA SCORES! (NEW)
    if "vina_scores" in generated_props and len(generated_props["vina_scores"]) > 0:
        # Check both possible column names in ground truth
        vina_col = None
        if "vina_score" in ground_truth_df.columns:
            vina_col = "vina_score"
        elif "vina_scores" in ground_truth_df.columns:
            vina_col = "vina_scores"

        if vina_col is not None:
            gen_vina = [s for s in generated_props["vina_scores"] if not np.isnan(s)]
            gt_vina = ground_truth_df[vina_col].dropna()

            if len(gen_vina) > 0 and len(gt_vina) > 0:
                distances["vina_wasserstein"] = wasserstein_distance(gen_vina, gt_vina)
                logging.info(f"✓ Calculated Vina Wasserstein distance: {distances['vina_wasserstein']:.4f}")

    return distances


def analyze_results(
    results_dir,
    output_file=None,
    ground_truth_csv=None,
    test_data_dir=None,
    enable_vina=True,
    vina_exhaustiveness=8,
    vina_score_only=True
):
    """Analyze generated molecules quality with optional Vina docking

    Args:
        results_dir: Directory containing generated SDF files
        output_file: Path to save analysis results CSV
        ground_truth_csv: Path to ground truth properties CSV for Wasserstein distance
        test_data_dir: Directory containing test dataset receptors (for Vina)
        enable_vina: If True, calculate Vina docking scores
        vina_exhaustiveness: Vina exhaustiveness parameter (8=default, 16=recommended)
        vina_score_only: If True, use fast score_only mode
    """

    results_dir = Path(results_dir)
    processed_dir = results_dir / "processed"
    raw_dir = results_dir / "raw"

    logging.info(f"Analyzing results in: {results_dir}")

    # Set up Vina if enabled
    if enable_vina:
        if test_data_dir is None:
            test_data_dir = Path("data/dummy_testing_dataset_10_tests/test")
        else:
            test_data_dir = Path(test_data_dir)

        if not test_data_dir.exists():
            logging.warning(f"Test data directory not found: {test_data_dir}")
            logging.warning("Disabling Vina scoring")
            enable_vina = False
        else:
            # Create directory for prepared receptors
            receptor_pdbqt_dir = results_dir / "receptors_pdbqt"
            receptor_pdbqt_dir.mkdir(exist_ok=True)
            logging.info(f"✓ Vina docking enabled (exhaustiveness={vina_exhaustiveness}, score_only={vina_score_only})")

    # Check if directories exist
    if not processed_dir.exists():
        logging.warning(f"Processed directory not found: {processed_dir}")
        processed_dir = raw_dir

    if not processed_dir.exists():
        logging.error(f"No results directory found: {processed_dir}")
        return None

    # Load dataset info for metrics
    try:
        dataset_info = dataset_params["crossdock_full"]
        mol_metrics = BasicMolecularMetrics(dataset_info)
        mol_props = MoleculeProperties()
        logging.info("Loaded dataset parameters and metrics")
    except Exception as e:
        logging.warning(f"Could not load advanced metrics: {e}")
        mol_metrics = None
        mol_props = None

    # Collect all molecules and calculate Vina scores per pocket
    all_molecules = []
    all_vina_scores = []
    pocket_results = []

    logging.info("Loading generated molecules...")
    sdf_files = list(processed_dir.glob("*.sdf"))

    if not sdf_files:
        logging.error(f"No SDF files found in {processed_dir}")
        return None

    for sdf_file in tqdm(sdf_files, desc="Processing SDF files"):
        try:
            suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
            pocket_mols = [mol for mol in suppl if mol is not None]

            pocket_name = sdf_file.stem.replace("_gen", "")
            pocket_info = {
                "pocket_name": pocket_name,
                "num_molecules": len(pocket_mols),
                "valid_molecules": len([m for m in pocket_mols if m is not None]),
            }

            if len(pocket_mols) > 0:
                all_molecules.extend(pocket_mols)

                # Calculate Vina scores for this pocket
                if enable_vina:
                    pocket_vina_scores = calculate_vina_scores(
                        pocket_mols,
                        sdf_file,
                        test_data_dir,
                        receptor_pdbqt_dir,
                        exhaustiveness=vina_exhaustiveness,
                        score_only=vina_score_only
                    )
                    all_vina_scores.extend(pocket_vina_scores)

                pocket_results.append(pocket_info)

        except Exception as e:
            logging.warning(f"Error processing {sdf_file}: {e}")
            continue

    logging.info(f"Total molecules loaded: {len(all_molecules)}")
    logging.info(f"Total pockets processed: {len(pocket_results)}")

    if enable_vina:
        valid_vina = [s for s in all_vina_scores if not np.isnan(s)]
        logging.info(f"Vina scores calculated: {len(valid_vina)}/{len(all_molecules)} ({100*len(valid_vina)/len(all_molecules):.1f}%)")

    if not all_molecules:
        logging.error("No valid molecules found!")
        return None

    # Calculate basic properties
    logging.info("Calculating molecular properties...")
    properties = calculate_basic_properties(all_molecules)

    # Add Vina scores to properties
    if enable_vina and all_vina_scores:
        properties["vina_scores"] = all_vina_scores

    # Calculate advanced metrics if available
    validity_metrics = {}
    if mol_metrics is not None:
        try:
            logging.info("Calculating validity metrics...")
            validity_results, (valid_mols, connected_mols) = (
                mol_metrics.evaluate_rdmols(all_molecules)
            )
            validity, connectivity, uniqueness, novelty = validity_results

            validity_metrics = {
                "validity": validity,
                "connectivity": connectivity,
                "uniqueness": uniqueness,
                "novelty": novelty,
                "num_valid": len(valid_mols),
                "num_connected": len(connected_mols),
            }

            # Calculate drug-likeness properties for valid molecules
            if mol_props is not None and len(connected_mols) > 0:
                qed, sa, logp, lipinski, diversity = mol_props.evaluate_mean(
                    connected_mols
                )
                validity_metrics.update(
                    {
                        "qed_advanced": qed,
                        "sa_advanced": sa,
                        "logp_advanced": logp,
                        "lipinski_advanced": lipinski,
                        "diversity": diversity,
                    }
                )

        except Exception as e:
            logging.warning(f"Error calculating advanced metrics: {e}")

    # Calculate Lipinski compliance
    lipinski_scores = [lipinski_compliance(mol) for mol in all_molecules]

    # Create summary statistics
    summary = {
        "total_molecules": len(all_molecules),
        "total_pockets": len(pocket_results),
        "avg_molecules_per_pocket": np.mean(
            [p["num_molecules"] for p in pocket_results]
        ),
        # Basic properties statistics
        "avg_molecular_weight": np.mean(properties["molecular_weights"]),
        "std_molecular_weight": np.std(properties["molecular_weights"]),
        "avg_logp": np.mean(properties["logp_values"]),
        "std_logp": np.std(properties["logp_values"]),
        "avg_hbd": np.mean(properties["hbd_counts"]),
        "avg_hba": np.mean(properties["hba_counts"]),
        "avg_rotatable_bonds": np.mean(properties["rotatable_bonds"]),
        "avg_num_atoms": np.mean(properties["num_atoms"]),
        "std_num_atoms": np.std(properties["num_atoms"]),
        "min_num_atoms": np.min(properties["num_atoms"]),
        "max_num_atoms": np.max(properties["num_atoms"]),
        "avg_qed": np.mean(properties["qed_scores"]),
        "std_qed": np.std(properties["qed_scores"]),
        # Lipinski compliance
        "avg_lipinski_rules": np.mean(lipinski_scores),
        "lipinski_compliant_percent": np.mean([score >= 4 for score in lipinski_scores])
        * 100,
        # SA scores (if available)
        "avg_sa_score": np.mean([s for s in properties["sa_scores"] if s is not None])
        if any(properties["sa_scores"])
        else None,
    }

    # Add Vina statistics
    if enable_vina and "vina_scores" in properties:
        valid_vina = [s for s in properties["vina_scores"] if not np.isnan(s)]
        if len(valid_vina) > 0:
            summary["avg_vina_score"] = np.mean(valid_vina)
            summary["std_vina_score"] = np.std(valid_vina)
            summary["min_vina_score"] = np.min(valid_vina)
            summary["max_vina_score"] = np.max(valid_vina)
            summary["vina_valid_percent"] = 100 * len(valid_vina) / len(properties["vina_scores"])

    # Add validity metrics
    summary.update(validity_metrics)

    # Calculate Wasserstein distances if ground truth is provided
    wasserstein_distances = {}
    if ground_truth_csv is not None:
        try:
            ground_truth_df = pd.read_csv(ground_truth_csv)
            logging.info(f"Loaded ground truth data from: {ground_truth_csv}")
            wasserstein_distances = compute_wasserstein_distances(properties, ground_truth_df)
            summary.update(wasserstein_distances)
            logging.info("Calculated Wasserstein distances to ground truth")
        except Exception as e:
            logging.warning(f"Could not compute Wasserstein distances: {e}")

    # Print results
    print("\n" + "=" * 60)
    print("DIFFSBDD MOLECULE GENERATION ANALYSIS (with Vina)")
    print("=" * 60)

    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  Total molecules generated: {summary['total_molecules']}")
    print(f"  Total pockets processed: {summary['total_pockets']}")
    print(f"  Average molecules per pocket: {summary['avg_molecules_per_pocket']:.1f}")

    print(f"\n🔬 MOLECULAR PROPERTIES:")
    print(
        f"  Molecular Weight: {summary['avg_molecular_weight']:.1f} ± {summary['std_molecular_weight']:.1f} Da"
    )
    print(f"  LogP: {summary['avg_logp']:.2f} ± {summary['std_logp']:.2f}")
    print(f"  H-bond donors: {summary['avg_hbd']:.1f}")
    print(f"  H-bond acceptors: {summary['avg_hba']:.1f}")
    print(f"  Rotatable bonds: {summary['avg_rotatable_bonds']:.1f}")
    print(
        f"  Number of atoms: {summary['avg_num_atoms']:.1f} ± {summary['std_num_atoms']:.1f} ({summary['min_num_atoms']}-{summary['max_num_atoms']})"
    )

    print(f"\n💊 DRUG-LIKENESS:")
    print(f"  QED Score: {summary['avg_qed']:.3f} ± {summary['std_qed']:.3f}")
    print(f"  Lipinski Rules (avg): {summary['avg_lipinski_rules']:.1f}/4")
    print(f"  Lipinski Compliant: {summary['lipinski_compliant_percent']:.1f}%")
    if summary["avg_sa_score"] is not None:
        print(f"  SA Score: {summary['avg_sa_score']:.2f}")

    # Print Vina scores
    if enable_vina and "avg_vina_score" in summary:
        print(f"\n🔬 VINA DOCKING SCORES:")
        print(f"  Mean: {summary['avg_vina_score']:.3f} ± {summary['std_vina_score']:.3f} kcal/mol")
        print(f"  Range: [{summary['min_vina_score']:.3f}, {summary['max_vina_score']:.3f}] kcal/mol")
        print(f"  Valid scores: {summary['vina_valid_percent']:.1f}%")

        # Categorize binding strength
        if 'vina_scores' in properties:
            valid_vina = [s for s in properties['vina_scores'] if not np.isnan(s)]
            excellent = sum(1 for s in valid_vina if s < -10.0)
            strong = sum(1 for s in valid_vina if -10.0 <= s < -9.0)
            good = sum(1 for s in valid_vina if -9.0 <= s < -7.0)
            moderate = sum(1 for s in valid_vina if -7.0 <= s < -5.0)

            print(f"  Binding strength distribution:")
            if excellent > 0:
                print(f"    Excellent (< -10.0): {excellent} ({100*excellent/len(valid_vina):.1f}%)")
            if strong > 0:
                print(f"    Strong (-10.0 to -9.0): {strong} ({100*strong/len(valid_vina):.1f}%)")
            if good > 0:
                print(f"    Good (-9.0 to -7.0): {good} ({100*good/len(valid_vina):.1f}%)")
            if moderate > 0:
                print(f"    Moderate (-7.0 to -5.0): {moderate} ({100*moderate/len(valid_vina):.1f}%)")

    if validity_metrics:
        print(f"\n✅ VALIDITY METRICS:")
        print(
            f"  Validity: {validity_metrics['validity']:.3f} ({validity_metrics['num_valid']}/{summary['total_molecules']})"
        )
        print(
            f"  Connectivity: {validity_metrics['connectivity']:.3f} ({validity_metrics['num_connected']}/{summary['total_molecules']})"
        )
        print(f"  Uniqueness: {validity_metrics['uniqueness']:.3f}")
        print(f"  Novelty: {validity_metrics['novelty']:.3f}")
        if "diversity" in validity_metrics:
            print(f"  Diversity: {validity_metrics['diversity']:.3f}")

    if wasserstein_distances:
        print(f"\n📏 WASSERSTEIN DISTANCES (vs Ground Truth):")
        if "qed_wasserstein" in wasserstein_distances:
            print(f"  QED: {wasserstein_distances['qed_wasserstein']:.4f}")
        if "sa_wasserstein" in wasserstein_distances:
            print(f"  SA Score: {wasserstein_distances['sa_wasserstein']:.4f}")
        if "logp_wasserstein" in wasserstein_distances:
            print(f"  LogP: {wasserstein_distances['logp_wasserstein']:.4f}")
        if "molwt_wasserstein" in wasserstein_distances:
            print(f"  Molecular Weight: {wasserstein_distances['molwt_wasserstein']:.4f}")
        if "numatoms_wasserstein" in wasserstein_distances:
            print(f"  Num Atoms: {wasserstein_distances['numatoms_wasserstein']:.4f}")
        if "vina_wasserstein" in wasserstein_distances:
            print(f"  Vina Score: {wasserstein_distances['vina_wasserstein']:.4f} ⭐")

    # Quality assessment (same as before)
    print(f"\n🎯 QUALITY ASSESSMENT:")
    quality_score = 0
    max_score = 0

    if "validity" in validity_metrics:
        if validity_metrics["validity"] > 0.8:
            print(f"  ✅ Excellent validity ({validity_metrics['validity']:.3f})")
            quality_score += 2
        elif validity_metrics["validity"] > 0.6:
            print(f"  ⚠️  Good validity ({validity_metrics['validity']:.3f})")
            quality_score += 1
        else:
            print(f"  ❌ Poor validity ({validity_metrics['validity']:.3f})")
        max_score += 2

    if summary["avg_qed"] > 0.5:
        print(f"  ✅ Good drug-likeness (QED: {summary['avg_qed']:.3f})")
        quality_score += 2
    elif summary["avg_qed"] > 0.3:
        print(f"  ⚠️  Moderate drug-likeness (QED: {summary['avg_qed']:.3f})")
        quality_score += 1
    else:
        print(f"  ❌ Poor drug-likeness (QED: {summary['avg_qed']:.3f})")
    max_score += 2

    if summary["lipinski_compliant_percent"] > 80:
        print(
            f"  ✅ Excellent Lipinski compliance ({summary['lipinski_compliant_percent']:.1f}%)"
        )
        quality_score += 2
    elif summary["lipinski_compliant_percent"] > 60:
        print(
            f"  ⚠️  Good Lipinski compliance ({summary['lipinski_compliant_percent']:.1f}%)"
        )
        quality_score += 1
    else:
        print(
            f"  ❌ Poor Lipinski compliance ({summary['lipinski_compliant_percent']:.1f}%)"
        )
    max_score += 2

    if 15 <= summary["avg_num_atoms"] <= 35:
        print(f"  ✅ Good size distribution ({summary['avg_num_atoms']:.1f} atoms)")
        quality_score += 1
    else:
        print(f"  ⚠️  Unusual size distribution ({summary['avg_num_atoms']:.1f} atoms)")
    max_score += 1

    # Add Vina quality check
    if enable_vina and "avg_vina_score" in summary:
        if summary["avg_vina_score"] < -7.0:
            print(f"  ✅ Good binding affinity ({summary['avg_vina_score']:.2f} kcal/mol)")
            quality_score += 2
        elif summary["avg_vina_score"] < -5.0:
            print(f"  ⚠️  Moderate binding affinity ({summary['avg_vina_score']:.2f} kcal/mol)")
            quality_score += 1
        else:
            print(f"  ❌ Weak binding affinity ({summary['avg_vina_score']:.2f} kcal/mol)")
        max_score += 2

    overall_quality = (quality_score / max_score) * 100 if max_score > 0 else 0
    print(
        f"\n🏆 OVERALL QUALITY SCORE: {quality_score}/{max_score} ({overall_quality:.1f}%)"
    )

    # Save detailed results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save summary
        pd.DataFrame([summary]).to_csv(output_path, index=False)

        # Save detailed molecule data
        detailed_file = output_path.with_suffix(".detailed.csv")
        detailed_data = []
        prop_idx = 0  # Track valid molecules in properties dict
        for i, mol in enumerate(all_molecules):
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
                # Only include molecules that have properties calculated
                if prop_idx < len(properties["molecular_weights"]):
                    row = {
                        "molecule_id": i,
                        "smiles": Chem.MolToSmiles(mol),
                        "molecular_weight": properties["molecular_weights"][prop_idx],
                        "logp": properties["logp_values"][prop_idx],
                        "hbd": properties["hbd_counts"][prop_idx],
                        "hba": properties["hba_counts"][prop_idx],
                        "rotatable_bonds": properties["rotatable_bonds"][prop_idx],
                        "num_atoms": properties["num_atoms"][prop_idx],
                        "qed": properties["qed_scores"][prop_idx],
                        "lipinski_rules": lipinski_scores[i],
                    }
                    if prop_idx < len(properties["sa_scores"]) and properties["sa_scores"][prop_idx] is not None:
                        row["sa_score"] = properties["sa_scores"][prop_idx]
                    if enable_vina and "vina_scores" in properties and prop_idx < len(properties["vina_scores"]):
                        row["vina_score"] = properties["vina_scores"][prop_idx]
                    detailed_data.append(row)
                    prop_idx += 1
            except:
                continue

        pd.DataFrame(detailed_data).to_csv(detailed_file, index=False)

        print(f"\n💾 Results saved to:")
        print(f"  Summary: {output_path}")
        print(f"  Detailed: {detailed_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze DiffSBDD generated molecules with Vina docking")
    parser.add_argument("results_dir", type=Path, help="Path to results directory")
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Output CSV file for results"
    )
    parser.add_argument(
        "--ground_truth", type=Path, default=None,
        help="Path to ground truth properties CSV for Wasserstein distance comparison"
    )
    parser.add_argument(
        "--test_data_dir", type=Path, default=None,
        help="Directory containing test dataset receptors (default: data/dummy_testing_dataset_10_tests/test)"
    )
    parser.add_argument(
        "--disable_vina", action="store_true",
        help="Disable Vina docking calculations"
    )
    parser.add_argument(
        "--vina_exhaustiveness", type=int, default=8,
        help="Vina exhaustiveness (8=default, 16=recommended, 32=publication)"
    )
    parser.add_argument(
        "--vina_full_docking", action="store_true",
        help="Use full docking instead of score_only mode (slower but more accurate)"
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.results_dir / "analysis_summary.csv"

    analyze_results(
        args.results_dir,
        args.output,
        args.ground_truth,
        test_data_dir=args.test_data_dir,
        enable_vina=not args.disable_vina,
        vina_exhaustiveness=args.vina_exhaustiveness,
        vina_score_only=not args.vina_full_docking
    )


if __name__ == "__main__":
    main()
