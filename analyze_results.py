#!/usr/bin/env python3
"""
Analysis script for DiffSBDD generated molecules
Evaluates molecular quality, validity, and drug-likeness properties
"""

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, Lipinski
from tqdm import tqdm
import argparse
import logging

from analysis.metrics import BasicMolecularMetrics, MoleculeProperties
from constants import dataset_params

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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

        properties["molecular_weights"].append(Descriptors.MolWt(mol))
        properties["logp_values"].append(Crippen.MolLogP(mol))
        properties["hbd_counts"].append(Descriptors.NumHDonors(mol))
        properties["hba_counts"].append(Descriptors.NumHAcceptors(mol))
        properties["rotatable_bonds"].append(Descriptors.NumRotatableBonds(mol))
        properties["num_atoms"].append(mol.GetNumAtoms())
        properties["num_heavy_atoms"].append(mol.GetNumHeavyAtoms())
        properties["qed_scores"].append(QED.qed(mol))

        # SA Score (if available)
        try:
            from analysis.SA_Score.sascorer import calculateScore

            properties["sa_scores"].append(calculateScore(mol))
        except:
            properties["sa_scores"].append(None)

    return properties


def lipinski_compliance(mol):
    """Check Lipinski Rule of Five compliance"""
    if mol is None:
        return 0

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


def analyze_results(results_dir, output_file=None):
    """Analyze generated molecules quality"""

    results_dir = Path(results_dir)
    processed_dir = results_dir / "processed"
    raw_dir = results_dir / "raw"

    logging.info(f"Analyzing results in: {results_dir}")

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

    # Collect all molecules
    all_molecules = []
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
                pocket_results.append(pocket_info)

        except Exception as e:
            logging.warning(f"Error processing {sdf_file}: {e}")
            continue

    logging.info(f"Total molecules loaded: {len(all_molecules)}")
    logging.info(f"Total pockets processed: {len(pocket_results)}")

    if not all_molecules:
        logging.error("No valid molecules found!")
        return None

    # Calculate basic properties
    logging.info("Calculating molecular properties...")
    properties = calculate_basic_properties(all_molecules)

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

    # Add validity metrics
    summary.update(validity_metrics)

    # Print results
    print("\n" + "=" * 60)
    print("DIFFSBDD MOLECULE GENERATION ANALYSIS")
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

    # Quality assessment
    print(f"\n🎯 QUALITY ASSESSMENT:")
    quality_score = 0
    max_score = 0

    # Validity check
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

    # Drug-likeness check
    if summary["avg_qed"] > 0.5:
        print(f"  ✅ Good drug-likeness (QED: {summary['avg_qed']:.3f})")
        quality_score += 2
    elif summary["avg_qed"] > 0.3:
        print(f"  ⚠️  Moderate drug-likeness (QED: {summary['avg_qed']:.3f})")
        quality_score += 1
    else:
        print(f"  ❌ Poor drug-likeness (QED: {summary['avg_qed']:.3f})")
    max_score += 2

    # Lipinski compliance
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

    # Size distribution
    if 15 <= summary["avg_num_atoms"] <= 35:
        print(f"  ✅ Good size distribution ({summary['avg_num_atoms']:.1f} atoms)")
        quality_score += 1
    else:
        print(f"  ⚠️  Unusual size distribution ({summary['avg_num_atoms']:.1f} atoms)")
    max_score += 1

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
        for i, mol in enumerate(all_molecules):
            if mol is not None:
                row = {
                    "molecule_id": i,
                    "smiles": Chem.MolToSmiles(mol),
                    "molecular_weight": properties["molecular_weights"][i],
                    "logp": properties["logp_values"][i],
                    "hbd": properties["hbd_counts"][i],
                    "hba": properties["hba_counts"][i],
                    "rotatable_bonds": properties["rotatable_bonds"][i],
                    "num_atoms": properties["num_atoms"][i],
                    "qed": properties["qed_scores"][i],
                    "lipinski_rules": lipinski_scores[i],
                }
                if properties["sa_scores"][i] is not None:
                    row["sa_score"] = properties["sa_scores"][i]
                detailed_data.append(row)

        pd.DataFrame(detailed_data).to_csv(detailed_file, index=False)

        print(f"\n💾 Results saved to:")
        print(f"  Summary: {output_path}")
        print(f"  Detailed: {detailed_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze DiffSBDD generated molecules")
    parser.add_argument("results_dir", type=Path, help="Path to results directory")
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Output CSV file for results"
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.results_dir / "analysis_summary.csv"

    analyze_results(args.results_dir, args.output)


if __name__ == "__main__":
    main()
