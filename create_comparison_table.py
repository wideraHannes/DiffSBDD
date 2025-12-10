#!/usr/bin/env python3
"""
Create comparison table across all baseline experiments
Similar to Table in DiffSBDD paper (Wasserstein distances)
"""

import pandas as pd
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_comparison_table(baseline_dirs, output_file):
    """
    Create comparison table from multiple baseline runs

    Args:
        baseline_dirs: dict mapping baseline names to result directories
        output_file: Path to save comparison table
    """

    results = []

    for baseline_name, result_dir in baseline_dirs.items():
        # Load analysis summary
        summary_file = Path(result_dir) / "analysis_summary.csv"
        if not summary_file.exists():
            logging.warning(f"Skipping {baseline_name}: no summary file at {summary_file}")
            continue

        logging.info(f"Loading results for: {baseline_name}")
        df = pd.read_csv(summary_file)

        # Extract key metrics
        row = {"Model": baseline_name}

        # Wasserstein distances (lower is better)
        if "qed_wasserstein" in df.columns:
            row["QED (W↓)"] = df["qed_wasserstein"].values[0]
        if "sa_wasserstein" in df.columns:
            row["SA (W↓)"] = df["sa_wasserstein"].values[0]
        if "logp_wasserstein" in df.columns:
            row["LogP (W↓)"] = df["logp_wasserstein"].values[0]
        if "molwt_wasserstein" in df.columns:
            row["MolWt (W↓)"] = df["molwt_wasserstein"].values[0]
        if "numatoms_wasserstein" in df.columns:
            row["NumAtoms (W↓)"] = df["numatoms_wasserstein"].values[0]
        if "vina_wasserstein" in df.columns:
            row["Vina (W↓)"] = df["vina_wasserstein"].values[0]

        # Validity metrics
        if "validity" in df.columns:
            row["Validity"] = df["validity"].values[0]
        if "connectivity" in df.columns:
            row["Connectivity"] = df["connectivity"].values[0]
        if "uniqueness" in df.columns:
            row["Uniqueness"] = df["uniqueness"].values[0]
        if "diversity" in df.columns:
            row["Diversity"] = df["diversity"].values[0]

        # Drug-likeness averages
        if "avg_qed" in df.columns:
            row["QED (avg)"] = df["avg_qed"].values[0]
        if "avg_sa_score" in df.columns:
            row["SA (avg)"] = df["avg_sa_score"].values[0]
        if "avg_logp" in df.columns:
            row["LogP (avg)"] = df["avg_logp"].values[0]
        if "avg_vina_score" in df.columns:
            row["Vina (avg)"] = df["avg_vina_score"].values[0]

        results.append(row)

    if not results:
        logging.error("No valid baseline results found!")
        return None

    # Create DataFrame
    comparison_df = pd.DataFrame(results)

    # Sort by QED Wasserstein distance (best first) if available
    if "QED (W↓)" in comparison_df.columns:
        comparison_df = comparison_df.sort_values("QED (W↓)")

    # Save to CSV
    comparison_df.to_csv(output_file, index=False, float_format="%.4f")

    # Print formatted table
    print("\n" + "=" * 120)
    print("BASELINE COMPARISON TABLE")
    print("=" * 120)
    print(
        "\nWasserstein Distances (↓ lower is better = closer to ground truth distribution)"
    )
    print("=" * 120)

    # Print with nice formatting
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(comparison_df.to_string(index=False))
    print("=" * 120)

    logging.info(f"\n✅ Comparison table saved to: {output_file}")

    return comparison_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create comparison table from baseline experiment results"
    )
    parser.add_argument(
        "--baseline_dir",
        type=Path,
        required=True,
        help="Directory containing all baseline result subdirectories",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Output CSV file path"
    )
    args = parser.parse_args()

    # Auto-detect baseline subdirectories
    baseline_dir = Path(args.baseline_dir)
    if not baseline_dir.exists():
        logging.error(f"Baseline directory not found: {baseline_dir}")
        exit(1)

    baseline_dirs = {d.name: d for d in baseline_dir.iterdir() if d.is_dir()}

    if not baseline_dirs:
        logging.error(f"No baseline subdirectories found in {baseline_dir}")
        exit(1)

    logging.info(f"Found {len(baseline_dirs)} baseline experiments: {list(baseline_dirs.keys())}")

    if args.output is None:
        args.output = baseline_dir / "comparison_table.csv"

    create_comparison_table(baseline_dirs, args.output)
