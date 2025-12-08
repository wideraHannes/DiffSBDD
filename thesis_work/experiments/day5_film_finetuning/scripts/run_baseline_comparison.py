#!/usr/bin/env python
"""
Run all three baseline experiments and compare results.

This script:
1. Loads the three baseline configs
2. Runs evaluation (test.py) for each baseline
3. Analyzes results and creates comparison table

Baselines:
- Baseline 1: Pretrained without FiLM (ground truth)
- Baseline 2: Identity FiLM (no-op verification)
- Baseline 3: Random FiLM (negative control)

Usage:
    uv run python thesis_work/experiments/day5_film_finetuning/scripts/run_baseline_comparison.py
    uv run python thesis_work/experiments/day5_film_finetuning/scripts/run_baseline_comparison.py --n_samples 50
    uv run python thesis_work/experiments/day5_film_finetuning/scripts/run_baseline_comparison.py --skip_test  # Only analyze existing results
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "thesis_work/experiments/day5_film_finetuning/configs"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/crossdocked_fullatom_cond.ckpt"

BASELINES = [
    {
        "name": "Baseline 1: No FiLM",
        "config": "baseline1_pretrained_no_film.yml",
        "short_name": "no_film",
        "use_film": False,
        "film_mode": "identity",
    },
    {
        "name": "Baseline 2: Identity FiLM",
        "config": "baseline2_identity_film.yml",
        "short_name": "identity",
        "use_film": True,
        "film_mode": "identity",
    },
    {
        "name": "Baseline 3: Random FiLM",
        "config": "baseline3_random_film.yml",
        "short_name": "random",
        "use_film": True,
        "film_mode": "random",
    },
]


def load_config(config_file: Path) -> dict:
    """Load YAML config file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def run_test_baseline(
    baseline: dict,
    checkpoint: Path,
    test_dir: Path,
    output_dir: Path,
    n_samples: int,
    batch_size: int,
) -> bool:
    """Run test.py for a single baseline configuration.

    Args:
        baseline: Baseline configuration dict
        checkpoint: Path to pretrained checkpoint
        test_dir: Directory containing test data
        output_dir: Output directory for results
        n_samples: Number of samples to generate per pocket
        batch_size: Batch size for generation

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Running: {baseline['name']}")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output dir: {output_dir}")
    print(f"Test dir: {test_dir}")
    print(f"N samples: {n_samples}")
    print(f"Use FiLM: {baseline['use_film']}")
    print(f"FiLM mode: {baseline['film_mode']}")
    print()

    # Check checkpoint exists
    if not checkpoint.exists():
        print(f"❌ ERROR: Checkpoint not found: {checkpoint}")
        return False

    # Check test dir exists
    if not test_dir.exists():
        print(f"❌ ERROR: Test directory not found: {test_dir}")
        return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "test.py"),
        str(checkpoint),
        "--test_dir",
        str(test_dir),
        "--outdir",
        str(output_dir),
        "--n_samples",
        str(n_samples),
        "--batch_size",
        str(batch_size),
        "--sanitize",
        "--skip_existing",
    ]

    # Add FiLM configuration
    if baseline["use_film"]:
        cmd.append("--use_film")
        cmd.extend(["--film_mode", baseline["film_mode"]])

    print(f"Command: {' '.join(cmd)}")
    print()

    # Run
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(f"✅ {baseline['name']} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Failed with exit code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Exception occurred: {e}")
        return False


def run_analysis(output_dir: Path, baseline_name: str, ground_truth_file: Path = None) -> dict:
    """Run analyze_results.py on generated molecules.

    Args:
        output_dir: Directory with generated molecules
        baseline_name: Name of baseline for logging
        ground_truth_file: Optional path to ground truth properties CSV

    Returns:
        Dict with metrics if successful, None otherwise
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {baseline_name}")
    print(f"{'='*70}")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "analyze_results.py"),
        str(output_dir),
    ]

    # Add ground truth file if provided
    if ground_truth_file and ground_truth_file.exists():
        cmd.extend(["--ground_truth", str(ground_truth_file)])
        print(f"Ground truth: {ground_truth_file}")

    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), check=True, capture_output=True, text=True
        )
        print(result.stdout)
        print(f"✅ {baseline_name} analysis completed!")

        # Try to parse metrics from analysis_summary.csv
        csv_path = output_dir / "analysis_summary.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            metrics = df.iloc[0].to_dict() if len(df) > 0 else {}
            return metrics
        return {}
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Analysis failed with exit code {e.returncode}")
        print(f"STDERR:\n{e.stderr}")
        return {}
    except Exception as e:
        print(f"❌ ERROR: Analysis exception: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Run all three baseline experiments and compare results"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of molecules to generate per pocket (default: 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation (default: 1)",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/dummy_dataset/test",
        help="Test data directory (default: data/dummy_dataset/test)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="thesis_work/experiments/day5_film_finetuning/outputs/baseline_comparison",
        help="Output directory for all results",
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Skip testing, only analyze existing results",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which baseline(s) to run (default: all)",
    )
    args = parser.parse_args()

    print("="*70)
    print("FiLM BASELINE COMPARISON")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Test dir: {args.test_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"N samples: {args.n_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Skip testing: {args.skip_test}")
    print()

    test_dir = PROJECT_ROOT / args.test_dir
    output_base = PROJECT_ROOT / args.output_dir
    output_base.mkdir(parents=True, exist_ok=True)

    # Prepare ground truth analysis
    print("="*70)
    print("PREPARING GROUND TRUTH ANALYSIS")
    print("="*70)

    dataset_dir = test_dir.parent  # Go from test/ to dataset root
    ground_truth_file = dataset_dir / "analysis" / "test_ground_truth_properties.csv"

    if not ground_truth_file.exists():
        print(f"Ground truth not found at: {ground_truth_file}")
        print("Generating ground truth analysis...")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "prepare_dataset_analysis.py"),
                    str(dataset_dir),
                    "--split",
                    "test",
                ],
                cwd=str(PROJECT_ROOT),
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            print("✅ Ground truth analysis generated")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to generate ground truth: {e.stderr}")
            ground_truth_file = None
    else:
        print(f"✅ Ground truth found: {ground_truth_file}")

    print()

    # Select baselines to run
    if args.baseline == "all":
        baselines_to_run = BASELINES
    else:
        idx = int(args.baseline) - 1
        baselines_to_run = [BASELINES[idx]]

    # Run each baseline
    results = {}
    for baseline in baselines_to_run:
        output_dir = output_base / baseline["short_name"]

        if not args.skip_test:
            success = run_test_baseline(
                baseline=baseline,
                checkpoint=CHECKPOINT_PATH,
                test_dir=test_dir,
                output_dir=output_dir,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
            )
            if not success:
                print(f"⚠️  {baseline['name']} failed, skipping analysis")
                continue

        # Analyze results with ground truth
        metrics = run_analysis(output_dir, baseline["name"], ground_truth_file)
        results[baseline["name"]] = metrics

    # Create comparison table
    if results:
        print("="*70)
        print("CREATING COMPARISON TABLE")
        print("="*70)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "create_comparison_table.py"),
                    "--baseline_dir",
                    str(output_base),
                    "--output",
                    str(output_base / "comparison_table.csv"),
                ],
                cwd=str(PROJECT_ROOT),
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            print()
            print("="*70)
            print("✅ BASELINE COMPARISON COMPLETE")
            print("="*70)
            print(f"Results: {output_base}")
            print(f"Table: {output_base / 'comparison_table.csv'}")
            print()
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Comparison table creation failed: {e.stderr}")
            sys.exit(1)
    else:
        print("\n❌ No results generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
