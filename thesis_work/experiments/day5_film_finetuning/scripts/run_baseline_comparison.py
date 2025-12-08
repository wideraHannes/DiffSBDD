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


def run_analysis(output_dir: Path, baseline_name: str) -> dict:
    """Run analyze_results.py on generated molecules.

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


def create_comparison_table(results: dict, output_file: Path):
    """Create comparison table of all baseline results."""
    print(f"\n{'='*70}")
    print("Creating Comparison Table")
    print(f"{'='*70}")

    # Prepare data
    data = []
    for baseline_name, metrics in results.items():
        if metrics:
            row = {"Baseline": baseline_name}
            row.update(metrics)
            data.append(row)

    if not data:
        print("⚠️  No results to compare")
        return

    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Comparison saved to: {output_file}")
    print()

    # Print table
    print("Baseline Comparison Results:")
    print(df.to_string(index=False))
    print()

    # Interpret results
    print("="*70)
    print("INTERPRETATION")
    print("="*70)

    # Extract key metrics (adjust based on what analyze_results.py returns)
    if len(data) >= 2:
        baseline1 = data[0]
        baseline2 = data[1] if len(data) > 1 else None
        baseline3 = data[2] if len(data) > 2 else None

        # Check connectivity (example - adjust key name based on actual output)
        conn_key = None
        for key in baseline1.keys():
            if "connect" in key.lower():
                conn_key = key
                break

        if conn_key:
            print("\n✓ Connectivity Check:")
            b1_conn = baseline1.get(conn_key, 0)
            print(f"  Baseline 1 (No FiLM): {b1_conn:.1%}")

            if baseline2:
                b2_conn = baseline2.get(conn_key, 0)
                diff = abs(b1_conn - b2_conn)
                print(f"  Baseline 2 (Identity): {b2_conn:.1%} (diff: {diff:.1%})")
                if diff < 0.05:
                    print("  ✅ Identity FiLM ≈ No FiLM (as expected)")
                else:
                    print("  ⚠️  Identity FiLM differs from No FiLM (unexpected!)")

            if baseline3:
                b3_conn = baseline3.get(conn_key, 0)
                print(f"  Baseline 3 (Random): {b3_conn:.1%}")
                if b3_conn < b1_conn * 0.5:
                    print("  ✅ Random FiLM << No FiLM (as expected)")
                else:
                    print("  ⚠️  Random FiLM not much worse (FiLM might not be active!)")
        print()


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

        # Analyze results
        metrics = run_analysis(output_dir, baseline["name"])
        results[baseline["name"]] = metrics

    # Create comparison
    if results:
        comparison_file = output_base / "baseline_comparison.csv"
        create_comparison_table(results, comparison_file)

        print("="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✅ Completed {len(results)}/{len(baselines_to_run)} baselines")
        print(f"\nResults saved to: {output_base}")
        print(f"Comparison table: {comparison_file}")
        print()
        print("Next steps:")
        print("  1. Review baseline_comparison.csv")
        print("  2. Check that Baseline 1 ≈ Baseline 2 (identity works)")
        print("  3. Check that Baseline 3 << Baseline 1 (FiLM is active)")
        print("  4. If all pass, proceed with FiLM training!")
        print()
    else:
        print("\n❌ No results generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
