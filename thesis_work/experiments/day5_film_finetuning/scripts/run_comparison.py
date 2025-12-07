#!/usr/bin/env python
"""
Comparison script for baseline vs FiLM-finetuned model.

Runs test.py for a single checkpoint at a time.
Use --mode baseline or --mode experiment to select which to run.

Usage:
    uv run python thesis_work/experiments/day5_film_finetuning/scripts/run_comparison.py --mode baseline
    uv run python thesis_work/experiments/day5_film_finetuning/scripts/run_comparison.py --mode experiment
    uv run python thesis_work/experiments/day5_film_finetuning/scripts/run_comparison.py --mode both
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from configs.paths import (
    PROJECT_ROOT,
    BASELINE_CHECKPOINT,
    EXPERIMENT_CHECKPOINT,
    TEST_DATA_DIR,
    RESULTS_DIR,
    BASELINE_RESULTS,
    EXPERIMENT_RESULTS,
)

# Generation parametersf
N_SAMPLES = 1  # Number of molecules to generate per pocket
BATCH_SIZE = (
    1  # Batch size for generation (should be >= N_SAMPLES to minimize iterations)
)


def run_test(
    checkpoint: Path, outdir: Path, name: str, init_film_identity: bool = False
) -> bool:
    """Run test.py for a checkpoint.

    Args:
        checkpoint: Path to the model checkpoint.
        outdir: Output directory for generated molecules.
        name: Display name for logging.
        init_film_identity: If True, initialize FiLM to identity (for baseline).
    """
    print(f"\n{'=' * 60}")
    print(f"Running {name}")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {outdir}")
    print(f"Test dir: {TEST_DATA_DIR}")
    print(f"N samples: {N_SAMPLES}")
    print(f"Init FiLM identity: {init_film_identity}")
    print()

    # Check checkpoint exists
    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        return False

    # Check test dir exists
    if not TEST_DATA_DIR.exists():
        print(f"ERROR: Test directory not found: {TEST_DATA_DIR}")
        return False

    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "test.py"),
        str(checkpoint),
        "--test_dir",
        str(TEST_DATA_DIR),
        "--outdir",
        str(outdir),
        "--n_samples",
        str(N_SAMPLES),
        "--batch_size",
        str(BATCH_SIZE),
        "--sanitize",
        "--skip_existing",  # Allow rerunning with existing output dir
    ]

    # Add identity initialization for baseline
    if init_film_identity:
        cmd.append("--init_film_identity")

    print(f"Command: {' '.join(cmd)}")
    print()

    # Run
    try:
        subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        print(f"\n{name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: {name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\nERROR: {name} failed with exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation for baseline or experiment checkpoint"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "experiment", "both"],
        required=True,
        help="Which evaluation to run: 'baseline', 'experiment', or 'both'",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"EVALUATION MODE: {args.mode.upper()}")
    print("=" * 60)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Results directory: {RESULTS_DIR}")
    print()

    # Create results directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.mode in ["baseline", "both"]:
        BASELINE_RESULTS.mkdir(parents=True, exist_ok=True)
        results["baseline"] = run_test(
            checkpoint=BASELINE_CHECKPOINT,
            outdir=BASELINE_RESULTS,
            name="BASELINE (pretrained DiffSBDD)",
            init_film_identity=True,  # Initialize FiLM to identity for fair comparison
        )

    if args.mode in ["experiment", "both"]:
        EXPERIMENT_RESULTS.mkdir(parents=True, exist_ok=True)
        results["experiment"] = run_test(
            checkpoint=EXPERIMENT_CHECKPOINT,
            outdir=EXPERIMENT_RESULTS,
            name="EXPERIMENT (FiLM fine-tuned)",
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{name.capitalize():12s}: {status}")
    print()
    print("Results saved to:")
    if "baseline" in results:
        print(f"  Baseline:   {BASELINE_RESULTS}")
    if "experiment" in results:
        print(f"  Experiment: {EXPERIMENT_RESULTS}")
    print()

    if all(results.values()):
        print("Next steps:")
        print("  1. Compare generated molecules in results/*/processed/")
        print("  2. Run analysis metrics (validity, QED, SA, docking)")
        print("  3. Visualize representative molecules")


if __name__ == "__main__":
    main()
