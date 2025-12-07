from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# ============================================================================
# DATA PATHS
# ============================================================================
CROSSDOCK_DATA_DIR = PROJECT_ROOT / "data" / "processed_crossdock_noH_full_fixed"
CROSSDOCK_TRAIN = CROSSDOCK_DATA_DIR / "train"
CROSSDOCK_VAL = CROSSDOCK_DATA_DIR / "val"
CROSSDOCK_TEST = CROSSDOCK_DATA_DIR / "test"

# Test data with structure files (PDB, SDF, TXT) for molecule generation
# Uses the test split from processed data (requires PDB, SDF, TXT files)
TEST_DATA_DIR = CROSSDOCK_TEST

# ============================================================================
# CHECKPOINT PATHS
# ============================================================================
BASELINE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "crossdocked_fullatom_cond.ckpt"
EXPERIMENT_CHECKPOINT = (
    PROJECT_ROOT
    / "thesis_work"
    / "experiments"
    / "day5_film_finetuning"
    / "outputs"
    / "film-finetuning-v2"
    / "checkpoints"
    / "last.ckpt"
)

# ============================================================================
# RESULTS PATHS
# ============================================================================
RESULTS_DIR = (
    PROJECT_ROOT / "thesis_work" / "experiments" / "day5_film_finetuning" / "results"
)
BASELINE_RESULTS = RESULTS_DIR / "baseline"
EXPERIMENT_RESULTS = RESULTS_DIR / "experiment"
