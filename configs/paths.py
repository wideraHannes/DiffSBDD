from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()


CROSSDOCK_DATA_DIR = PROJECT_ROOT.parent / "data" / "processed_crossdock_noH_full_temp"
CROSSDOCK_TRAIN = CROSSDOCK_DATA_DIR / "train"
CROSSDOCK_VAL = CROSSDOCK_DATA_DIR / "val"
CROSSDOCK_TEST = CROSSDOCK_DATA_DIR / "test"
