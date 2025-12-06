# Day 3: Overfit Test Experiment

## Status: READY TO RUN (data fixed)

## Quick Start

```bash
# Run training
uv run python train.py --config thesis_work/experiments/day3_overfit/configs/day3_overfit_config.yml

# Check loss (while training or after)
uv run python thesis_work/experiments/day3_overfit/check_loss.py

# Recreate data (if needed)
uv run python thesis_work/experiments/day3_overfit/create_1sample_dataset.py
```

## Data Verified
- Ligand: 31 atoms (17 C, 8 N, 6 O)
- Pocket: 369 atoms
- Size distribution: 2D format (48, 646)

## What Was Wrong (Fixed)

The crossdock data format stores all atoms **concatenated** with a mask array:
```
lig_coords: (265, 3)   # ALL atoms from 10 samples concatenated
lig_mask:   (265,)     # Values 0-9 indicating which sample each atom belongs to
```

Old broken code did `data[:1]` = got 1 atom
Fixed code uses `data[mask == 0]` = gets all 31 atoms for sample 0

## Expected Results

After 50 epochs, if working correctly:
- Loss should decrease significantly
- Validity: 100%
- Connectivity: should improve (target: 100%)
- Generated molecule should have 31 atoms matching training sample

## Files
```
day3_overfit/
├── README.md
├── create_1sample_dataset.py    # Fixed dataset creation
├── configs/
│   └── day3_overfit_config.yml
├── data_1sample/                # READY - correct data
│   ├── train.npz (31 atoms)
│   ├── val.npz
│   ├── test.npz
│   ├── size_distribution.npy (2D)
│   └── esmc_embeddings/
└── outputs/                     # Empty - ready for training
```
