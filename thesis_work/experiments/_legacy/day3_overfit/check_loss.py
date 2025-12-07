#!/usr/bin/env python3
"""Check training loss from checkpoint."""
import torch
from pathlib import Path

ckpt_dir = Path("thesis_work/experiments/day3_overfit/outputs/day3_overfit_esmc/checkpoints")

for ckpt_path in sorted(ckpt_dir.glob("*.ckpt")):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch', '?')

    best_loss = None
    for k, v in ckpt.get('callbacks', {}).items():
        if 'best_model_score' in v:
            best_loss = v['best_model_score']

    print(f"{ckpt_path.name}: epoch={epoch}, best_val_loss={best_loss}")
