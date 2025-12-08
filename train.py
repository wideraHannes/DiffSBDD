import argparse
from argparse import Namespace
from pathlib import Path
import warnings
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import yaml
import numpy as np

from lightning_modules import LigandPocketDDPM


class LossLoggingCallback(Callback):
    """Callback to print train/val loss to console at end of each epoch."""

    def _timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get("loss/train")
        if train_loss is not None:
            print(
                f"\n[{self._timestamp()}] [Epoch {trainer.current_epoch}] Train Loss: {train_loss:.6f}"
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get("loss/val")
        train_loss = metrics.get("loss/train")
        if val_loss is not None:
            ts = self._timestamp()
            epoch_str = f"[Epoch {trainer.current_epoch}]"
            train_str = f"Train: {train_loss:.6f}" if train_loss is not None else ""
            val_str = f"Val: {val_loss:.6f}"
            print(f"\n[{ts}] {epoch_str} {train_str} | {val_str}")


def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(
                f"Command line argument '{key}' (value: "
                f"{arg_dict[key]}) will be overwritten with value "
                f"{value} provided in the config file."
            )
        if isinstance(value, dict):
            arg_dict[key] = Namespace(**value)
        else:
            arg_dict[key] = value

    return args


def merge_configs(config, resume_config):
    """Merge resume_config into config, but config takes priority."""
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__
        if key not in config or config[key] is None:
            # Only use resume_config value if not specified in config
            config[key] = value
        # If key is in config with a non-None value, keep the config value
    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    assert "resume" not in config

    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume)
    if args.resume is not None:
        resume_config = torch.load(
            ckpt_path, map_location=torch.device("cpu"), weights_only=False
        )["hyper_parameters"]

        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)

    out_dir = Path(args.logdir, args.run_name)
    histogram_file = Path(args.datadir, "size_distribution.npy")
    histogram = np.load(histogram_file).tolist()
    pl_module = LigandPocketDDPM(
        outdir=out_dir,
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        virtual_nodes=args.virtual_nodes,
        esmc_path=getattr(args, "esmc_path", None),
        film_only_training=getattr(args, "film_only_training", False),
    )

    logger = pl.loggers.WandbLogger(
        save_dir=args.logdir,
        project="ligand-pocket-ddpm",
        group=args.wandb_params.group,
        name=args.run_name,
        id=args.run_name,
        resume="allow",  # Use 'allow' to resume if exists, start new otherwise
        entity=args.wandb_params.entity,
        mode=args.wandb_params.mode,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(out_dir, "checkpoints"),
        filename="best-model-epoch={epoch:02d}",
        monitor="loss/val",
        save_top_k=1,
        save_last=True,
        mode="min",
    )

    # Configure accelerator and devices based on GPU availability
    # Note: MPS (Apple Silicon) doesn't support float64, so we use CPU on Mac
    import platform

    if args.gpus > 0 and platform.system() != "Darwin":
        # Use CUDA GPU on Linux/Windows
        accelerator = "gpu"
        devices = args.gpus
        strategy = "ddp" if args.gpus > 1 else "auto"
    else:
        # Use CPU (including on Mac where MPS has float64 issues)
        accelerator = "cpu"
        devices = "auto"
        strategy = "auto"
        if args.gpus > 0 and platform.system() == "Darwin":
            print(
                "Note: Using CPU instead of MPS (Apple GPU) due to float64 compatibility issues"
            )

    loss_logging_callback = LossLoggingCallback()

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, loss_logging_callback],
        enable_progress_bar=args.enable_progress_bar,
        num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=getattr(args, "log_every_n_steps", 50),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
    )

    # For FiLM fine-tuning: load pretrained weights manually with strict=False
    # This allows missing FiLM keys and size mismatches to be handled
    if ckpt_path is not None and getattr(args, "film_only_training", False):
        print(f"\n=== FiLM Fine-Tuning Mode ===")
        print(f"Loading pretrained weights from: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]

        # Load with strict=False to allow missing FiLM keys
        missing, unexpected = pl_module.load_state_dict(state_dict, strict=False)

        print(f"Missing keys (expected for FiLM): {len(missing)}")
        for k in missing:
            print(f"  - {k}")
        print(f"Unexpected keys: {len(unexpected)}")

        # Verify FiLM is initialized to identity (gamma=1, beta=0)
        film = pl_module.ddpm.dynamics.film_network
        final_layer = film[-1]
        joint_nf = final_layer.out_features // 2
        gamma_vals = final_layer.bias.data[:joint_nf]
        beta_vals = final_layer.bias.data[joint_nf:]
        print(f"\nFiLM initialization check:")
        print(f"  Gamma mean: {gamma_vals.mean():.4f} (expected: 1.0000)")
        print(f"  Gamma std:  {gamma_vals.std():.4f} (expected: 0.0000)")
        print(f"  Beta mean:  {beta_vals.mean():.4f} (expected: 0.0000)")
        print(f"  Beta std:   {beta_vals.std():.4f} (expected: 0.0000)")

        # Verify initialization is correct
        if abs(gamma_vals.mean() - 1.0) < 0.01 and gamma_vals.std() < 0.01 and abs(beta_vals.mean()) < 0.01:
            print("✅ FiLM correctly initialized to identity!")
        else:
            print("⚠️  WARNING: FiLM NOT initialized to identity! Training may fail.")

        print("=== Starting FiLM-only training ===\n")

        # Don't pass ckpt_path to trainer - we already loaded weights
        trainer.fit(model=pl_module)
    else:
        # Normal training or full resume
        trainer.fit(model=pl_module, ckpt_path=ckpt_path)
