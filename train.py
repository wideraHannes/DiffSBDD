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
        train_loss = metrics.get('loss/train')
        if train_loss is not None:
            print(f"\n[{self._timestamp()}] [Epoch {trainer.current_epoch}] Train Loss: {train_loss:.6f}")

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get('loss/val')
        train_loss = metrics.get('loss/train')
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
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__
        if key in config and config[key] != value:
            warnings.warn(
                f"Config parameter '{key}' (value: "
                f"{config[key]}) will be overwritten with value "
                f"{value} from the checkpoint."
            )
        config[key] = value
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
        resume_config = torch.load(ckpt_path, map_location=torch.device("cpu"))[
            "hyper_parameters"
        ]

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
    )

    logger = pl.loggers.WandbLogger(
        save_dir=args.logdir,
        project="ligand-pocket-ddpm",
        group=args.wandb_params.group,
        name=args.run_name,
        id=args.run_name,
        resume="must" if args.resume is not None else False,
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
    if args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
        strategy = "ddp" if args.gpus > 1 else "auto"
    else:
        accelerator = "cpu"
        devices = "auto"
        strategy = "auto"

    loss_logging_callback = LossLoggingCallback()

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, loss_logging_callback],
        enable_progress_bar=args.enable_progress_bar,
        num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=getattr(args, 'log_every_n_steps', 50),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
    )

    trainer.fit(model=pl_module, ckpt_path=ckpt_path)
