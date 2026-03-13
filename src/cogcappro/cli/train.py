"""Training CLI for the refactored open-source runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from ..runtime.paths import dump_resolved_config, load_runtime_config, prepare_runtime_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cogcappro.yaml", help="path to config which constructs model")
    parser.add_argument("--seed", type=int, default=0, help="the seed (for reproducible sampling)")
    parser.add_argument("--subjects", type=str, default="sub-08", help="the subjects")
    parser.add_argument("--exp_setting", type=str, default="intra-subject", help="the exp_setting")
    parser.add_argument("--epoch", type=int, default=80, help="train epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="lr")
    parser.add_argument("--brain_backbone", type=str, help="brain_backbone")
    parser.add_argument("--vision_backbone", default="RN50", type=str, help="vision_backbone")
    parser.add_argument("--c", type=int, default=6, help="c")
    parser.add_argument("--selected_region", type=str, default=None, help="Select channels from a specific brain region")
    parser.add_argument("--uncertainty_aware", action="store_true", help="Enable uncertainty-aware training")
    parser.add_argument("--mask_count", type=int, default=1, help="Number of modalities to mask")
    parser.add_argument("--staged_training", action="store_true", help="Enable staged training")
    parser.add_argument("--devices", type=str, default="0,1", help="GPU ids, comma separated, e.g. 0,1,2,3 ")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save experiment outputs")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Checkpoint path for pretrained model")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="ClipLoss_Modified_DDP",
        choices=["ClipLoss_Original", "ClipLoss_Modified_DDP"],
        help="Loss function type",
    )
    parser.add_argument(
        "--filter_band",
        type=str,
        default=None,
        choices=[None, "delta", "theta", "alpha", "beta", "gamma"],
        help="Filter band: delta (0.5-4Hz), theta (4-8Hz), alpha (8-13Hz), beta (13-30Hz), gamma (30-100Hz)",
    )
    parser.add_argument("--data_type", type=str, default="EEG", choices=["EEG", "MEG"], help="Data type: EEG or MEG")
    parser.add_argument("--data_root", type=str, default=None, help="Optional data root override")
    parser.add_argument("--weights_root", type=str, default=None, help="Optional weights root override")
    parser.add_argument("--runs_root", type=str, default=None, help="Optional runs root override")
    parser.add_argument("--print_config", action="store_true", help="Print the resolved runtime config and exit")
    return parser


def prepare_config(args: argparse.Namespace):
    config = load_runtime_config(args.config)
    return prepare_runtime_config(args, config)


def run(args: argparse.Namespace):
    config = prepare_config(args)
    if args.print_config:
        print(dump_resolved_config(config))
        return config

    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.strategies import DDPStrategy

    from ..training.module import load_data, load_model

    seed_everything(args.seed)

    save_dir = config.get("save_dir") or "runs"
    os.makedirs(save_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir)
    os.makedirs(logger.log_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(logger.log_dir, Path(args.config).name))

    train_loader, val_loader, test_loader = load_data(config)
    pl_model = load_model(config, train_loader, test_loader)

    checkpoint_callback = ModelCheckpoint(save_last=True)
    if config["exp_setting"] == "inter-subject":
        early_stop_callback = EarlyStopping(
            monitor="val_top1_acc_fusion",
            min_delta=0.001,
            patience=50,
            verbose=False,
            mode="max",
        )
    else:
        early_stop_callback = EarlyStopping(
            monitor="total_loss",
            min_delta=0.001,
            patience=50,
            verbose=False,
            mode="min",
        )

    trainer = Trainer(
        log_every_n_steps=10,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=config["train"]["epoch"],
        devices=config["devices"],
        accelerator="cuda",
        num_nodes=1,
        sync_batchnorm=True,
        logger=logger,
    )

    ckpt_path = args.pretrained_ckpt if args.pretrained_ckpt is not None else "last"
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    if config["exp_setting"] == "inter-subject":
        test_results = trainer.test(ckpt_path="best", dataloaders=test_loader)
    else:
        test_results = trainer.test(ckpt_path="last", dataloaders=test_loader)

    with open(os.path.join(logger.log_dir, "test_results.json"), "w", encoding="utf-8") as file:
        json.dump(test_results, file, indent=4)
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
