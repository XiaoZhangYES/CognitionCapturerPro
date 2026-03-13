"""Dataset helpers for the refactored align runtime."""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Any

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import torch
from torch.utils.data import DataLoader

from .diffusion_pipe import EmbeddingDataset
from ..data.eeg import load_data as load_eeg_data
from ..runtime.paths import (
    load_runtime_config,
    merge_with_public_config,
    prepare_runtime_config,
)


def _raw_nested_get(raw_config: dict[str, Any], *keys: str, default=None):
    current = raw_config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if isinstance(current, str) and "${" in current:
        return default
    return current


def build_align_config(
    config_path: str,
    subject: str,
    exp_setting: str = "intra-subject",
    seed: int = 20200220,
    devices: list[int] | None = None,
    similarity_dir: str | None = None,
    vision_backbone: str = "ViT-H-14",
    brain_backbone: str = "EEGProject",
    c: int = 6,
    epoch: int | None = None,
    lr: float | None = None,
) -> Any:
    """Resolve a training config into the same runtime shape used by the new CLI."""
    devices = devices or [0]
    raw_config = load_runtime_config(config_path)
    data_type = str(raw_config.get("data_type", "EEG")).upper()
    config = merge_with_public_config(raw_config, data_type=data_type)
    raw_merged = OmegaConf.to_container(config, resolve=False)

    original_selected_ch = _raw_nested_get(raw_merged, "data", "selected_ch", default=False)
    original_c_num = _raw_nested_get(raw_merged, "models", "brain", "params", "c_num", default=None)
    original_save_dir = config.get("save_dir")

    runtime_args = Namespace(
        config=config_path,
        seed=seed,
        subjects=subject,
        exp_setting=exp_setting,
        epoch=epoch if epoch is not None else int(_raw_nested_get(raw_merged, "train", "epoch", default=80)),
        lr=lr if lr is not None else float(_raw_nested_get(raw_merged, "train", "lr", default=1e-4)),
        brain_backbone=brain_backbone,
        vision_backbone=vision_backbone,
        c=c,
        selected_region=original_selected_ch if isinstance(original_selected_ch, str) else None,
        uncertainty_aware=bool(_raw_nested_get(raw_merged, "data", "uncertainty_aware", default=True)),
        mask_count=int(_raw_nested_get(raw_merged, "train", "mask_count", default=1)),
        staged_training=bool(_raw_nested_get(raw_merged, "train", "staged_training", default=False)),
        devices=",".join(str(device) for device in devices),
        save_dir=original_save_dir,
        pretrained_ckpt=None,
        loss_type=str(_raw_nested_get(raw_merged, "train", "loss_type", default="ClipLoss_Modified_DDP")),
        filter_band=_raw_nested_get(raw_merged, "filter_band", default=None),
        data_type=data_type,
        data_root=None,
        weights_root=None,
        runs_root=None,
        print_config=False,
    )

    config = prepare_runtime_config(runtime_args, config)
    if original_c_num is not None:
        config.models.brain.params.c_num = original_c_num
    if original_selected_ch not in (None, False):
        config.data.selected_ch = original_selected_ch
    if similarity_dir is not None:
        config.similarity_dir = similarity_dir
    return config


def load_eeg_dataset(
    config_path: str,
    subjects: list[str],
    mode: str = "train",
    exp_setting: str = "intra-subject",
    seed: int = 20200220,
    devices: list[int] | None = None,
    similarity_dir: str | None = None,
    vision_backbone: str = "ViT-H-14",
    brain_backbone: str = "EEGProject",
    c: int = 6,
    epoch: int | None = None,
    lr: float | None = None,
):
    """Load EEG loaders with the same ordering assumptions as the historical align workflow."""
    del mode
    seed_everything(seed)
    config = build_align_config(
        config_path=config_path,
        subject=subjects[0],
        exp_setting=exp_setting,
        seed=seed,
        devices=devices,
        similarity_dir=similarity_dir,
        vision_backbone=vision_backbone,
        brain_backbone=brain_backbone,
        c=c,
        epoch=epoch,
        lr=lr,
    )
    config.data.subjects = subjects
    train_loader, val_loader, test_loader = load_eeg_data(config, shuffle_train=False)
    return train_loader, val_loader, test_loader


def load_diffusion_embeddings(embedding_path: str, img_paths: list[str]) -> dict[str, torch.Tensor]:
    """Load multimodal embeddings from generated embedding files in image-path order."""
    full_embeddings = torch.load(embedding_path, map_location="cpu")
    modalities = list(full_embeddings.keys())
    h_embeddings = {mod: [] for mod in modalities}

    for img_path in img_paths:
        img_filename = os.path.basename(img_path)
        for mod in modalities:
            if img_filename not in full_embeddings[mod]:
                raise ValueError(f"Embedding not found for {mod} modality of {img_filename} in {embedding_path}")
            h_embeddings[mod].append(full_embeddings[mod][img_filename])

    for mod in modalities:
        h_embeddings[mod] = torch.stack(h_embeddings[mod], dim=0)

    return h_embeddings


def prepare_embedding_dataset(
    pl_model,
    train_loader,
    device: str,
    h_embeddings: dict[str, torch.Tensor],
    save_path: str | None = None,
):
    """Prepare multimodal embedding datasets for align training."""
    pl_model.eval()
    c_embeddings = {mod: [] for mod in list(h_embeddings.keys()) + ["fusion"]}
    img_paths: list[str] = []

    with torch.no_grad():
        for batch in train_loader:
            processed_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            eeg_z, _, _ = pl_model(processed_batch)
            img_paths.extend(batch["img_path"])
            for modality in c_embeddings.keys():
                c_embeddings[modality].append(eeg_z[modality].cpu())

    for modality in c_embeddings.keys():
        c_embeddings[modality] = torch.cat(c_embeddings[modality], dim=0)
        if modality != "fusion":
            assert len(c_embeddings[modality]) == len(h_embeddings[modality]), (
                f"{modality} conditional embedding sample count inconsistent with target embedding"
            )

    h_embeddings = {
        mod: emb.float().cpu() if not isinstance(emb, torch.Tensor) else emb.float().cpu()
        for mod, emb in h_embeddings.items()
    }
    h_embeddings["fusion"] = c_embeddings["fusion"]
    dataset = EmbeddingDataset(c_embeddings, h_embeddings)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"dataset": dataset, "img_paths": img_paths}, save_path)
        print(f"Dataset saved to: {save_path}, containing {len(img_paths)} samples")

    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    return dataloader, dataset, img_paths
