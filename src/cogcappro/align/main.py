"""CLI for the refactored align runtime."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import re
from typing import Dict

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    build_align_config,
    load_diffusion_embeddings,
    load_eeg_dataset,
    prepare_embedding_dataset,
)
from .evaluation import evaluate_eeg_accuracy
from .model import (
    init_diffusion_prior,
    init_simple_align_prior,
    load_pl_model_from_checkpoint,
)
from ..runtime.paths import PRETRAIN_MAP, resolve_diffusion_embeddings_root


def _raw_nested_get(raw_config: dict, *keys: str, default=None):
    current = raw_config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if isinstance(current, str) and "${" in current:
        return default
    return current


def generate_dataset_with_diffusion(
    diffusion_pipe,
    pl_model,
    data_loader: torch.utils.data.DataLoader,
    img_paths: list[str],
    device: str,
    save_dir: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    output_suffix: str = "",
) -> Dict[str, Dict[str, torch.Tensor]]:
    embeddings: Dict[str, Dict[str, torch.Tensor]] = {
        "image": {},
        "depth": {},
        "edge": {},
        "image_before": {},
        "depth_before": {},
        "edge_before": {},
    }

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"generated_embeddings{output_suffix}.pt")

    pl_model.eval()
    pipe_net = diffusion_pipe.net if hasattr(diffusion_pipe, "net") else diffusion_pipe.diffusion_prior
    pipe_net.eval()

    sample_idx = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating multi-modal embeddings"):
            processed_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            eeg_z, _, _ = pl_model(processed_batch)

            condition_embeds = {mod: eeg_z[mod].to(device) for mod in diffusion_pipe.modalities if mod in eeg_z}
            if "fusion" in eeg_z:
                condition_embeds["fusion"] = eeg_z["fusion"].to(device)

            for mod in condition_embeds:
                if mod == "fusion":
                    continue
                for i in range(condition_embeds[mod].shape[0]):
                    img_filename = os.path.basename(img_paths[sample_idx + i])
                    embeddings[f"{mod}_before"][img_filename] = condition_embeds[mod][i].cpu().float()

            generated_embeds = diffusion_pipe.generate(
                condition_embeds=condition_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

            batch_size = next(iter(generated_embeds.values())).shape[0]
            for i in range(batch_size):
                img_filename = os.path.basename(img_paths[sample_idx + i])
                for mod in ["image", "depth", "edge"]:
                    if mod in generated_embeds:
                        embeddings[mod][img_filename] = generated_embeds[mod][i].cpu().float()
            sample_idx += batch_size

        if embeddings["image"]:
            test_fn = next(iter(embeddings["image"].keys()))
            print(
                f"Validation output - {test_fn} | "
                f"image: {embeddings['image'][test_fn].shape} | "
                f"depth: {embeddings['depth'][test_fn].shape} | "
                f"edge: {embeddings['edge'][test_fn].shape}"
            )

        torch.save(embeddings, save_path)
        print(f"\nGenerated embedding dataset saved to: {save_path}")

    return embeddings


def load_eeg_pl_model(
    exp_dir: str,
    subject: str,
    seed: int,
    devices: list[int],
    lr: float,
    exp_setting: str,
    brain_backbone: str,
    vision_backbone: str,
    z_dim: int = 1024,
    epoch: int = 80,
    staged_training: bool = False,
    yaml_path: str | None = None,
):
    del z_dim, staged_training
    if yaml_path is None or not os.path.exists(yaml_path):
        raise FileNotFoundError(yaml_path)

    config = build_align_config(
        config_path=yaml_path,
        subject=subject,
        exp_setting=exp_setting,
        seed=seed,
        devices=devices,
        vision_backbone=vision_backbone,
        brain_backbone=brain_backbone,
        c=6,
        epoch=epoch,
        lr=lr,
    )
    device = f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu"
    config.devices = devices
    config.z_dim = 1024

    train_loader, val_loader, test_loader = load_eeg_dataset(
        config_path=yaml_path,
        subjects=[subject],
        mode="test",
        exp_setting=exp_setting,
        seed=seed,
        devices=devices,
        brain_backbone=brain_backbone,
        vision_backbone=vision_backbone,
        similarity_dir=config.get("similarity_dir"),
        c=6,
        epoch=epoch,
        lr=lr,
    )

    pl_model = load_pl_model_from_checkpoint(
        config=config,
        train_loader=train_loader,
        test_loader=test_loader,
        checkpoint_path=find_best_ckpt(os.path.join(exp_dir, "checkpoints")),
        device=device,
    )
    return pl_model, config, train_loader, val_loader, test_loader


def parse_exp_dir(exp_dir: str):
    """Parse exp_setting, subject, seed from the experiment path."""
    exp_setting = None
    for part in exp_dir.split(os.sep):
        if part.startswith("intra-subject"):
            exp_setting = "intra-subject"
            break
        if part.startswith("inter-subject"):
            exp_setting = "inter-subject"
            break
    if exp_setting is None:
        raise ValueError(f"Neither intra-subject nor inter-subject found in path: {exp_dir}")

    basename = os.path.basename(os.path.normpath(exp_dir))
    match = re.match(r"(sub-\d+)_seed(\d+)", basename)
    if not match:
        raise ValueError(f"Directory name format incorrect, cannot parse sub and seed: {basename}")
    subject, seed = match.group(1), int(match.group(2))

    return exp_setting, subject, seed


def _infer_backbones_from_exp_name(exp_dir: str) -> tuple[str | None, str | None]:
    exp_part = os.path.basename(os.path.dirname(os.path.normpath(exp_dir)))
    tokens = exp_part.split("_")

    vision_backbone = None
    vision_idx = None
    for idx, token in enumerate(tokens):
        if token in PRETRAIN_MAP:
            vision_backbone = token
            vision_idx = idx
            break

    if vision_idx is None:
        return None, None

    brain_tokens = tokens[:vision_idx]
    for marker in ["final", "cogcappro"]:
        if marker in brain_tokens:
            brain_tokens = brain_tokens[brain_tokens.index(marker) + 1 :]
    brain_backbone = "_".join(brain_tokens) if brain_tokens else None
    return brain_backbone, vision_backbone


def parse_backbones(config, exp_dir: str) -> tuple[str, str]:
    if isinstance(config, dict):
        raw_config = config
    else:
        raw_config = OmegaConf.to_container(config, resolve=False)
    brain_backbone = raw_config.get("brain_backbone")
    vision_backbone = raw_config.get("vision_backbone")
    data_section = raw_config.get("data", {})
    models_section = raw_config.get("models", {})
    brain_section = models_section.get("brain", {})
    target = brain_section.get("target")

    if isinstance(vision_backbone, str) and "${" in vision_backbone:
        vision_backbone = None
    if vision_backbone is None:
        model_type = data_section.get("model_type")
        if isinstance(model_type, str) and "${" not in model_type:
            vision_backbone = model_type

    if isinstance(brain_backbone, str) and "${" in brain_backbone:
        brain_backbone = None
    if brain_backbone is None and isinstance(target, str) and "${" not in target:
        brain_backbone = target.split(".")[-1]

    inferred_brain, inferred_vision = _infer_backbones_from_exp_name(exp_dir)
    if brain_backbone is None:
        brain_backbone = inferred_brain
    if vision_backbone is None:
        vision_backbone = inferred_vision

    if brain_backbone is None or vision_backbone is None:
        raise ValueError(f"Unable to infer brain/vision backbones from {exp_dir} and {config}")
    return brain_backbone, vision_backbone


def find_best_ckpt(checkpoint_dir: str) -> str:
    """Prioritize *best*, otherwise use *last*."""
    base_dir = Path(checkpoint_dir)
    best = glob.glob(os.path.join(checkpoint_dir, "*best*.ckpt"))
    if best:
        return best[0]
    last = glob.glob(os.path.join(checkpoint_dir, "*last.ckpt"))
    if last:
        return last[0]
    if base_dir.parent.exists():
        best = sorted(str(path) for path in base_dir.parent.rglob("*best*.ckpt"))
        if best:
            return best[-1]
        last = sorted(str(path) for path in base_dir.parent.rglob("*last.ckpt"))
        if last:
            return last[-1]
    raise FileNotFoundError(f"No best or last checkpoint in {checkpoint_dir}")


def find_config_yaml(exp_dir: str, exclude_pattern: str = "hparams", preferred_dir: str | Path | None = None) -> Path:
    """Find the unique runtime config file stored next to an experiment directory."""
    exp_dir_path = Path(exp_dir)
    if not exp_dir_path.is_dir():
        raise FileNotFoundError(f"{exp_dir_path} does not exist or is not a directory")

    candidates = [path for path in exp_dir_path.rglob("*.yaml") if not re.search(exclude_pattern, path.name, re.I)]
    if not candidates:
        raise FileNotFoundError(f"No available yaml under {exp_dir_path} (excluded *{exclude_pattern}*)")
    preferred_path = Path(preferred_dir).resolve() if preferred_dir is not None else None

    def rank(path: Path):
        version = -1
        for part in path.parts:
            match = re.fullmatch(r"version_(\d+)", part)
            if match:
                version = max(version, int(match.group(1)))
        preferred = 1 if preferred_path is not None and path.parent.resolve() == preferred_path else 0
        preferred_name = 1 if path.name == "zkf_final.yaml" else 0
        return (preferred, version, preferred_name, len(path.parts))

    return sorted(candidates, key=rank)[-1]


def _load_or_build_precomputed_dataset(
    *,
    save_path: str,
    source_loader,
    pl_model,
    device: str,
    embedding_path: str,
    batch_size: int,
    shuffle: bool,
):
    if os.path.exists(save_path):
        payload = torch.load(save_path, map_location="cpu")
        worker_count = os.cpu_count() or 0
        return DataLoader(
            payload["dataset"],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=worker_count,
            pin_memory=True,
            persistent_workers=worker_count > 0,
        )

    img_paths = [path for batch in source_loader for path in batch["img_path"]]
    h_embeddings = load_diffusion_embeddings(embedding_path, img_paths)
    dataloader, _, _ = prepare_embedding_dataset(
        pl_model=pl_model,
        train_loader=source_loader,
        device=device,
        save_path=save_path,
        h_embeddings=h_embeddings,
    )
    return dataloader


def _resolve_align_embedding_paths(config) -> tuple[str, str]:
    embedding_root = resolve_diffusion_embeddings_root(config)
    train_embedding_path = embedding_root / "diffusion_clip_embeddings_train.pt"
    val_embedding_path = embedding_root / "diffusion_clip_embeddings_test.pt"
    if not train_embedding_path.exists() or not val_embedding_path.exists():
        raise FileNotFoundError(
            "Missing diffusion embedding files. Expected "
            f"{train_embedding_path} and {val_embedding_path}."
        )
    return str(train_embedding_path), str(val_embedding_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_version", type=str, default="ViT-H-14")
    parser.add_argument("--device", type=int, default=0, help="CUDA device number, e.g. 0 or 1")
    parser.add_argument(
        "--model_type",
        type=str,
        default="diffusion",
        choices=["diffusion", "simple"],
        help="Select model type to use: diffusion (DiffusionPipe) or simple (SimpleAlignPipe)",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Output file suffix, e.g. '_original' will produce diffusion_model_best_original.pth and generated_embeddings_original.pt",
    )
    opt = parser.parse_args()

    exp_setting, subject, run_seed = parse_exp_dir(opt.exp_dir)
    checkpoint_hint = None
    try:
        checkpoint_hint = find_best_ckpt(os.path.join(opt.exp_dir, "checkpoints"))
    except FileNotFoundError:
        checkpoint_hint = None

    preferred_config_dir = Path(checkpoint_hint).parent.parent if checkpoint_hint is not None else None
    cfg_path = find_config_yaml(opt.exp_dir, exclude_pattern="hparams", preferred_dir=preferred_config_dir)
    raw_config = OmegaConf.load(cfg_path)
    raw_container = OmegaConf.to_container(raw_config, resolve=False)
    assert isinstance(raw_container, dict)
    brain_backbone, vision_backbone = parse_backbones(raw_container, opt.exp_dir)
    device = f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu"
    training_epoch = int(_raw_nested_get(raw_container, "train", "epoch", default=80))
    staged_training = bool(_raw_nested_get(raw_container, "train", "staged_training", default=False))

    pl_model, config, train_loader, val_loader, test_loader = load_eeg_pl_model(
        exp_dir=opt.exp_dir,
        subject=subject,
        seed=run_seed,
        devices=[opt.device],
        lr=opt.lr,
        exp_setting=exp_setting,
        brain_backbone=brain_backbone,
        vision_backbone=vision_backbone,
        epoch=training_epoch,
        staged_training=staged_training,
        yaml_path=str(cfg_path),
    )

    evaluate_eeg_accuracy(pl_model=pl_model, test_loader=test_loader, device=device)

    align_config = {
        "num_epochs": opt.epoch,
        "learning_rate": opt.lr,
        "save_path": os.path.join(opt.exp_dir, "diffusion_ckpt"),
        "early_stop_patience": 20,
    }

    dataset_root = os.path.join(opt.exp_dir, "precomputed_datasets")
    os.makedirs(dataset_root, exist_ok=True)
    train_dataset_path = os.path.join(dataset_root, f"{subject}_train_dataset.pt")
    val_dataset_path = os.path.join(dataset_root, f"{subject}_val_dataset.pt")

    train_embedding_path, val_embedding_path = _resolve_align_embedding_paths(config)
    diffusion_train_loader = _load_or_build_precomputed_dataset(
        save_path=train_dataset_path,
        source_loader=train_loader,
        pl_model=pl_model,
        device=device,
        embedding_path=train_embedding_path,
        batch_size=10240,
        shuffle=True,
    )
    diffusion_val_loader = _load_or_build_precomputed_dataset(
        save_path=val_dataset_path,
        source_loader=val_loader,
        pl_model=pl_model,
        device=device,
        embedding_path=val_embedding_path,
        batch_size=200,
        shuffle=False,
    )

    eeg_encoder = pl_model.brain.to(device)
    eeg_encoder.eval()

    if opt.model_type == "diffusion":
        align_pipe = init_diffusion_prior(
            device=device,
            cond_dim=config.z_dim,
            modalities=["image", "depth", "edge", "fusion"],
            output_suffix=opt.output_suffix,
        )
    else:
        align_pipe = init_simple_align_prior(
            device=device,
            cond_dim=config.z_dim,
            modalities=["image", "depth", "edge", "fusion"],
            separate_optimizers=True,
        )

    diffusion_ckpt_dir = os.path.join(opt.exp_dir, "diffusion_ckpt")
    os.makedirs(diffusion_ckpt_dir, exist_ok=True)
    if opt.model_type == "diffusion":
        best_ckpt = glob.glob(os.path.join(diffusion_ckpt_dir, f"*best{opt.output_suffix}*.pth"))
    else:
        best_ckpt = glob.glob(os.path.join(diffusion_ckpt_dir, "*best*.pth"))

    latest_ckpt = best_ckpt[0] if best_ckpt else None
    if latest_ckpt and os.path.exists(latest_ckpt):
        align_pipe.load_ckpt(latest_ckpt)
    else:
        train_config = align_config.copy()
        train_config["save_path"] = diffusion_ckpt_dir
        align_pipe.train(
            train_dataloader=diffusion_train_loader,
            val_dataloader=diffusion_val_loader,
            config=train_config,
        )

    img_paths = [path for batch in test_loader for path in batch["img_path"]]
    save_dir = os.path.join(opt.exp_dir, "generated_datasets")
    generate_dataset_with_diffusion(
        diffusion_pipe=align_pipe,
        pl_model=pl_model,
        data_loader=test_loader,
        img_paths=img_paths,
        device=device,
        save_dir=save_dir,
        num_inference_steps=50,
        guidance_scale=5.0,
        output_suffix=opt.output_suffix,
    )


if __name__ == "__main__":
    main()
