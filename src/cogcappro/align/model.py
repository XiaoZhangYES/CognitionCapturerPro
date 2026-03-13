"""Model helpers for the refactored align runtime."""

from __future__ import annotations

import os

import torch

from ..models.brain_backbone import ProjMod_multimodal
from ..models.fusion_backbone import CogcapFusion
from ..utils import ClipLoss_Modified_DDP
from .diffusion_pipe import (
    DiffusionPipe,
    MultiModalDiffusionPrior,
    SimpleAlignNet,
    SimpleAlignPipe,
)
from ..training.module import load_model


def init_fusion_models():
    """Initialize fusion modules kept by the original checkpoint layout."""
    fusion_mod = CogcapFusion(
        modal_dims=[1024, 1024, 1024, 1024],
        hidden_dim=255,
        num_heads=1,
        dropout=0.1,
    )

    fusion_eeg = CogcapFusion(
        modal_dims=[1024, 1024, 1024, 1024],
        hidden_dim=255,
        num_heads=1,
        dropout=0.1,
    )

    mod_proj = ProjMod_multimodal()
    return fusion_mod, fusion_eeg, mod_proj


def load_pl_model_from_checkpoint(config, train_loader, test_loader, checkpoint_path, device):
    """Load the EEG LightningModule while preserving state-dict compatibility."""
    pl_model = load_model(config, train_loader, test_loader).to(device)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        pl_model.load_state_dict(state_dict, strict=True)
        print("Checkpoint loaded successfully")
    else:
        if checkpoint_path is not None:
            print(f"Warning: checkpoint path does not exist - {checkpoint_path}, using initialized model")
        else:
            print("No checkpoint specified, using initialized model")

    pl_model.criterion = ClipLoss_Modified_DDP(top_k=10, cos_batch=512)
    return pl_model


def init_diffusion_prior(device, cond_dim=1024, modalities=None, output_suffix=""):
    """Initialize the multimodal diffusion prior and training pipe."""
    modalities = modalities or ["image", "text", "depth", "edge"]
    unet = MultiModalDiffusionPrior(
        modalities=modalities,
        embed_dim=1024,
        cond_dim=cond_dim,
        dropout=0.1,
    ).to(device)
    return DiffusionPipe(
        unet,
        device=device,
        modalities=modalities,
        separate_optimizers=True,
        output_suffix=output_suffix,
    )


def init_simple_align_prior(device, cond_dim=1024, modalities=None, **kwargs):
    """Initialize the simple alignment baseline."""
    modalities = modalities or ["image", "text", "depth", "edge"]
    net = SimpleAlignNet(modalities=modalities, cond_dim=cond_dim, out_dim=1024)
    return SimpleAlignPipe(align_net=net, device=device, modalities=modalities, **kwargs)
