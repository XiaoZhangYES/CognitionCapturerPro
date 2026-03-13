import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from diffusers.schedulers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
import os
from typing import Optional, Tuple, Dict, Any, Union
from torch import nn
from torch.utils.data import Dataset, DataLoader
from diffusers.models.embeddings import Timesteps, TimestepEmbedding


class DiffusionPriorUNet(nn.Module):

    def __init__(
            self,
            embed_dim: int = 1024,
            cond_dim: int = 42,
            hidden_dim: list[int] = [1024, 512, 256, 128, 64],
            time_embed_dim: int = 512,
            act_fn: nn.Module = nn.SiLU,
            dropout: float = 0.0,
    ):
        super().__init__()
        assert len(hidden_dim) >= 2, "hidden_dim must contain at least 2 elements"
        assert all(isinstance(dim, int) and dim > 0 for dim in hidden_dim), "hidden_dim elements must be positive integers"

        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)

        self.time_proj = Timesteps(time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.act_fn = act_fn()

        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim[0]),
            nn.LayerNorm(hidden_dim[0]),
            self.act_fn
        )

        self.encode_blocks = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.encode_blocks.append(nn.ModuleDict({
                "time_emb": TimestepEmbedding(time_embed_dim, hidden_dim[i]),
                "cond_emb": nn.Linear(cond_dim, hidden_dim[i]),
                "downsample": nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i + 1]),
                    nn.LayerNorm(hidden_dim[i + 1]),
                    self.act_fn,
                    nn.Dropout(dropout)
                )
            }))

        self.decode_blocks = nn.ModuleList()
        for i in range(self.num_layers - 1, 0, -1):
            self.decode_blocks.append(nn.ModuleDict({
                "time_emb": TimestepEmbedding(time_embed_dim, hidden_dim[i]),
                "cond_emb": nn.Linear(cond_dim, hidden_dim[i]),
                "upsample": nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i - 1]),
                    nn.LayerNorm(hidden_dim[i - 1]),
                    self.act_fn,
                    nn.Dropout(dropout)
                )
            }))

        self.output_layer = nn.Linear(hidden_dim[0], embed_dim)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[1] == self.embed_dim, f"x shape should be (batch, {self.embed_dim})"
        assert t.ndim == 1, f"t shape should be (batch, )"
        if c is not None:
            assert c.ndim == 2 and c.shape[1] == self.cond_dim, f"c shape should be (batch, {self.cond_dim})"

        t_emb = self.time_proj(t)
        x = self.input_layer(x)

        encoder_feats = []
        for i, block in enumerate(self.encode_blocks):
            encoder_feats.append(x)
            t_emb_i = block["time_emb"](t_emb)
            c_emb_i = block["cond_emb"](c) if c is not None else 0
            x = x + t_emb_i + c_emb_i
            x = block["downsample"](x)

        encoder_feats_rev = encoder_feats[::-1]
        for i, block in enumerate(self.decode_blocks):
            t_emb_i = block["time_emb"](t_emb)
            c_emb_i = block["cond_emb"](c) if c is not None else 0
            x = x + t_emb_i + c_emb_i
            x = block["upsample"](x)
            x += encoder_feats_rev[i]

        return self.output_layer(x)


class MultiModalDiffusionPrior(nn.Module):
    """Multi-modal diffusion prior model (maintains independent UNet for each modality)"""

    def __init__(
            self,
            modalities: list[str] = ['image', 'text', 'depth', 'edge'],
            embed_dim: int = 1024,
            cond_dim: int = 1024,
            hidden_dim: list[int] = [1024, 512, 256, 128],
            dropout: float = 0.1
    ):
        super().__init__()
        self.modalities = modalities
        self.embed_dim = embed_dim

        # Create independent UNet for each modality
        self.priors = nn.ModuleDict({
            mod: DiffusionPriorUNet(
                embed_dim=embed_dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            ) for mod in modalities
        })

    def forward(
            self,
            x: Dict[str, torch.Tensor],  # {modality: noisy embeddings}
            t: torch.Tensor,  # timesteps (shared across all modalities)
            c: Dict[str, torch.Tensor]  # {modality: conditional embeddings}
    ) -> Dict[str, torch.Tensor]:  # {modality: predicted noise}
        outputs = {}
        for mod in x.keys():
            # Ensure current modality key exists in c, pass None if not exists
            current_c = c[mod] if mod in c else None
            outputs[mod] = self.priors[mod](x[mod], t, current_c)
        return outputs


class SimpleAlignMLP(nn.Module):
    """Accept all modalities at once, output all modalities"""

    def __init__(self,
                 in_dim=1024,          # single modality dimension
                 modalities=['image', 'depth', 'edge'],         # number of modalities
                 hidden=1024,
                 dropout=0.1,
                 num_blocks=4):
        super().__init__()
        self.modalities = modalities
        total_in = in_dim * len(modalities)          # total dimension after concatenation

        # 1. Shared trunk
        trunk = [nn.Sequential(
            nn.Linear(total_in, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout)
        )]
        for _ in range(num_blocks):
            trunk.append(nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Dropout(dropout)
            ))
        self.trunk = nn.Sequential(*trunk)

        # 2. Independent head for each modality
        self.heads = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Linear(hidden, in_dim)   # restore output dimension
            ) for mod in modalities
        })

    def forward(self, c: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        c: dict[mod: [B, D]]
        return: dict[mod: [B, D]]  (L2-normalized)
        """
        # 1. Concatenate in fixed order → [B, M*D]
        keys = sorted(c.keys())          # ensure consistent order
        x = torch.cat([c[k] for k in keys], dim=1)

        # 2. Shared trunk
        feat = self.trunk(x)

        # 3. Modality heads
        out = {k: self.heads[k](feat) for k in keys}

        # 4. L2 normalization
        return {k: F.normalize(v, p=2, dim=1) for k, v in out.items()}


class SimpleAlignNet(nn.Module):
    def __init__(self, modalities, cond_dim=1024, out_dim=1024, dropout=0.1):
        super().__init__()
        self.modalities = modalities
        self.mlp = SimpleAlignMLP(
            in_dim=cond_dim,
            modalities=modalities,
            hidden=cond_dim,
            dropout=dropout
        )

    def forward(self, c: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.mlp(c)          # return all modalities at once


class EmbeddingDataset(Dataset):

    def __init__(
            self,
            c_embeddings: Dict[str, Union[torch.Tensor, np.ndarray]],  # key: modality name, value: conditional embedding (c is eeg output)
            h_embeddings: Dict[str, Union[torch.Tensor, np.ndarray]]  # key: modality name, value: target embedding
    ):
        # Validate modality consistency
        assert set(c_embeddings.keys()) == set(h_embeddings.keys()), "Modalities of conditional and target embeddings must be consistent"
        self.modalities = list(c_embeddings.keys())  # modality order: image, text, depth, edge

        # Data type conversion and validation
        self.c_embeddings = {}
        self.h_embeddings = {}
        for mod in self.modalities:
            # Convert conditional embeddings (maintain original logic)
            if isinstance(c_embeddings[mod], np.ndarray):
                self.c_embeddings[mod] = torch.from_numpy(c_embeddings[mod]).float()
            else:
                self.c_embeddings[mod] = c_embeddings[mod].float()

            # Convert target embeddings (added dimension compression)
            if isinstance(h_embeddings[mod], np.ndarray):
                h_tensor = torch.from_numpy(h_embeddings[mod]).float()
            else:
                h_tensor = h_embeddings[mod].float()

            if h_tensor.ndim == 3 and h_tensor.shape[1] == 1:
                h_tensor = h_tensor.squeeze(1)  # after squeeze: (num_samples, embed_dim)

            self.h_embeddings[mod] = h_tensor

            # Length and dimension validation
            assert len(self.c_embeddings[mod]) == len(self.h_embeddings[mod]), \
                f"{mod} modality conditional and target embedding length mismatch"
            assert self.c_embeddings[mod].ndim == 2, f"{mod} conditional embedding must be 2D tensor (num_samples, cond_dim)"
            assert self.h_embeddings[
                       mod].ndim == 2, f"{mod} target embedding must be 2D tensor (num_samples, embed_dim), currently {self.h_embeddings[mod].ndim}D"

    def __len__(self) -> int:
        return len(self.c_embeddings[self.modalities[0]])  # all modalities have same number of samples

    def __getitem__(self, idx: int):
        """Return embeddings of all modalities for given index (in dict format)"""
        return {
            "c_embedding": {mod: self.c_embeddings[mod][idx] for mod in self.modalities},
            "h_embedding": {mod: self.h_embeddings[mod][idx] for mod in self.modalities}
        }

    def __repr__(self) -> str:
        return (f"EmbeddingDataset(\n"
                f"  num_samples={len(self)},\n"
                f"  modalities={self.modalities},\n"
                f"  c_embedding_shapes={{{', '.join(f'{k}: {v.shape}' for k, v in self.c_embeddings.items())}}},\n"
                f"  h_embedding_shapes={{{', '.join(f'{k}: {v.shape}' for k, v in self.h_embeddings.items())}}}\n)")


class CosineLoss(nn.Module):
    """1 - cos(x, y), normalize on dim=1 by default, return scalar"""

    def __init__(self, reduction='mean', eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        """
        pred/target: [B, D] shapes can be any same
        """
        # L2 normalization
        pred_norm = F.normalize(pred, p=2, dim=1, eps=self.eps)
        target_norm = F.normalize(target, p=2, dim=1, eps=self.eps)
        cos_sim = (pred_norm * target_norm).sum(dim=1)  # [B]
        loss = 1.0 - cos_sim  # [B]
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class SDEmbeddingLoss(nn.Module):
    """
    Input: u, v both (B, 1024)
    1. MSE
    2. Cosine (direction)
    3. L2 regularization
    Usage:
        criterion = SDEmbeddingLoss1D()
        loss = criterion(u, v)
    """

    def __init__(self,
                 w_mse: float = 1.0,
                 w_cos: float = 0.5,
                 w_reg: float = 1e-4):
        super().__init__()
        self.w_mse = w_mse
        self.w_cos = w_cos
        self.w_reg = w_reg

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Model output  (B, 1024)
            v: Target ground truth  (B, 1024)
        Returns:
            Scalar loss
        """
        # 1. MSE
        loss_mse = F.mse_loss(u, v)

        # 2. Cosine (feature level)
        u_norm = F.normalize(u, p=2, dim=-1)
        v_norm = F.normalize(v, p=2, dim=-1)
        loss_cos = (1 - F.cosine_similarity(u_norm, v_norm, dim=-1)).mean()

        # 3. L2 regularization
        loss_reg = u.pow(2).mean()

        loss = (self.w_mse * loss_mse +
                self.w_cos * loss_cos +
                self.w_reg * loss_reg)
        return loss


class SimpleAlignPipe:

    def __init__(self,
                 align_net: Optional[SimpleAlignNet] = None,
                 scheduler=None,
                 device: str = 'cuda',
                 modalities: Optional[list[str]] = None,
                 lr: float = 3e-4,
                 separate_optimizers=None # reserved argument
                 ):
        self.net = align_net.to(device)
        self.device = device
        self.modalities = modalities or ['image', 'depth', 'edge']
        self.criterion = SDEmbeddingLoss()
        self.epoch = 0
        self.best_avg_cos_sim = -1.0
        self.lr = lr

        # Only one optimizer
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, fused=True)
        self.scheduler = None   # will be instantiated in train

    def _random_mask_modalities(self) -> list[str]:
        """
        Always randomly mask 1 modality (return empty if modality list is empty).
        Return list of length 1, containing the masked modality name.
        """
        if not self.modalities:  # fallback
            return []
        # randomly select 1
        idx = torch.randint(len(self.modalities), (1,)).item()
        return [self.modalities[idx]]

    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              config: Dict[str, Any]):
        num_epochs = config.get('num_epochs', 10)
        lr = config.get('learning_rate', self.lr)
        save_path = config.get('save_path', './checkpoints')
        patience = config.get('early_stop_patience', 50)
        os.makedirs(save_path, exist_ok=True)

        steps_per_epoch = len(train_dataloader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr * 8, 
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.15, 
            div_factor=10.0, 
            final_div_factor=1e4 
        )

        best_cos, early_stop_counter = -1.0, 0
        p_mask = config.get("mask_prob", 0.15)

        for epoch in range(num_epochs):
            if early_stop_counter >= patience:
                print(f"Early stopping: no improvement for {patience} consecutive epochs")
                break

            # ==================== Training ====================
            self.net.train()
            total_loss = 0.

            for batch in train_dataloader:
                c = {k: v.to(self.device) for k, v in batch["c_embedding"].items()}
                h = {k: v.to(self.device) for k, v in batch["h_embedding"].items()}

                # masked_mods = self._random_mask_modalities()
                # for mod in masked_mods:
                #     # Directly set to a zero vector; gradients can still propagate
                #     c[mod] = torch.zeros_like(c[mod])

                self.optimizer.zero_grad()
                pred = self.net(c)              # Get predictions for all modalities at once
                loss = sum(self.criterion(pred[m], h[m]) for m in self.modalities)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()

            # ==================== Validation ====================
            self.net.eval()
            val_loss_sum, total_samples = 0., 0
            val_cos = {m: 0. for m in self.modalities}

            with torch.no_grad():
                for batch in val_dataloader:
                    c = {k: v.to(self.device) for k, v in batch["c_embedding"].items()}
                    h = {k: v.to(self.device) for k, v in batch["h_embedding"].items()}
                    pred = self.net(c)          # All modalities
                    bs = h[self.modalities[0]].shape[0]
                    total_samples += bs

                    loss = sum(self.criterion(pred[m], h[m]) for m in self.modalities)
                    val_loss_sum += loss.item() * bs
                    for m in self.modalities:
                        val_cos[m] += F.cosine_similarity(pred[m], h[m], dim=1).sum().item()

            avg_val_loss = val_loss_sum / total_samples
            val_cos = {m: val_cos[m] / total_samples for m in self.modalities}
            avg_cos = sum(val_cos.values()) / len(val_cos)

            # ==================== Logging & Saving ====================
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"train-loss {total_loss / len(train_dataloader):.6f} | "
                  f"val-loss {avg_val_loss:.6f} | "
                  f"val-cos {avg_cos:.6f}")
            if avg_cos > best_cos:
                best_cos = avg_cos
                early_stop_counter = 0
                self.best_avg_cos_sim = best_cos
                self.epoch = epoch + 1
                self.save_ckpt(os.path.join(save_path, "diffusion_model_best.pth"))
                print(f"  Best model saved, average cosine similarity {best_cos:.6f}")
            else:
                early_stop_counter += 1

    def save_ckpt(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"align_net": self.net.state_dict(),
                    "epoch": self.epoch,
                    "best_avg_cos_sim": self.best_avg_cos_sim}, path)

    def load_ckpt(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["align_net"], strict=True)
        self.epoch = ckpt.get("epoch", 0)
        self.best_avg_cos_sim = ckpt.get("best_avg_cos_sim", -1.0)
        print(f"Loaded alignment model: {path} (epoch {self.epoch}, best-cos {self.best_avg_cos_sim:.6f})")
        self.net.eval()

    @torch.no_grad()
    def generate(self,
                 condition_embeds: Optional[dict[str, torch.Tensor]] = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 5.0,
                 generator=None) -> dict[str, torch.Tensor]:
        self.net.eval()
        if condition_embeds is None:
            raise ValueError("SimpleAlignPipe must provide condition_embeds")
        c = {k: v.to(self.device) for k, v in condition_embeds.items()}
        return self.net(c)          # directly return all modalities


class DiffusionPipe:
    """Pipeline class supporting multi-modal simultaneous training"""

    def __init__(self,
                 diffusion_prior: nn.Module,
                 scheduler: Optional[Any] = None,
                 device: str = 'cuda',
                 modalities: Optional[list[str]] = None,
                 separate_optimizers: bool = False,
                 output_suffix: str = ""
                 ):
        supported_modalities = ['image', 'text', 'depth', 'edge', 'fusion']
        if modalities is None:
            modalities = supported_modalities
        else:
            for mod in modalities:
                if mod not in supported_modalities:
                    raise ValueError(f"Modality {mod} not supported, must be one of {supported_modalities}")

        self.diffusion_prior = diffusion_prior.to(device)
        self.scheduler = scheduler or DDPMScheduler()
        self.modalities = modalities  # All modalities to train
        self.device = device
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        # For multi optimizers
        self.separate_optimizers = separate_optimizers  # new switch
        self.modal_optimizers = None  # independent optimizer container
        self.modal_lr_schs = None  # independent scheduler container
        self.output_suffix = output_suffix  # output file suffix

    def _prepare_optimizers(self,
                            learning_rate: float,
                            num_training_steps: int):
        """
        Based on separate_optimizers switch, return:
          1) Joint mode -> (optimizer, scheduler) 
          2) Separate mode -> (dict[mod: optimizer], dict[mod: scheduler])
        """
        if not self.separate_optimizers:
            # --- Original logic: one optimizer for all modalities ---
            optimizer = optim.AdamW(self.diffusion_prior.parameters(), lr=learning_rate, fused=True)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=500,
                num_training_steps=num_training_steps)
            return optimizer, scheduler

        # --- New logic: separate Adam + scheduler for each modality ---
        if self.modal_optimizers is None:  # first initialization
            self.modal_optimizers = {}
            self.modal_lr_schs = {}
            for mod in self.modalities:
                # Only collect current modality UNet parameters
                mod_params = self.diffusion_prior.priors[mod].parameters()
                opt = optim.AdamW(mod_params, lr=learning_rate, fused=True)
                sch = get_cosine_schedule_with_warmup(
                    opt, num_warmup_steps=100,
                    num_training_steps=num_training_steps)
                self.modal_optimizers[mod] = opt  # wrap with ModuleDict, auto .cuda/cpu
                self.modal_lr_schs[mod] = sch
        return self.modal_optimizers, self.modal_lr_schs

    def _train_step(self,
                    batch: Dict[str, Any],
                    criterion: nn.Module) -> float:
        """Multi-modal training step"""
        # 1. Get conditional and target embeddings for all modalities from dataset
        condition_embeds = {mod: (batch["c_embedding"][mod]).to(self.device).float()
                            for mod in self.modalities}
        hidden_embeds = {mod: (batch["h_embedding"][mod]).to(self.device).float()
                         for mod in self.modalities}

        # 10% probability to clear conditional embeddings (classifier-free guidance)
        if torch.rand(1) < 0.1:
            condition_embeds = {mod: None for mod in self.modalities}

        # 2. Diffusion process: add noise → predict noise → compute loss
        batch_size = next(iter(hidden_embeds.values())).shape[0]
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=self.device)

        # Prepare noisy embeddings for each modality
        noisy_embeds = {}
        noise = {}
        for mod in self.modalities:
            noise[mod] = torch.randn_like(hidden_embeds[mod])
            noisy_embeds[mod] = self.scheduler.add_noise(hidden_embeds[mod], noise[mod], timesteps)

        # Predict noise and compute loss
        noise_pred = self.diffusion_prior(noisy_embeds, timesteps, condition_embeds)

        # 5. Compute loss (per modality)
        losses = {mod: criterion(noise_pred[mod], noise[mod]).mean()
                  for mod in self.modalities}

        # 6. Backpropagation + parameter update
        if self.separate_optimizers:
            # --- Independent optimizer mode ---
            for mod in self.modalities:
                self.modal_optimizers[mod].zero_grad()
                losses[mod].backward(retain_graph=False)  # last graph doesn't need retain
                self.modal_optimizers[mod].step()
                self.modal_lr_schs[mod].step()
                torch.nn.utils.clip_grad_norm_(self.diffusion_prior.priors[mod].parameters(), 5.0)
            total_loss = sum(losses.values()) / len(self.modalities)  # only for logging
        else:
            # --- Joint optimizer mode (original logic) ---
            total_loss = sum(losses.values()) / len(self.modalities)
            # optimizer.zero_grad / backward / step completed in outer caller
            return total_loss

        return total_loss.item()

    def _evaluate_step(self,
                       batch: Dict[str, Any],
                       criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """Multi-modal evaluation step"""
        with torch.no_grad():
            # 1. Get embeddings from dataset
            condition_embeds = {mod: (batch["c_embedding"][mod]).to(self.device).float()
                                for mod in self.modalities}
            hidden_embeds = {mod: (batch["h_embedding"][mod]).to(self.device).float()
                             for mod in self.modalities}

            # 2. Generate features for all modalities at once (no loop needed)
            generated_embeds = self.generate(
                condition_embeds=condition_embeds,  # pass multi-modal dict
                num_inference_steps=50,
                guidance_scale=5.0
            )

            # Compute cosine similarity for each modality
            cos_sim = {}
            for mod in self.modalities:
                cos_sim[mod] = F.cosine_similarity(
                    generated_embeds[mod],
                    hidden_embeds[mod],
                    dim=1
                ).mean().item()

            # 3. Compute evaluation loss (keep unchanged)
            batch_size = next(iter(hidden_embeds.values())).shape[0]
            timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=self.device)

            noisy_embeds = {}
            noise = {}
            for mod in self.modalities:
                noise[mod] = torch.randn_like(hidden_embeds[mod])
                noisy_embeds[mod] = self.scheduler.add_noise(hidden_embeds[mod], noise[mod], timesteps)

            noise_pred = self.diffusion_prior(noisy_embeds, timesteps, condition_embeds)

            total_loss = 0.0
            for mod in self.modalities:
                total_loss += criterion(noise_pred[mod], noise[mod]).mean().item()

            avg_loss = total_loss / len(self.modalities)
            return avg_loss, cos_sim

    def train(self,
              train_dataloader: torch.utils.data.DataLoader,
              val_dataloader: torch.utils.data.DataLoader,
              config: Dict[str, Any]) -> None:
        """Multi-modal training logic"""
        num_epochs = config.get('num_epochs', 150)
        learning_rate = config.get('learning_rate', 3e-3)
        save_path = config.get('save_path')
        early_stop_patience = config.get('early_stop_patience', 50)

        os.makedirs(save_path, exist_ok=True)
        criterion = nn.MSELoss(reduction='mean')
        num_training_steps = len(train_dataloader) * num_epochs

        # 1. Construct optimizer
        optim_ret = self._prepare_optimizers(learning_rate, num_training_steps)
        if self.separate_optimizers:
            # Separate mode: no top-level optimizer needed
            optimizer, lr_scheduler = None, None
        else:
            # Joint mode: keep old variable names
            optimizer, lr_scheduler = optim_ret

        # Track best cosine similarity for each modality
        best_cos_sim = {mod: -1.0 for mod in self.modalities}
        early_stop_counter = 0
        best_avg_cos_sim = -1.0

        for epoch in range(num_epochs):
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered: no improvement for {early_stop_patience} consecutive epochs, terminating training")
                break

            # Training phase
            self.diffusion_prior.train()
            total_train_loss = 0.0

            for batch in train_dataloader:
                if not self.separate_optimizers:
                    # Joint mode: zero_grad / step in outer layer
                    optimizer.zero_grad()
                loss = self._train_step(batch, criterion)
                total_train_loss += loss.item() if hasattr(loss, "item") else loss

                if not self.separate_optimizers:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 5.0)
                    optimizer.step()
                    lr_scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)

            # Validation phase
            self.diffusion_prior.eval()
            total_val_loss = 0.0
            total_cos_sim = {mod: 0.0 for mod in self.modalities}

            for batch in val_dataloader:
                val_loss, cos_sim = self._evaluate_step(batch, criterion)
                total_val_loss += val_loss
                for mod in self.modalities:
                    total_cos_sim[mod] += cos_sim[mod]

            avg_val_loss = total_val_loss / len(val_dataloader)
            avg_cos_sim = {mod: total_cos_sim[mod] / len(val_dataloader) for mod in self.modalities}
            current_avg_cos_sim = sum(avg_cos_sim.values()) / len(self.modalities)

            # Print logs
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train loss: {avg_train_loss:.6f} | Validation loss: {avg_val_loss:.6f}")
            for mod in self.modalities:
                print(f"  Validation cosine similarity ({mod}): {avg_cos_sim[mod]:.6f}")

            # Save best model (based on average similarity across all modalities)
            if current_avg_cos_sim > best_avg_cos_sim:
                best_avg_cos_sim = current_avg_cos_sim
                for mod in self.modalities:
                    if avg_cos_sim[mod] > best_cos_sim[mod]:
                        best_cos_sim[mod] = avg_cos_sim[mod]
                early_stop_counter = 0
                model_path = os.path.join(save_path, f"diffusion_model_best{self.output_suffix}.pth")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)

                # Save complete checkpoint (including model weights and training state)
                torch.save({
                    "diffusion_prior": self.diffusion_prior.state_dict(),
                    "epoch": epoch + 1,
                    "best_avg_cos_sim": best_avg_cos_sim,
                    "best_cos_sim": best_cos_sim
                }, model_path)
                print(f"  Model updated: average cosine similarity improved to {best_avg_cos_sim:.6f}, saved to {model_path}")
            else:
                early_stop_counter += 1

    def generate(self,
                 condition_embeds: Optional[torch.Tensor] = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 5.0,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Generate features for specific modality"""
        self.diffusion_prior.eval()
        # Get batch size from conditional embeddings (all modalities have same batch size)
        if condition_embeds is not None:
            batch_size = next(iter(condition_embeds.values())).shape[0]
        else:
            batch_size = 1

        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            self.device
        )

        # Prepare multi-modal conditional embedding dict (ensure all modalities in dict, None means unconditional)
        condition_embeds_dict = {}
        for mod in self.modalities:
            if condition_embeds is not None and mod in condition_embeds:
                condition_embeds_dict[mod] = condition_embeds[mod].to(self.device)
            else:
                condition_embeds_dict[mod] = None

        # Initialize noisy embeddings for each modality
        hidden_dim = self.diffusion_prior.embed_dim
        noisy_embeds = {
            mod: torch.randn(
                batch_size,
                hidden_dim,
                generator=generator,
                device=self.device
            ) for mod in self.modalities
        }

        # Diffusion process (update all modalities simultaneously)
        for t in timesteps:
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.float32)

            if guidance_scale > 0 and condition_embeds is not None:
                # Conditional prediction (all modalities input simultaneously)
                noise_pred_cond = self.diffusion_prior(
                    noisy_embeds,  # multi-modal noisy embeddings
                    t_tensor,
                    condition_embeds_dict  # multi-modal conditional embeddings
                )

                # Unconditional prediction (all modalities pass None)
                noise_pred_uncond = self.diffusion_prior(
                    noisy_embeds,
                    t_tensor,
                    {mod: None for mod in self.modalities}  # all modalities unconditional
                )

                # Combine conditional and unconditional predictions (process per modality)
                noise_pred = {
                    mod: noise_pred_uncond[mod] + guidance_scale * (noise_pred_cond[mod] - noise_pred_uncond[mod])
                    for mod in self.modalities
                }
            else:
                # Direct prediction without guidance
                noise_pred = self.diffusion_prior(
                    noisy_embeds,
                    t_tensor,
                    condition_embeds_dict
                )

            # Update noisy embeddings per modality
            for mod in self.modalities:
                noisy_embeds[mod] = self.scheduler.step(
                    noise_pred[mod],
                    t,
                    noisy_embeds[mod],
                    generator=generator
                ).prev_sample

        return noisy_embeds  # return generation results for all modalities

    def load_ckpt(self, ckpt_path: str) -> None:
        """
        Load diffusion model checkpoint

        Args:
            ckpt_path: Checkpoint file path (.pth format)
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # Handle different checkpoint formats (support model weights only or with additional info)
        if isinstance(checkpoint, dict) and "diffusion_prior" in checkpoint:
            # Checkpoint with complete info (may include training state)
            self.diffusion_prior.load_state_dict(checkpoint["diffusion_prior"], strict=True)
            self.epoch = checkpoint.get("epoch", 0)
            self.best_avg_cos_sim = checkpoint.get("best_avg_cos_sim", -1.0)
            print(f"Loaded complete checkpoint: {ckpt_path} (epoch: {self.epoch}, best average cosine similarity: {self.best_avg_cos_sim:.6f})")
        else:
            # Checkpoint with model weights only
            self.diffusion_prior.load_state_dict(checkpoint, strict=True)
            print(f"Loaded model weights: {ckpt_path}")

        # Ensure model is in evaluation mode
        self.diffusion_prior.eval()
