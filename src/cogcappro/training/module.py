import torch
import pytorch_lightning as pl
import numpy as np
from collections import Counter
from scipy.stats import norm
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD

from ..data.eeg import load_data as load_eeg_data
from ..data import meg as meg_module
from ..models.brain_backbone import ProjMod_multimodal
from ..models.fusion_backbone import CogcapFusion
from ..utils import ClipLoss_Modified_DDP, ClipLoss_Original, instantiate_from_config

__all__ = ["PLModel", "load_data", "load_model"]


def load_data(config):
    """Load EEG or MEG data according to the configuration."""
    # Check whether this is an MEG configuration (based on data directory).
    if 'THINGS-MEG' in config.get('data', {}).get('data_dir', ''):
        return meg_module.load_data(config)
    else:
        return load_eeg_data(config)


def load_model(config, train_loader, test_loader):
    model = {}
    for k, v in config['models'].items():
        print(f"init {k}")
        model[k] = instantiate_from_config(v)

    pl_model = PLModel(model, config, train_loader, test_loader)
    return pl_model



class PLModel(pl.LightningModule):
    def __init__(self, model, config, train_loader, test_loader):
        super().__init__()

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)

        self.criterion = ClipLoss_Modified_DDP(top_k=10, cos_batch=512)
        self.fusion_mod = CogcapFusion(modal_dims=[1024, 1024, 1024, 1024],
                                       hidden_dim=255,
                                       num_heads=1,
                                       dropout=0.1)
        self.fusion_eeg = CogcapFusion(modal_dims=[1024, 1024, 1024, 1024],
                                       hidden_dim=255,
                                       num_heads=1,
                                       dropout=0.1)
        self.modproj = ProjMod_multimodal()

        # Stage control parameters.
        self.staged_training = self.config.get('train', {}).get('staged_training', False)
        if self.staged_training:
            self.stage1_epochs = self.config['train']['stage1_epochs']
            self.stage2_epochs = self.config['train']['stage2_epochs']
            self.stage3_epochs = self.config['train']['stage3_epochs']

            total_train_epoch = self.config['train']['epoch']
            staged_epoch_sum = self.stage1_epochs + self.stage2_epochs + self.stage3_epochs
            if staged_epoch_sum != total_train_epoch:
                raise ValueError(
                    f"[Stage Training Config Error] Sum of stage epochs does not match total training epochs.\n"
                    f"- Stage epochs: stage1={self.stage1_epochs}, stage2={self.stage2_epochs}, stage3={self.stage3_epochs}\n"
                    f"- Stage sum: {staged_epoch_sum}\n"
                    f"- Config total epochs: {total_train_epoch}\n"
                    f"Please adjust stage1_epochs/stage2_epochs/stage3_epochs under config['train'] so their sum equals total epoch."
                )
        else:
            # Fall back to a "pseudo-stage": stage 1 = total epochs, others = 0.
            total = self.config['train']['epoch']
            self.stage1_epochs = total
            self.stage2_epochs = 0
            self.stage3_epochs = 0
        self.text_max_epochs = self.config.get('train', {}).get('text_max_epochs', 9999)


        # These metrics include fusion.
        self.all_predicted_classes = {'image': [], 'text': [], 'depth': [], 'edge': [], 'fusion': []}
        self.all_true_labels = {'image': [], 'text': [], 'depth': [], 'edge': [], 'fusion': []}
        self.mAP_total = {'image': 0, 'text': 0, 'depth': 0, 'edge': 0, 'fusion': 0}
        self.match_similarities = {'image': [], 'text': [], 'depth': [], 'edge': [], 'fusion': []}

        self.z_dim = self.config['z_dim']
        # (16540,) same as image; TODO: revisit this default size.
        # Check whether train_loader exists and has a dataset attribute.
        if train_loader is not None and hasattr(train_loader, 'dataset'):
            dataset_len = len(train_loader.dataset)
        else:
            # If train_loader is None or has no dataset, use a safe default length.
            dataset_len = 16540
            # Emit a warning when using fallback length.
            import warnings
            warnings.warn(f"train_loader is None or has no dataset attribute, using default length {dataset_len}")

        # Initialize and register self.sim as tensor buffers.
        self.sim = {}
        for key in ['image', 'text', 'depth', 'edge']:
            # Create tensors and register buffers (unique names required).
            sim_tensor = torch.ones(dataset_len, dtype=torch.float32)  # Initial value: 1.0
            self.register_buffer(f"sim_{key}", sim_tensor, persistent=True)  # Keep in state_dict.
            self.sim[key] = getattr(self, f"sim_{key}")  # Build dict mapping.

        # Initialize and register self.match_label as tensor buffers.
        self.match_label = {}
        for key in ['image', 'text', 'depth', 'edge']:
            # Create tensors and register buffers.
            label_tensor = torch.ones(dataset_len, dtype=torch.int64)  # Initial value: 1
            self.register_buffer(f"match_label_{key}", label_tensor, persistent=True)
            self.match_label[key] = getattr(self, f"match_label_{key}")  # Build dict mapping.

        self.alpha = 0.05
        self.gamma = 0.3
        self.mask_count = config['train'].get('mask_count', 2)
        self.automatic_optimization = False

    def setup(self, stage: str) -> None:
        """Initialize the loss function after DDP environment setup."""
        if self.criterion is None:
            # Select loss function according to config.
            if self.config.get('train', {}).get('loss_type', 'ClipLoss_Modified_DDP') == 'ClipLoss_Original':
                self.criterion = ClipLoss_Original()
            else:
                self.criterion = ClipLoss_Modified_DDP(top_k=10, cos_batch=512)

    def forward(self, batch, sample_posterior=False):
        def _uncertainty_aware(logits, key, sim_idx):
            """
            input: logit, key: modality_idx, sim_idx: similarity's idx
            """
            if not self.training:
                return

            diagonal_elements = torch.diagonal(logits).cpu().detach().numpy()
            sim_idx_tensor = torch.tensor(sim_idx, device=self.sim[key].device, dtype=torch.long)
            sim_values = self.sim[key][sim_idx_tensor].cpu().detach().numpy()  # Tensor indexing.
            gamma = self.gamma
            batch_sim = gamma * diagonal_elements + (1 - gamma) * sim_values  # Element-wise update.

            mean_sim = np.mean(batch_sim)
            std_sim = np.std(batch_sim, ddof=1)
            match_label = np.ones_like(batch_sim)
            z_alpha_2 = norm.ppf(1 - self.alpha / 2)

            lower_bound = mean_sim - z_alpha_2 * std_sim
            upper_bound = mean_sim + z_alpha_2 * std_sim

            match_label[diagonal_elements > upper_bound] = 0
            match_label[diagonal_elements < lower_bound] = 2

            # Critical change: assign values directly to tensors instead of NumPy arrays.
            # Note: sim_idx is a NumPy array (from batch['idx']) and must be converted.
            sim_idx_tensor = torch.tensor(sim_idx, device=self.sim[key].device, dtype=torch.long)  # Convert index to tensor.
            self.sim[key][sim_idx_tensor] = torch.tensor(batch_sim, device=self.sim[key].device, dtype=torch.float32)  # Update sim.
            self.match_label[key][sim_idx_tensor] = torch.tensor(match_label, device=self.sim[key].device,
                                                                 dtype=torch.int64)  # Update label.
            

        def _random_mask_non_eeg_modals(modals):
            """
            Randomly mask non-EEG modalities while keeping at least one.

            Args:
                modals: list of non-EEG modalities in order [img_z, text_z, depth_z, edge_z]

            Returns:
                masked_modals: modality list after masking (masked ones are zero vectors)
                mask_indices: indices of masked modalities (for debugging/visualization)
            """
            if not self.training:
                return modals, torch.tensor([], device=modals[0].device)  # Empty indices mean no masking.

            num_modal = len(modals)
            assert num_modal == 4, "Modality count mismatch"

            # Number of modalities to mask this round (bounded by max allowed).
            # Keep at least one modality, so at most 3 can be masked.
            actual_mask_num = min(self.mask_count, num_modal - 1) 
            actual_mask_num = max(actual_mask_num, 0)  # Ensure non-negative.

            # Randomly select modality indices to mask.
            mask_indices = torch.randperm(num_modal, device=modals[0].device)[:actual_mask_num]

            # Clone and zero out selected modalities.
            masked_modals = [modal.clone() for modal in modals]
            for idx in mask_indices:
                masked_modals[idx] = torch.zeros_like(masked_modals[idx])

            return masked_modals, mask_indices

        # ---------- 1. Feature extraction ----------
        idx = batch['idx'].cpu().detach().numpy()
        eeg = batch['eeg']
        img_z = batch['img_features']
        text_z = batch['text_features']
        depth_z = batch['depth_features']
        edge_z = batch['edge_features']

        img_z = img_z / img_z.norm(dim=-1, keepdim=True)
        text_z = text_z / text_z.norm(dim=-1, keepdim=True)
        depth_z = depth_z / depth_z.norm(dim=-1, keepdim=True)
        edge_z = edge_z / edge_z.norm(dim=-1, keepdim=True)

        eeg_z = self.brain(eeg)
        mod_output = self.modproj(img_z, text_z, depth_z, edge_z)

        # ---------- 2. Fusion ----------
        eeg_fusion = self.fusion_eeg(*[e.detach() for e in eeg_z])
        non_eeg = [mod_output[m] for m in ('image', 'text', 'depth', 'edge')]
        masked, _ = _random_mask_non_eeg_modals(non_eeg)
        mod_fusion = self.fusion_mod(*masked)

        # ---------- 3. Pack dictionaries ----------
        eeg_z = dict(zip(('image', 'text', 'depth', 'edge'), eeg_z))
        eeg_z['fusion'] = eeg_fusion
        mod_z = {**mod_output, 'fusion': mod_fusion}

        # ---------- 4. Unified loss + uncertainty computation ----------
        logit_scale = self.brain.softplus(self.brain.logit_scale)
        total_loss, logits = {}, {}
        for k in ('image', 'text', 'depth', 'edge', 'fusion'):
            if isinstance(self.criterion, ClipLoss_Original):
                # ClipLoss_Original returns 3 values: image_loss, text_loss, logits.
                image_loss, text_loss, logit_k = self.criterion(eeg_z[k], mod_z[k], logit_scale)
                total_loss[k] = (image_loss.mean() + text_loss.mean()) / 2
                logits[k] = logit_k
            else:
                # ClipLoss_Modified_DDP requires 4 arguments.
                image_loss, text_loss, logit_k = self.criterion(eeg_z[k], mod_z[k], logit_scale, batch['img_index'])
                total_loss[k] = (image_loss.mean() + text_loss.mean()) / 2
                logits[k] = logit_k

        if self.config['data']['uncertainty_aware']:
            for k in ('image', 'text', 'depth', 'edge'):  # Fusion excluded for now.
                _uncertainty_aware(logits[k], k, idx)

        return eeg_z, mod_z, total_loss

    def on_train_epoch_start(self):
        """Update staged-training status at the start of each training epoch."""
        current_epoch = self.current_epoch
        if not self.staged_training:  # switch=false -> always stage 2 (joint)
            self.current_stage = 2
            self._freeze_fusion_modules(False)
            self._freeze_single_modalities(False)
            return

        if current_epoch < self.stage1_epochs:
            self.current_stage = 1
            self._freeze_fusion_modules(True)  # Freeze fusion modules.
            self._freeze_single_modalities(False)  # Unfreeze single-modality modules.
        elif current_epoch < (self.stage1_epochs + self.stage2_epochs):
            self.current_stage = 2
            self._freeze_fusion_modules(False)  # Unfreeze fusion modules.
            self._freeze_single_modalities(False)  # Unfreeze single-modality modules.
        else:
            self.current_stage = 3
            self._freeze_fusion_modules(False)  # Unfreeze fusion modules.
            self._freeze_single_modalities(True)  # Freeze single-modality modules.

        # Print current stage information.
        if self.global_rank == 0:
            print(f"\n===== Epoch {current_epoch}: Stage {self.current_stage} =====")
            print(f"Fusion modules frozen: {self.fusion_frozen}")
            print(f"Single modalities frozen: {self.single_modalities_frozen}")

    def _freeze_fusion_modules(self, freeze=True):
        """Freeze or unfreeze fusion modules."""
        self.fusion_frozen = freeze
        self.fusion_mod.requires_grad_(not freeze)
        self.fusion_eeg.requires_grad_(not freeze)

        # Set modules to eval mode when frozen.
        if freeze:
            self.fusion_mod.eval()
            self.fusion_eeg.eval()

    def _freeze_single_modalities(self, freeze=True):
        """Freeze or unfreeze single-modality modules (brain + corresponding modproj)."""
        self.single_modalities_frozen = freeze
        # Iterate brain models and modproj models in pairs.
        for brain_model, modproj_model in zip(self.brain.models, self.modproj.models):
            # Freeze/unfreeze brain model parameters.
            brain_model.requires_grad_(not freeze)
            # Freeze/unfreeze corresponding modproj parameters.
            modproj_model.requires_grad_(not freeze)

            # Set to eval mode when frozen.
            if freeze:
                brain_model.eval()
                modproj_model.eval()
                
    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, mod_z, loss = self(batch, sample_posterior=True)

        # Compute loss based on current training stage.
        if self.current_stage == 1:
            loss = {k: loss[k] for k in ('image', 'text', 'depth', 'edge')}
        elif self.current_stage == 2:
            pass  # Keep all 5 keys.
        else:  # Stage 3
            loss = {'fusion': loss['fusion']}
        if self.current_epoch >= self.text_max_epochs and 'text' in loss:
            del loss['text']

        opt = self.optimizers()
        opt_dict = {
            'image': opt[0],
            'text': opt[1],
            'depth': opt[2],
            'edge': opt[3],
            'fusion': opt[4]
        }
        for idx, key in enumerate(loss):
            self.log(f'train_loss_{key}', loss[key], on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True,
                     batch_size=batch_size)
            opt_dict[key].zero_grad()
            self.manual_backward(loss[key], retain_graph=True)
            opt_dict[key].step()

        total_loss = sum(loss.values())
        self.log(f'total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True,
                 batch_size=batch_size)

        for key in eeg_z:
            eeg_z[key] = eeg_z[key] / eeg_z[key].norm(dim=-1, keepdim=True)

        similarity_new = {}
        for key in mod_z:
            similarity_new[key] = (eeg_z[key] @ mod_z[key].T)

        top_k_values_new = {}
        top_k_indices_new = {}
        for key in similarity_new:
            top_k_values_new[key], top_k_indices_new[key] = similarity_new[key].topk(5, dim=-1)

        for key in top_k_indices_new:
            self.all_predicted_classes.setdefault(key, [])
            self.all_predicted_classes[key].append(top_k_indices_new[key].cpu().numpy())

        label = torch.arange(0, batch_size).to(self.device)

        for key in self.all_true_labels:
            self.all_true_labels[key].extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = {}
            top_1_predictions = {}
            top_1_correct = {}
            top_1_accuracy = {}
            top_k_correct = {}
            top_k_accuracy = {}
            count_dict_mapped = {}
            for key in self.all_predicted_classes:
                all_predicted_classes[key] = np.concatenate(self.all_predicted_classes[key], axis=0)
                all_true_labels = np.array(self.all_true_labels[key])
                top_1_predictions[key] = all_predicted_classes[key][:, 0]
                top_1_correct[key] = top_1_predictions[key] == all_true_labels
                top_1_accuracy[key] = sum(top_1_correct[key]) / len(top_1_correct[key])

                top_k_correct[key] = (all_predicted_classes[key] == all_true_labels[:, np.newaxis]).any(axis=1)
                top_k_accuracy[key] = sum(top_k_correct[key]) / len(top_k_correct[key])

                self.log(f'train_top1_acc_{key}', top_1_accuracy[key], on_step=False, on_epoch=True, prog_bar=True,
                         logger=True,
                         sync_dist=True, batch_size=batch_size)
                self.log(f'train_top1_acc_{key}', top_k_accuracy[key], on_step=False, on_epoch=True, prog_bar=True,
                         logger=True,
                         sync_dist=True, batch_size=batch_size)

                if key != 'fusion':
                    # Fix: convert tensor -> NumPy -> Python int list before counting.
                    match_label_np = self.match_label[key].cpu().detach().numpy()  # Tensor -> NumPy array.
                    match_label_list = match_label_np.tolist()  # NumPy array -> Python int list.
                    counter = Counter(match_label_list)  # Counter keys are now Python ints.

                    count_dict = dict(counter)
                    key_mapping = {0: 'low', 1: 'medium', 2: 'high'}
                    count_dict_mapped[key] = {key_mapping[k]: v for k, v in count_dict.items()}
                    self.log_dict(count_dict_mapped[key], on_step=False, on_epoch=True, logger=True, sync_dist=True)
                    self.trainer.train_dataloader.dataset.match_label[key] = self.match_label[key].cpu().detach().numpy()

            self.all_predicted_classes = {'image': [], 'text': [], 'depth': [], 'edge': [], 'fusion': []}
            self.all_true_labels = {'image': [], 'text': [], 'depth': [], 'edge': [], 'fusion': []}

        return total_loss

    def validation_step(self, batch, batch_idx): # todo change this for ddp
        batch_size = batch['idx'].shape[0]

        eeg_z, mod_z, loss = self(batch)
        for key in loss:
            self.log(f'val_loss_{key}', loss[key], on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True,
                     batch_size=batch_size)

        total_loss = sum(loss.values())
        self.log(f'val_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True,
                 batch_size=batch_size)

        for key in eeg_z:
            eeg_z[key] = eeg_z[key] / eeg_z[key].norm(dim=-1, keepdim=True)

        similarity = {}

        for key in mod_z:
            similarity[key] = (eeg_z[key] @ mod_z[key].T)
            top_k_values, top_k_indices = similarity[key].topk(5, dim=-1)
            self.all_predicted_classes[key].append(top_k_indices.cpu().numpy())
            label = torch.arange(0, batch_size).to(self.device)
            self.all_true_labels[key].extend(label.cpu().numpy())

        return total_loss

    # def on_validation_epoch_end(self):
    #     for key in self.all_predicted_classes:
    #         all_predicted_classes = np.concatenate(self.all_predicted_classes[key], axis=0)
    #         all_true_labels = np.array(self.all_true_labels[key])
    #         top_1_predictions = all_predicted_classes[:, 0]
    #         top_1_correct = top_1_predictions == all_true_labels
    #         top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
    #         top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
    #         top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
    #
    #         self.log(f'val_top1_acc_{key}', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
    #                  sync_dist=True)
    #         self.log(f'val_top5_acc_{key}', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
    #                  sync_dist=True)
    #
    #     self.all_predicted_classes = {'image': [], 'text': [], 'depth': [], 'edge': []}
    #     self.all_true_labels = {'image': [], 'text': [], 'depth': [], 'edge': []}

    def on_validation_epoch_end(self):
        # Store predictions and labels for all modalities.
        modality_predictions = {}
        modality_labels = {}
        sample_count = None

        # Step 1: Iterate modalities, compute Top-1/Top-5, and cache data.
        for key in self.all_predicted_classes:
            all_predicted = np.concatenate(self.all_predicted_classes[key], axis=0)  # (N, k)
            all_true = np.array(self.all_true_labels[key])  # (N,)
            modality_predictions[key] = all_predicted
            modality_labels[key] = all_true

            # Ensure sample counts and labels are consistent.
            if sample_count is None:
                sample_count = len(all_true)
            else:
                assert len(all_true) == sample_count, f"Validation modality {key} sample count mismatch"
            if key != 'image':
                assert np.all(all_true == modality_labels['image']), f"Validation modality {key} label mismatch"

            # Original logic: per-modality Top-1 and Top-5.
            top1_acc = np.mean(all_predicted[:, 0] == all_true)
            top5_acc = np.mean(np.any(all_predicted == all_true[:, np.newaxis], axis=1))
            self.log(f'val_top1_acc_{key}', top1_acc, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f'val_top5_acc_{key}', top5_acc, on_epoch=True, prog_bar=True, sync_dist=True)

        # Step 2: Cross-modality Top-1 accuracy (existing).
        combined_top1 = np.stack([modality_predictions[key][:, 0] for key in modality_predictions], axis=1)
        any_top1_correct = np.any(combined_top1 == modality_labels['image'][:, np.newaxis], axis=1)
        any_top1_acc = np.mean(any_top1_correct)
        self.log('val_any_modality_top1_acc', any_top1_acc, on_epoch=True, prog_bar=True, sync_dist=True)

        # Step 3: Cross-modality Top-5 accuracy.
        # Stack Top-5 predictions from all modalities (shape (N, 4, 5)).
        combined_top5 = np.stack(
            [modality_predictions[key][:, :5] for key in ['image', 'text', 'depth', 'edge']],
            axis=1  # Dim-1: modality, dim-2: Top-5 predictions.
        )  # Shape: (N, 4, 5)

        # Expand labels to (N, 1, 1) for broadcast comparison.
        expanded_labels = modality_labels['image'][:, np.newaxis, np.newaxis]  # (N, 1, 1)

        # Check whether each modality's Top-5 contains the ground-truth label (N, 4).
        modality_top5_correct = np.any(combined_top5 == expanded_labels, axis=2)

        # Mark sample correct if any modality is correct (N,).
        any_top5_correct = np.any(modality_top5_correct, axis=1)

        # Compute accuracy.
        any_top5_acc = np.mean(any_top5_correct)
        self.log('val_any_modality_top5_acc', any_top5_acc, on_epoch=True, prog_bar=True, sync_dist=True)

        # Reset caches.
        self.all_predicted_classes = {'image': [], 'text': [], 'depth': [], 'edge': [], 'fusion': []}
        self.all_true_labels = {'image': [], 'text': [], 'depth': [], 'edge': [], 'fusion': []}

        return

    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, mod_z, loss = self(batch)
        for key in loss:
            self.log(f'test_loss_{key}', loss[key], on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True,
                     batch_size=batch_size)

        total_loss = sum(loss.values())
        self.log(f'test_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True,
                 batch_size=batch_size)

        for key in eeg_z:
            eeg_z[key] = eeg_z[key] / eeg_z[key].norm(dim=-1, keepdim=True)

        for key in mod_z:
            similarity = (eeg_z[key] @ mod_z[key].T)

            top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
            self.all_predicted_classes[key].append(top_k_indices.cpu().numpy())
            label = batch['label']
            self.all_true_labels[key].extend(label.cpu().numpy())

            # compute sim and map
            self.match_similarities[key].extend(similarity.diag().detach().cpu().tolist())

            for i in range(similarity.shape[0]):
                true_index = i
                sims = similarity[i, :]
                sorted_indices = torch.argsort(-sims)
                rank = (sorted_indices == true_index).nonzero()[0][0] + 1
                ap = 1 / rank
                self.mAP_total[key] += ap

        return total_loss

    def on_test_epoch_end(self):
        # Store predictions, labels, and mAP-related data for all modalities.
        modality_predictions = {}
        modality_labels = {}
        sample_count = None

        # Step 1: Iterate modalities and compute per-modality metrics.
        for key in self.all_predicted_classes:
            all_predicted = np.concatenate(self.all_predicted_classes[key], axis=0)  # (N, k)
            all_true = np.array(self.all_true_labels[key])  # (N,)
            modality_predictions[key] = all_predicted
            modality_labels[key] = all_true

            # Ensure sample counts and labels are consistent.
            if sample_count is None:
                sample_count = len(all_true)
            else:
                assert len(all_true) == sample_count, f"Test modality {key} sample count mismatch"
            if key != 'image':
                assert np.all(all_true == modality_labels['image']), f"Test modality {key} label mismatch"

            # Original logic: compute Top-1, Top-5, mAP, etc. per modality.
            top1_acc = np.mean(all_predicted[:, 0] == all_true)
            top5_acc = np.mean(np.any(all_predicted == all_true[:, np.newaxis], axis=1))
            mAP = (self.mAP_total[key].cpu().item() / sample_count)
            similarity = np.mean(self.match_similarities[key]) if self.match_similarities[key] else 0

            self.log(f'test_top1_acc_{key}', top1_acc, sync_dist=True)
            self.log(f'test_top5_acc_{key}', top5_acc, sync_dist=True)
            self.log(f'mAP_{key}', mAP, sync_dist=True)
            self.log(f'similarity_{key}', similarity, sync_dist=True)

        # Step 2: Cross-modality Top-1 accuracy (existing).
        combined_top1 = np.stack([modality_predictions[key][:, 0] for key in modality_predictions], axis=1)
        any_top1_acc = np.mean(np.any(combined_top1 == modality_labels['image'][:, np.newaxis], axis=1))
        self.log('test_any_modality_top1_acc', any_top1_acc, sync_dist=True)

        # Step 3: Cross-modality Top-5 accuracy.
        combined_top5 = np.stack(
            [modality_predictions[key][:, :5] for key in ['image', 'text', 'depth', 'edge']],
            axis=1  # Dim-1: modality, dim-2: Top-5 predictions.
        )  # Shape: (N, 4, 5)
        expanded_labels = modality_labels['image'][:, np.newaxis, np.newaxis]  # (N, 1, 1)
        modality_top5_correct = np.any(combined_top5 == expanded_labels, axis=2)
        any_top5_correct = np.any(modality_top5_correct, axis=1)

        # Compute accuracy.
        any_top5_acc = np.mean(any_top5_correct)
        self.log('test_any_modality_top5_acc', any_top5_acc, sync_dist=True)

        # Reset caches.
        self.all_predicted_classes = {'image': [], 'text': [], 'depth': [], 'edge': [], 'fusion': []}
        self.all_true_labels = {'image': [], 'text': [], 'depth': [], 'edge': [], 'fusion': []}

        # Build output summary.
        avg_test_loss = self.trainer.callback_metrics['test_total_loss']
        output = {'test_total_loss': avg_test_loss.item()}
        for key in ['image', 'text', 'depth', 'edge', 'fusion']:
            mid = {
                f'test_top1_acc_{key}': np.mean(modality_predictions[key][:, 0] == modality_labels[key]),
                f'test_top5_acc_{key}': np.mean(
                    np.any(modality_predictions[key] == modality_labels[key][:, np.newaxis], axis=1)),
                f'mAP_{key}': (self.mAP_total[key].cpu().item() / sample_count),
                f'similarity_{key}': np.mean(self.match_similarities[key]) if self.match_similarities[key] else 0
            }
            output.update(mid)
        output.update({
            'test_any_modality_top1_acc': any_top1_acc,
            'test_any_modality_top5_acc': any_top5_acc,
        })
        return output

    def configure_optimizers(self):
        # Original optimizer configuration.
        # original:
        # optimizer = globals()[self.config['train']['optimizer']](self.parameters(),
        #                                                          lr=self.config['train']['lr'],
        #                                                          weight_decay=1e-4)

        # seperate:
        optimizers = []
        # Collect parameters from all brain and modproj models.
        all_brain_params = []
        for model in self.brain.models:
            all_brain_params.extend(list(model.parameters()))

        all_modproj_params = []
        for model in self.modproj.models:
            all_modproj_params.extend(list(model.parameters()))

        # Merge all parameters.
        combined_params = all_brain_params + all_modproj_params

        # Use the same full parameter set for each optimizer (keeping original count).
        for i in range(len(self.brain.models)):
            optimizer = globals()[self.config['train']['optimizer']](
                combined_params,  # Use all model parameters here.
                lr=self.config['train']['lr'],
                weight_decay=1e-4
            )
            optimizers.append(optimizer)
        # 2. Collect fusion module parameters (fusion_mod and fusion_eeg).
        fusion_params = list(self.fusion_mod.parameters()) + list(self.fusion_eeg.parameters())
        fusion_optimizer = globals()[self.config['train']['optimizer']](
            fusion_params,
            lr=self.config['train']['lr'],  # Adjust fusion LR if needed.
            weight_decay=1e-4
        )
        optimizers.append(fusion_optimizer)  # Add to optimizer list.

        return optimizers

