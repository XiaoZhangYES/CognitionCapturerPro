import random
import importlib
import math
import subprocess

import numpy as np
import torch
from torch import distributed as dist, nn as nn
from torch.nn import functional as F


TARGET_PREFIX_ALIASES = {
    "base.brain_backbone.": "cogcappro.models.brain_backbone.",
    "base.eeg_backbone.": "cogcappro.models.brain_backbone.",
    "base.fusion_backbone.": "cogcappro.models.fusion_backbone.",
    "base.inpainting_data.": "cogcappro.models.inpainting_data.",
    "base.inpating_data.": "cogcappro.models.inpainting_data.",
    "base.utils.": "cogcappro.utils.",
    "base.eeg_dataset.": "cogcappro.data.eeg.",
    "base.meg_dataset.": "cogcappro.data.meg.",
}


def normalize_target(target: str) -> str:
    for old_prefix, new_prefix in TARGET_PREFIX_ALIASES.items():
        if target.startswith(old_prefix):
            return f"{new_prefix}{target[len(old_prefix):]}"
    return target


def get_obj_from_str(string, reload=False):
    normalized = normalize_target(string)
    module, cls = normalized.rsplit(".", 1)
    try:
        module_imp = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if not module.startswith("cogcappro."):
            raise
        fallback_module = module.replace("cogcappro.", "src.cogcappro.", 1)
        module_imp = importlib.import_module(fallback_module)
    if reload:
        importlib.reload(module_imp)
    return getattr(module_imp, cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    target = normalize_target(config["target"])
    if target.endswith("StableDiffusionXLPipeline"):
        return get_obj_from_str(target).from_pretrained(
            **config.get("params", dict()) if config.get("params", dict()) else {})
    else:
        return get_obj_from_str(target)(
            **config.get("params", dict()) if config.get("params", dict()) else {})


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_config(args, config):
    for key in config.keys():
        if hasattr(args, key):
            if getattr(args, key) != None:
                config[key] = getattr(args, key)
    for key in args.__dict__.keys():
        config[key] = getattr(args, key)
    return config


def get_device(gpu_ids):
    if gpu_ids == 'auto':
        nvidia_smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free,temperature.gpu', '--format=csv,noheader,nounits'])
        gpu_info_lines = nvidia_smi_output.decode('utf-8').strip().split('\n')
        gpu_info = []
        for line in gpu_info_lines:
            gpu_data = line.strip().split(', ')
            index, memory_free, temperature = map(int, gpu_data)
            gpu_info.append((index, memory_free, temperature))
        gpu_info.sort(key=lambda x: x[1], reverse=True)

        memeory_rank_num = math.ceil(0.4 * len(gpu_info))
        selected_gpus = gpu_info[:memeory_rank_num]
        selected_gpus.sort(key=lambda x: x[2])
        selected_device = selected_gpus[0][0]
        # device = torch.device(f'cuda:{selected_device}')
    elif gpu_ids == "cpu":
        device = torch.device('cpu')
    else:
        gpu_ids = list(map(int, gpu_ids.split(",")))
        selected_device = gpu_ids[0]
        # device = torch.device(f'cuda:{selected_device}')
    return selected_device


# def get_device(gpu_ids):
#     if gpu_ids == 'auto':
#         nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,temperature.gpu', '--format=csv,noheader,nounits'])
#         gpu_info_lines = nvidia_smi_output.decode('utf-8').strip().split('\n')
#         gpu_info = []
#         for line in gpu_info_lines:
#             gpu_data = line.strip().split(', ')
#             index, memory_free, temperature = map(int, gpu_data)
#             gpu_info.append((index, memory_free, temperature))
#         gpu_info.sort(key=lambda x: x[1], reverse=True)
#
#         memeory_rank_num = math.ceil(0.4 * len(gpu_info))
#         selected_gpus = gpu_info[:memeory_rank_num]
#         selected_gpus.sort(key=lambda x: x[2])
#         selected_devices = [gpu[0] for gpu in selected_gpus]
#     elif gpu_ids == "cpu":
#         selected_devices = ['cpu']
#     else:
#         selected_devices = list(map(int, gpu_ids.split(",")))
#     return selected_devices

class ClipLoss(nn.Module):  # Original Paper's loss. zkf
    def __init__(self):
        super().__init__()

    def compute_ranking_weights(self, loss_list):
        sorted_indices = torch.argsort(loss_list)
        weights = torch.zeros_like(loss_list)
        for i, idx in enumerate(sorted_indices):
            weights[idx] = 1 / (i + 1)
        return weights

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        image_loss = F.cross_entropy(logits_per_image, labels, reduction='none')
        text_loss = F.cross_entropy(logits_per_text, labels, reduction='none')

        # total_loss = (image_loss + text_loss) / 2

        return image_loss, text_loss, logits_per_image


class ClipLoss_Modified(nn.Module):
    def __init__(self, top_k=10, cos_batch=256):
        super().__init__()
        self.cos_batch = cos_batch
        self.top_k = top_k

    def _get_similarity(self, text_features):
        text_features_normalized = torch.nn.functional.normalize(text_features, p=2, dim=1)
        N = text_features_normalized.size(0)
        batch_size = self.cos_batch
        cosine_sims = torch.zeros(N, N, device=text_features.device)

        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            batch_vectors = text_features_normalized[i:end_i]
            block_sim = torch.mm(batch_vectors, text_features_normalized.T)
            cosine_sims[i:end_i] = block_sim

        return cosine_sims

    def _get_class_mask(self, labels):
        labels_a = labels.unsqueeze(0)
        labels_b = labels.unsqueeze(1)
        mask = labels_a == labels_b
        similarity_matrix = mask.int()
        return similarity_matrix

    def forward(self, image_features, text_features, logit_scale, img_index):
        device = image_features.device

        # 1. Calculate image-text similarity
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        with torch.no_grad():
            # 2. Get text feature similarity matrix
            sim = self._get_similarity(text_features)

            # 3. Create top-k mask (keep most similar samples)
            values, indices = torch.topk(
                sim.clone().fill_diagonal_(0),
                k=min(self.top_k, sim.shape[0] - 1),  # Ensure k doesn't exceed number of samples
                dim=1,
                sorted=False
            )
            mask_sim = torch.zeros_like(sim)
            mask_sim.scatter_(1, indices, 1).fill_diagonal_(1)  # Keep diagonal (self)

            # 4. Create class mask (samples of the same class)
            mask_class = self._get_class_mask(img_index)

            # 5. Combine masks and normalize
            sim_mask = (mask_sim * mask_class).float()
            row_sums = sim_mask.sum(dim=1, keepdim=True)
            row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)  # Prevent division by 0
            labels = sim_mask / row_sums

            labels = labels.to(device)

        # 6. Calculate task loss
        image_loss = F.cross_entropy(logits_per_image, labels, reduction='mean')
        text_loss = F.cross_entropy(logits_per_text, labels, reduction='mean')

        return image_loss, text_loss, logits_per_image


class ClipLoss_Modified_DDP(nn.Module):
    def __init__(self, top_k=10, cos_batch=256):
        super().__init__()
        self.cos_batch = cos_batch
        self.top_k = top_k
        self.world_size = 1
        self.rank = 0

    def _get_similarity(self, text_features):
        text_features_normalized = torch.nn.functional.normalize(text_features, p=2, dim=1)
        N = text_features_normalized.size(0)
        batch_size = self.cos_batch
        cosine_sims = torch.zeros(N, N, device=text_features.device)

        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            batch_vectors = text_features_normalized[i:end_i]
            block_sim = torch.mm(batch_vectors, text_features_normalized.T)
            cosine_sims[i:end_i] = block_sim

        return cosine_sims

    def _get_class_mask(self, labels):
        labels_a = labels.unsqueeze(0)
        labels_b = labels.unsqueeze(1)
        mask = labels_a == labels_b
        similarity_matrix = mask.int()
        return similarity_matrix

    def _gather_tensors(self, tensor):
        """Gather tensors from all GPUs (fixing device issues)"""
        # Critical fix: Get world_size and rank at runtime
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

        if self.world_size == 1:
            return tensor

        # Get tensor list from all processes
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)

        # Exclude placeholder from current process
        tensor_list[self.rank] = tensor
        return torch.cat(tensor_list, dim=0)

    def forward(self, image_features, text_features, logit_scale, img_index):
        device = image_features.device

        local_batch_size = image_features.size(0)  # Local sample count for each GPU

        gathered_text = self._gather_tensors(text_features)
        gathered_img_index = self._gather_tensors(img_index)
        gathered_image = self._gather_tensors(image_features)

        # 1. Calculate image-text similarity
        logits_per_image = logit_scale * gathered_image @ gathered_text.T
        logits_per_text = logit_scale * gathered_text @ gathered_image.T

        # 4. Calculate local logit: Split global logit by rank, keeping only the data portion for current GPU
        # Global starting index for current GPU's data = rank * local batch size
        start_idx = self.rank * local_batch_size
        end_idx = start_idx + local_batch_size
        local_logits_per_image = logits_per_image[start_idx:end_idx, start_idx:end_idx]  # Split current GPU's logit
        local_logits_per_text = logits_per_text[start_idx:end_idx, start_idx:end_idx]    # Same for text side

        with torch.no_grad():
            # 2. Get text feature similarity matrix
            sim = self._get_similarity(gathered_text)

            # 3. Create top-k mask (keep most similar samples)
            values, indices = torch.topk(
                sim.clone().fill_diagonal_(0),
                k=min(self.top_k, sim.shape[0] - 1),  # Ensure k doesn't exceed sample count
                dim=1,
                sorted=False
            )
            mask_sim = torch.zeros_like(sim)
            mask_sim.scatter_(1, indices, 1).fill_diagonal_(1)  # Keep diagonal (self)

            # 4. Create class mask (samples of the same class)
            mask_class = self._get_class_mask(gathered_img_index)

            # 5. Combine masks and normalize
            sim_mask = (mask_sim * mask_class).float()
            row_sums = sim_mask.sum(dim=1, keepdim=True)
            row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)  # Prevent division by 0
            labels = sim_mask / row_sums

            labels = labels.to(device)

        # 6. Calculate task loss
        image_loss = F.cross_entropy(logits_per_image, labels, reduction='mean')
        text_loss = F.cross_entropy(logits_per_text, labels, reduction='mean')

        return image_loss, text_loss, local_logits_per_image

class ClipLoss_Original(nn.Module):
    """
    Original CLIP loss function implementation, used as baseline
    """
    def __init__(self, local_loss=False, gather_with_grad=False, rank=0, world_size=1, use_horovod=False):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def _gather_tensors(self, image_features, text_features):
        """Gather tensors from all GPUs (refer to ClipLoss_Modified_DDP implementation)"""
        # Only use torch.distributed, do not use horovod
        if self.gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(self.world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(self.world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            
            if not self.local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[self.rank] = image_features
                gathered_text_features[self.rank] = text_features
                
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

        return all_image_features, all_text_features

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = self._gather_tensors(
                image_features, text_features)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logit_scale * all_text_features @ all_image_features.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            self.labels[device] = labels
            self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        image_loss = F.cross_entropy(logits_per_image, labels, reduction='none')
        text_loss = F.cross_entropy(logits_per_text, labels, reduction='none')
        
        # Return same format as ClipLoss_Modified_DDP: (image_loss, text_loss, logits)
        return image_loss, text_loss, logits_per_image


def clip_loss_original():
    """Test ClipLoss_Original class"""
    clip_loss_original = ClipLoss_Original()
    image_features_original = torch.randn(10, 512)
    text_features_original = torch.randn(10, 512)
    logit_scale_original = torch.tensor(1.0)
    image_loss, text_loss, logits = clip_loss_original(image_features_original, text_features_original, logit_scale_original)
    print(f"Original CLIP Loss - Image: {image_loss}, Text: {text_loss}")
    print(f"Logits shape: {logits.shape}")
    return image_loss, text_loss, logits


if __name__ == '__main__':
    # Test ClipLoss_Original
    clip_loss_original()

