"""Image generation helpers for the refactored runtime."""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import numpy as np
import PIL.Image
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

from ..runtime.paths import (
    DEFAULT_DIFFUSION_EMBEDDINGS_REL,
    load_runtime_config,
    merge_with_public_config,
    optional_root,
    resolve_dataset_root,
    resolve_diffusion_embeddings_root,
    resolve_ip_adapter_root,
    resolve_path_like,
    resolve_sdxl_root,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def show_image(image, title):
    print(f"Display image: {title}")
    image.show()


def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_generation_config(config_path: str = "configs/cogcappro.yaml", data_type: str = "EEG"):
    config = load_runtime_config(config_path)
    return merge_with_public_config(config, data_type=data_type.upper())


def _preferred_diffusion_embeddings_root(config, data_type: str = "EEG") -> Path:
    resolved = resolve_diffusion_embeddings_root(config, required=False)
    if resolved is not None:
        return resolved

    weights_root = optional_root(config, "weights_root")
    if weights_root is not None:
        rel_value = config.paths.get("diffusion_embeddings_rel", DEFAULT_DIFFUSION_EMBEDDINGS_REL)
        return resolve_path_like(rel_value, weights_root)

    dataset_root = resolve_dataset_root(config, data_type)
    return dataset_root / "Image_feature_new" / "data_features"


def resolve_generator_model_paths(
    *,
    config_path: str = "configs/cogcappro.yaml",
    data_type: str = "EEG",
    sd_path: str | None = None,
    ip_adapter_path: str | None = None,
):
    config = load_generation_config(config_path=config_path, data_type=data_type)
    resolved_sd_path = resolve_sdxl_root(config, override=sd_path)
    resolved_ip_adapter_path = resolve_ip_adapter_root(config, override=ip_adapter_path)
    return config, str(resolved_sd_path), str(resolved_ip_adapter_path)


class IPAdapterGenerator:
    """IP-Adapter generator with a concise image generation interface."""

    _MODALITY_SCALES = {
        "image": {
            "down": {"block_2": [1.0, 1.0]},
            "up": {"block_0": [1.0, 1.0, 1.0]},
        },
        "depth": {
            "down": {"block_2": [0.0, 0.5]},
            "up": {"block_0": [0.0, 0.0, 0.0]},
        },
        "edge": {
            "down": {"block_2": [0.0, 0.5]},
            "up": {"block_0": [0.0, 0.0, 0.0]},
        },
    }

    def __init__(
        self,
        sd_path,
        ip_adapter_path,
        device=None,
        seed=42,
        num_inference_steps=5,
        guidance_scale=0.0,
        modalities=None,
        modality_scales=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.modalities = modalities or ["image", "depth", "edge"]
        self.num_modalities = len(self.modalities)

        if modality_scales is not None:
            self._MODALITY_SCALES = modality_scales

        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            ip_adapter_path,
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        ).to(self.device)

        self._init_pipeline(sd_path, ip_adapter_path)

    def _init_pipeline(self, sd_path, ip_adapter_path):
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            sd_path,
            image_encoder=self.image_encoder,
            torch_dtype=torch.float16,
            safety_checker=None,
            use_safetensors=True,
            variant="fp16",
        )
        self.pipe.upcast_vae()
        self.pipe.load_ip_adapter(
            ip_adapter_path,
            subfolder="sdxl_models",
            weight_name=["ip-adapter_sdxl_vit-h.safetensors"] * self.num_modalities,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        if hasattr(self.pipe, "vae") and getattr(self.pipe.vae.config, "force_upcast", False):
            self.pipe.vae.config.force_upcast = False
            self.pipe.vae.to(dtype=torch.float16)

        scales = [self._MODALITY_SCALES[mod] for mod in self.modalities]
        self.pipe.set_ip_adapter_scale(scales)
        self.pipe.enable_model_cpu_offload(device=self.device)

    def _prepare_embeddings(self, embeds_dict):
        ip_embeds = []
        for mod in self.modalities:
            embed = embeds_dict[mod]
            if embed.ndim == 1:
                embed = embed.unsqueeze(0)
            elif embed.shape[0] != 1:
                embed = embed[:1]

            embed = embed.to(torch.float16).to(self.device)
            uncond_embed = torch.zeros_like(embed, dtype=embed.dtype, device=self.device)
            ip_embeds.append(torch.stack([uncond_embed, embed], dim=0))
        return ip_embeds

    def _image_to_embedding(self, image):
        if isinstance(image, str):
            image = load_image(image)
        elif not isinstance(image, PIL.Image.Image):
            raise ValueError("Input image must be a file path or a PIL.Image object")

        return self.pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=[image],
            ip_adapter_image_embeds=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )

    def generate(self, input_data_dict, prompt="", negative_prompt="deformed, ugly, low quality"):
        embeds_dict = {}
        for mod in self.modalities:
            data = input_data_dict[mod]
            if isinstance(data, (str, PIL.Image.Image)):
                embeds = self._image_to_embedding(data)
                embeds_dict[mod] = embeds[0][1]
            elif isinstance(data, (torch.Tensor, np.ndarray)):
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                embeds_dict[mod] = data

        ip_adapter_embeds = self._prepare_embeddings(embeds_dict)
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image_embeds=ip_adapter_embeds,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=self.generator,
        ).images[0]


def prepare_embedding(
    generator,
    modalities=None,
    config_path: str = "configs/cogcappro.yaml",
    data_type: str = "EEG",
    output_dir: str | None = None,
):
    """Generate diffusion_clip_embeddings_{train,test}.pt for the ThingsEEG layout."""
    config = load_generation_config(config_path=config_path, data_type=data_type)
    dataset_root = resolve_dataset_root(config, data_type)
    embeddings_root = Path(output_dir) if output_dir else _preferred_diffusion_embeddings_root(config, data_type)
    train_output_file = embeddings_root / "diffusion_clip_embeddings_train.pt"
    test_output_file = embeddings_root / "diffusion_clip_embeddings_test.pt"

    modality_mapping = {
        "image": "Image_set_Resize",
        "depth": "Image_depth_set_Resize",
        "edge": "Image_edge_set_Resize",
    }

    modalities = modalities or ["image", "depth", "edge"]
    valid_modalities = [modality for modality in modalities if modality in modality_mapping]
    if not valid_modalities:
        raise ValueError(f"Invalid modality. Supported modalities: {list(modality_mapping.keys())}")

    train_embeddings = {modality: {} for modality in valid_modalities}
    test_embeddings = {modality: {} for modality in valid_modalities}
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")

    for modality in valid_modalities:
        modal_dir = dataset_root / modality_mapping[modality]
        if not modal_dir.exists():
            logger.warning("Folder for modality %s does not exist: %s. Skipping this modality", modality, modal_dir)
            continue

        image_paths = []
        for root, _, files in os.walk(modal_dir):
            for file_name in files:
                if file_name.lower().endswith(supported_extensions):
                    image_paths.append(os.path.join(root, file_name))

        if not image_paths:
            logger.warning("No image files found under modality %s", modality)
            continue

        logger.info("Start processing modality %s, total %s images", modality, len(image_paths))
        for img_path in tqdm(image_paths, desc=f"Processing {modality} modality"):
            try:
                img_filename = os.path.basename(img_path)
                rel_path = os.path.relpath(img_path, modal_dir)
                if rel_path.startswith("train_images"):
                    target_dict = train_embeddings[modality]
                elif rel_path.startswith("test_images"):
                    target_dict = test_embeddings[modality]
                else:
                    logger.warning("Image %s does not belong to train or test split. Skipping", img_filename)
                    continue

                with Image.open(img_path) as source_img:
                    img = source_img.convert("RGB")
                    with torch.no_grad():
                        embeddings = generator._image_to_embedding(img)
                        valid_embedding = embeddings[0][1].float().cpu()

                if img_filename in target_dict:
                    logger.warning("Duplicate filename %s; previous embedding will be overwritten", img_filename)
                target_dict[img_filename] = valid_embedding

                del embeddings
                torch.cuda.empty_cache()
            except Exception as exc:
                logger.error("Error processing image %s: %s. Skipping this image", img_path, exc)
                if "embeddings" in locals():
                    del embeddings
                torch.cuda.empty_cache()
                continue

    embeddings_root.mkdir(parents=True, exist_ok=True)
    torch.save(train_embeddings, train_output_file)
    torch.save(test_embeddings, test_output_file)
    logger.info("Train embeddings saved to %s", train_output_file)
    logger.info("Test embeddings saved to %s", test_output_file)
    return train_embeddings, test_embeddings


def load_embeddings(file_path, image_name="banana_09s.jpg"):
    """Load a subset of embeddings from a .pt file."""
    embeddings = torch.load(file_path, map_location=torch.device("cpu"))
    available_modalities = [
        mod for mod in ["image", "depth", "edge"] if mod in embeddings and image_name in embeddings[mod]
    ]
    if not available_modalities:
        raise ValueError(f"Image {image_name} not found in {file_path}")

    return {mod: embeddings[mod][image_name] for mod in available_modalities}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cogcappro.yaml", help="Runtime config path")
    parser.add_argument("--data_type", type=str, default="EEG", choices=["EEG", "MEG"], help="Config profile")
    parser.add_argument("--sd_path", type=str, default=None, help="Path to the Stable Diffusion model directory")
    parser.add_argument("--ip_adapter_path", type=str, default=None, help="Path to the IP-Adapter model directory")
    parser.add_argument("--embedding_path", type=str, default="generated_embeddings.pt", help="Path to generated embeddings")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device type",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--modalities", type=str, nargs="+", default=["image", "depth", "edge"], help="Modalities to use")
    parser.add_argument("--image_name", type=str, default="aircraft_carrier_06s.jpg", help="Image name to load")
    args = parser.parse_args()

    embeddings_dict = load_embeddings(args.embedding_path, args.image_name)
    seed_everything(args.seed)
    _, resolved_sd_path, resolved_ip_adapter_path = resolve_generator_model_paths(
        config_path=args.config,
        data_type=args.data_type,
        sd_path=args.sd_path,
        ip_adapter_path=args.ip_adapter_path,
    )

    generator = IPAdapterGenerator(
        sd_path=resolved_sd_path,
        ip_adapter_path=resolved_ip_adapter_path,
        device=args.device,
        seed=args.seed,
        num_inference_steps=15,
        guidance_scale=0.0,
        modalities=args.modalities,
    )

    generated_image = generator.generate(
        input_data_dict=embeddings_dict,
        prompt="",
        negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
    )
    show_image(generated_image, "Multi IP-Adapter")


if __name__ == "__main__":
    main()
