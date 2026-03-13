"""Batch image generation entrypoint for align outputs."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm

from .generator import (
    IPAdapterGenerator,
    resolve_generator_model_paths,
    seed_everything,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_images_from_pt_file(
    generator,
    pt_file_path,
    output_dir,
    modalities=None,
    use_before_align=False,
    resume_generation=False,
):
    """Generate images from one generated_embeddings*.pt file."""
    modalities = modalities or ["image", "depth", "edge"]
    try:
        embeddings = torch.load(pt_file_path, map_location="cpu")
        os.makedirs(output_dir, exist_ok=True)

        image_names = list(embeddings.get("image", {}).keys())
        if not image_names:
            logger.warning("No image embeddings found in %s", pt_file_path)
            return

        logger.info("Starting to process %s images", len(image_names))
        processed_count = 0
        skipped_count = 0

        for image_name in tqdm(image_names, desc=f"Generating images from {os.path.basename(pt_file_path)}"):
            try:
                image_output_path = os.path.join(output_dir, image_name)
                if resume_generation and os.path.exists(image_output_path):
                    skipped_count += 1
                    continue

                input_data_dict = {}
                for mod in modalities:
                    mod_key = f"{mod}_before" if use_before_align else mod
                    if mod_key in embeddings and image_name in embeddings[mod_key]:
                        input_data_dict[mod] = embeddings[mod_key][image_name]
                    else:
                        logger.warning("No %s modality found for %s in %s", mod_key, image_name, pt_file_path)

                if not input_data_dict:
                    logger.warning("No available modality data for generating %s", image_name)
                    continue

                generated_image = generator.generate(
                    input_data_dict=input_data_dict,
                    prompt="",
                    negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
                )
                generated_image.save(image_output_path)
                processed_count += 1
            except Exception as exc:
                logger.error("Error generating image %s: %s", image_name, exc)
                continue

        logger.info(
            "Finished processing %s, generated %s images, skipped %s images",
            pt_file_path,
            processed_count,
            skipped_count,
        )
    except Exception as exc:
        logger.error("Error processing %s: %s", pt_file_path, exc)


def batch_generate_images(
    base_dir,
    sd_path=None,
    ip_adapter_path=None,
    modality_mode="all",
    modality_scales=None,
    device=None,
    subjects=None,
    use_before_align=False,
    use_original_file=False,
    resume_generation=False,
    config_path="configs/cogcappro.yaml",
    data_type="EEG",
):
    """Batch-generate images for all matching subject runs under one experiment root."""
    _, resolved_sd_path, resolved_ip_adapter_path = resolve_generator_model_paths(
        config_path=config_path,
        data_type=data_type,
        sd_path=sd_path,
        ip_adapter_path=ip_adapter_path,
    )

    modality_configs = {
        "all": ["image", "depth", "edge"],
        "image": ["image"],
        "depth": ["depth"],
        "edge": ["edge"],
        "image_depth": ["image", "depth"],
        "image_edge": ["image", "edge"],
        "depth_edge": ["depth", "edge"],
    }
    if modality_mode not in modality_configs:
        raise ValueError(f"Unsupported modality mode: {modality_mode}")

    modalities = modality_configs[modality_mode]
    if modality_scales is None:
        modality_scales = {}
        for mode_key in ["image", "depth", "edge"]:
            if modality_mode == "all":
                modality_scales[mode_key] = IPAdapterGenerator._MODALITY_SCALES[mode_key]
            elif mode_key in modalities:
                modality_scales[mode_key] = {"down": {"block_2": [1.0, 1.0]}, "up": {"block_0": [1.0, 1.0, 1.0]}}
            else:
                modality_scales[mode_key] = {"down": {"block_2": [0.0, 0.0]}, "up": {"block_0": [0.0, 0.0, 0.0]}}

    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_path}")

    exp_dirs = [path for path in base_path.iterdir() if path.is_dir()]
    logger.info("Found %s experiment directories", len(exp_dirs))

    for exp_dir in exp_dirs:
        logger.info("Processing experiment directory: %s", exp_dir.name)
        sub_dirs = [path for path in exp_dir.iterdir() if path.is_dir() and path.name.startswith("sub-")]
        if subjects is not None:
            sub_dirs = [path for path in sub_dirs if any(path.name.startswith(subject) for subject in subjects)]
            logger.info("Found %s matching subject directories after filtering", len(sub_dirs))

        for sub_dir in sub_dirs:
            logger.info("Processing subject directory: %s", sub_dir.name)
            seed = 42
            if "_seed" in sub_dir.name:
                try:
                    seed = int(sub_dir.name.split("_seed")[-1])
                except ValueError:
                    logger.warning("Cannot parse seed value for %s, using default seed %s", sub_dir.name, seed)

            seed_everything(seed)
            generated_datasets_dir = sub_dir / "generated_datasets"
            pt_file_name = "generated_embeddings_original.pt" if use_original_file else "generated_embeddings.pt"
            pt_file_path = generated_datasets_dir / pt_file_name
            if not pt_file_path.exists():
                logger.warning("File not found: %s", pt_file_path)
                continue

            generator = IPAdapterGenerator(
                sd_path=resolved_sd_path,
                ip_adapter_path=resolved_ip_adapter_path,
                device=device,
                seed=seed,
                num_inference_steps=15,
                guidance_scale=0.0,
                modalities=modalities,
                modality_scales=modality_scales,
            )

            modality_dir_name = f"{modality_mode}_before" if use_before_align else modality_mode
            if use_original_file:
                modality_dir_name += "_original_model"
            output_dir = sub_dir / "generated_image" / modality_dir_name
            os.makedirs(output_dir, exist_ok=True)

            logger.info(
                "Generating %s modality images with seed %s to %s",
                modality_dir_name,
                seed,
                output_dir,
            )
            logger.info("Resume functionality: %s", "Enabled" if resume_generation else "Disabled")
            generate_images_from_pt_file(
                generator,
                pt_file_path,
                output_dir,
                modalities,
                use_before_align,
                resume_generation,
            )
            logger.info("Finished processing %s", sub_dir)


def main():
    parser = argparse.ArgumentParser(description="Batch generate images")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to experiment root directory")
    parser.add_argument("--config", type=str, default="configs/cogcappro.yaml", help="Runtime config path")
    parser.add_argument("--data_type", type=str, default="EEG", choices=["EEG", "MEG"], help="Config profile")
    parser.add_argument("--sd_path", type=str, default=None, help="Path to Stable Diffusion model")
    parser.add_argument("--ip_adapter_path", type=str, default=None, help="Path to IP-Adapter")
    parser.add_argument(
        "--modality_mode",
        type=str,
        choices=["all", "image", "depth", "edge", "image_depth", "image_edge", "depth_edge"],
        default="all",
        help="Modality mode to use",
    )
    parser.add_argument("--use_before_align", action="store_true", help="Whether to use unaligned embeddings for generation")
    parser.add_argument(
        "--use_original_file",
        action="store_true",
        help="Whether to use generated_embeddings_original.pt file for generation",
    )
    parser.add_argument("--resume", action="store_true", help="Whether to skip already generated images")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device type",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        help="List of subjects to process, e.g. sub-01 sub-02, if not specified process all subjects",
    )
    args = parser.parse_args()

    logger.info("Starting batch image generation...")
    logger.info("Base directory: %s", args.base_dir)
    logger.info("Modality mode: %s", args.modality_mode)
    logger.info("Use unaligned embeddings: %s", args.use_before_align)
    logger.info("Use original file: %s", args.use_original_file)
    logger.info("Resume: %s", args.resume)
    logger.info("Device: %s", args.device)
    if args.subjects:
        logger.info("Specified subjects: %s", ", ".join(args.subjects))
    else:
        logger.info("Processing all subjects")

    batch_generate_images(
        base_dir=args.base_dir,
        sd_path=args.sd_path,
        ip_adapter_path=args.ip_adapter_path,
        modality_mode=args.modality_mode,
        device=args.device,
        subjects=args.subjects,
        use_before_align=args.use_before_align,
        use_original_file=args.use_original_file,
        resume_generation=args.resume,
        config_path=args.config,
        data_type=args.data_type,
    )


if __name__ == "__main__":
    main()
