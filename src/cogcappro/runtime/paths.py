"""Public-safe config loading and runtime path resolution."""

from __future__ import annotations

from argparse import Namespace
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]


LOCAL_CONFIG_PATH = REPO_ROOT / "configs" / "local.yaml"
ENV_TO_PATH_KEY = {
    "COGCAPPRO_DATA_ROOT": "data_root",
    "COGCAPPRO_WEIGHTS_ROOT": "weights_root",
    "COGCAPPRO_RUNS_ROOT": "runs_root",
}
DEFAULT_CONFIGS = {
    "EEG": REPO_ROOT / "configs" / "cogcappro.yaml",
    "MEG": REPO_ROOT / "configs" / "cogcappro_meg.yaml",
}
PRETRAIN_MAP = {
    "RN50": {"pretrained": "openai", "resize": (224, 224), "z_dim": 1024},
    "RN101": {"pretrained": "openai", "resize": (224, 224), "z_dim": 512},
    "ViT-B-16": {"pretrained": "laion2b_s34b_b88k", "resize": (224, 224), "z_dim": 512},
    "ViT-B-32": {"pretrained": "laion2b_s34b_b79k", "resize": (224, 224), "z_dim": 512},
    "ViT-L-14": {"pretrained": "laion2b_s32b_b82k", "resize": (224, 224), "z_dim": 768},
    "ViT-H-14": {"pretrained": "laion2b_s32b_b79k", "resize": (224, 224), "z_dim": 1024},
    "ViT-g-14": {"pretrained": "laion2b_s34b_b88k", "resize": (224, 224), "z_dim": 1024},
    "ViT-bigG-14": {"pretrained": "laion2b_s39b_b160k", "resize": (224, 224), "z_dim": 1280},
}
DATA_CONFIG = {
    "MEG": {
        "c_num_all": 271,
        "timesteps": [0, 201],
        "region_channels": {
            "frontal": 67,
            "temporal": 68,
            "central": 52,
            "parietal": 45,
            "occipital": 39,
            "self": 0,
        },
    },
    "EEG": {
        "c_num_all": 63,
        "timesteps": [0, 250],
        "region_channels": {
            "frontal": 15,
            "temporal": 10,
            "central": 21,
            "parietal": 9,
            "occipital": 8,
            "self": 17,
        },
    },
}
DEFAULT_CLIP_WEIGHTS_REL = {
    "RN50": "CLIPRN50/RN50.pt",
    "ViT-H-14": "Things_dataset/model_pretrained/clip/open_clip_pytorch_model.bin",
}
DEFAULT_DIFFUSION_EMBEDDINGS_REL = "diffusion_embeddings"
DEFAULT_SDXL_REL = "Things_dataset/model_pretrained/sdxl-turbo"
DEFAULT_IP_ADAPTER_REL = "Things_dataset/model_pretrained/ip_adapter"
DEFAULT_MEG_IMAGE_DESCRIPTION_REL = "THINGS-MEG/Image_text_description"


def load_runtime_config(config_path: str) -> Any:
    config = OmegaConf.load(config_path)
    if LOCAL_CONFIG_PATH.exists():
        config = OmegaConf.merge(config, OmegaConf.load(LOCAL_CONFIG_PATH))
    return config


def load_public_runtime_config(data_type: str = "EEG") -> Any:
    config_path = DEFAULT_CONFIGS[data_type.upper()]
    return load_runtime_config(str(config_path))


def ensure_paths_section(config: Any) -> Any:
    if "paths" not in config or config.paths is None:
        config.paths = OmegaConf.create({})
    return config


def _path_from_value(value: str | os.PathLike | None, base_root: str | Path | None = None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if base_root is not None:
        return Path(base_root).expanduser() / path
    return REPO_ROOT / path


def resolve_path_like(value: str | os.PathLike | None, base_root: str | Path | None = None) -> Path | None:
    return _path_from_value(value, base_root)


def _first_existing_path(candidates: list[Path | None], label: str, required: bool = True) -> Path | None:
    seen = []
    for candidate in candidates:
        if candidate is None:
            continue
        seen.append(str(candidate))
        if candidate.exists():
            return candidate
    if required:
        joined = ", ".join(seen) if seen else "<none>"
        raise FileNotFoundError(f"Unable to resolve {label}. Checked: {joined}")
    return None


def apply_path_overrides(config: Any, args: Namespace) -> Any:
    ensure_paths_section(config)
    for env_name, key in ENV_TO_PATH_KEY.items():
        env_value = os.getenv(env_name)
        if env_value:
            config.paths[key] = env_value

    cli_overrides = {
        "data_root": getattr(args, "data_root", None),
        "weights_root": getattr(args, "weights_root", None),
        "runs_root": getattr(args, "runs_root", None),
    }
    for key, value in cli_overrides.items():
        if value:
            config.paths[key] = value
    return config


def _require_path(config: Any, key: str, message: str) -> Path:
    ensure_paths_section(config)
    value = config.paths.get(key)
    if not value:
        raise ValueError(message)
    return Path(value).expanduser()


def optional_root(config: Any, key: str) -> Path | None:
    ensure_paths_section(config)
    value = config.paths.get(key)
    if not value:
        return None
    return Path(value).expanduser()


def _clip_weights_rel(config: Any) -> dict[str, str]:
    ensure_paths_section(config)
    rel = config.paths.get("clip_weights_rel")
    if rel:
        return dict(rel)
    return dict(DEFAULT_CLIP_WEIGHTS_REL)


def resolve_clip_weight_paths(config: Any) -> dict[str, str]:
    weights_root = _require_path(
        config,
        "weights_root",
        "Missing weights root. Set configs/local.yaml or COGCAPPRO_WEIGHTS_ROOT.",
    )
    resolved = {}
    for name, rel in _clip_weights_rel(config).items():
        rel_path = Path(rel)
        resolved[name] = str(rel_path if rel_path.is_absolute() else weights_root / rel_path)
    config.paths.clip_weights = OmegaConf.create(resolved)
    return resolved


def resolve_diffusion_embeddings_root(config: Any, required: bool = True) -> Path | None:
    ensure_paths_section(config)
    weights_root = optional_root(config, "weights_root")
    data_root = optional_root(config, "data_root")
    configured = _path_from_value(config.paths.get("diffusion_embeddings_root"))
    configured_rel = _path_from_value(
        config.paths.get("diffusion_embeddings_rel", DEFAULT_DIFFUSION_EMBEDDINGS_REL),
        weights_root,
    ) if weights_root else None
    candidates = [
        configured,
        configured_rel,
        REPO_ROOT / "weights" / "diffusion_embeddings",
        data_root / "ThingsEEG" / "Image_feature_new" / "data_features" if data_root else None,
    ]
    resolved = _first_existing_path(candidates, "diffusion embeddings root", required=required)
    if resolved is not None:
        config.paths.diffusion_embeddings_root = str(resolved)
    return resolved


def resolve_sdxl_root(config: Any, override: str | None = None, required: bool = True) -> Path | None:
    ensure_paths_section(config)
    weights_root = optional_root(config, "weights_root")
    configured = _path_from_value(override) or _path_from_value(config.paths.get("sdxl_root"))
    default_path = _path_from_value(config.paths.get("sdxl_rel", DEFAULT_SDXL_REL), weights_root) if weights_root else None
    resolved = _first_existing_path([configured, default_path], "SDXL model root", required=required)
    if resolved is not None:
        config.paths.sdxl_root = str(resolved)
    return resolved


def resolve_ip_adapter_root(config: Any, override: str | None = None, required: bool = True) -> Path | None:
    ensure_paths_section(config)
    weights_root = optional_root(config, "weights_root")
    configured = _path_from_value(override) or _path_from_value(config.paths.get("ip_adapter_root"))
    default_path = _path_from_value(config.paths.get("ip_adapter_rel", DEFAULT_IP_ADAPTER_REL), weights_root) if weights_root else None
    resolved = _first_existing_path([configured, default_path], "IP-Adapter root", required=required)
    if resolved is not None:
        config.paths.ip_adapter_root = str(resolved)
    return resolved


def resolve_image_description_root(config: Any, data_type: str, required: bool = False) -> Path | None:
    ensure_paths_section(config)
    data_type = data_type.upper()
    data_root = optional_root(config, "data_root")
    if data_type == "MEG":
        configured = _path_from_value(config.paths.get("things_meg_image_description_root"))
        rel_default = config.paths.get("things_meg_image_description_rel", DEFAULT_MEG_IMAGE_DESCRIPTION_REL)
        fallback = _path_from_value(rel_default, data_root) if data_root else None
        candidates = [
            configured,
            fallback,
            REPO_ROOT / "weights" / "texts" / "meg",
        ]
        resolved = _first_existing_path(candidates, "MEG image description root", required=required)
        if resolved is not None:
            config.paths.things_meg_image_description_root = str(resolved)
        return resolved

    candidates = [REPO_ROOT / "weights" / "texts" / "eeg"]
    return _first_existing_path(candidates, "EEG image description root", required=required)


def _data_rel_key(data_type: str) -> str:
    if data_type.upper() == "MEG":
        return "things_meg_rel"
    return "things_eeg_rel"


def resolve_base_data_dir(config: Any, data_type: str) -> str:
    data_root = _require_path(
        config,
        "data_root",
        "Missing data root. Set configs/local.yaml or COGCAPPRO_DATA_ROOT.",
    )
    rel_key = _data_rel_key(data_type)
    rel_value = config.paths.get(rel_key)
    if not rel_value:
        raise ValueError(f"Missing paths.{rel_key} in config.")

    rel_path = Path(rel_value)
    full_path = rel_path if rel_path.is_absolute() else data_root / rel_path
    return str(full_path)


def resolve_dataset_root(config: Any, data_type: str) -> Path:
    return Path(resolve_base_data_dir(config, data_type)).parent


def resolve_filtered_data_dir(base_data_dir: str, data_type: str, filter_band: str | None) -> str:
    if not filter_band:
        return base_data_dir

    base_path = Path(base_data_dir)
    suffix = base_path.name
    if data_type.upper() == "MEG":
        return str(base_path.with_name(f"{suffix}_{filter_band}"))
    return str(base_path.with_name(f"{suffix}_{filter_band}"))


def resolve_runs_dir(config: Any, args: Namespace) -> str:
    cli_save_dir = getattr(args, "save_dir", None)
    if cli_save_dir:
        return cli_save_dir

    ensure_paths_section(config)
    runs_root = config.paths.get("runs_root")
    if runs_root:
        return str(Path(runs_root).expanduser())

    yaml_save_dir = config.get("save_dir")
    if yaml_save_dir:
        return yaml_save_dir

    return "runs"


def finalize_runtime_paths(config: Any, args: Namespace) -> Any:
    ensure_paths_section(config)
    data_type = getattr(args, "data_type", None) or config.get("data_type", "EEG")

    resolve_clip_weight_paths(config)
    base_data_dir = resolve_base_data_dir(config, data_type)
    config.data.data_dir = resolve_filtered_data_dir(base_data_dir, data_type, getattr(args, "filter_band", None))
    config.save_dir = resolve_runs_dir(config, args)
    return config


def dump_resolved_config(config: Any) -> str:
    return OmegaConf.to_yaml(config, resolve=True)


def apply_cli_arg_overrides(config: Any, args: Namespace) -> Any:
    for key in config.keys():
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key, value in vars(args).items():
        config[key] = value
    return config


def prepare_runtime_config(args: Namespace, config: Any) -> Any:
    config = apply_cli_arg_overrides(config, args)
    config = apply_path_overrides(config, args)
    config["data"]["subjects"] = [args.subjects]
    config["train"]["loss_type"] = args.loss_type

    if args.devices is not None:
        config["devices"] = [int(x) for x in args.devices.split(",")]
    elif "devices" not in config or config["devices"] is None:
        config["devices"] = [0]

    current_data_config = DATA_CONFIG[args.data_type]
    if args.selected_region:
        config["data"]["selected_ch"] = args.selected_region
        c_num = current_data_config["region_channels"].get(args.selected_region, current_data_config["c_num_all"])
    else:
        config["data"]["selected_ch"] = False
        c_num = current_data_config["c_num_all"]

    config["models"]["brain"]["params"]["c_num"] = c_num
    config["models"]["brain"]["params"]["data_type"] = args.data_type.lower()
    config["timesteps"] = current_data_config["timesteps"]
    config["data"]["timesteps"] = current_data_config["timesteps"]
    config["data"]["uncertainty_aware"] = args.uncertainty_aware
    config["train"]["mask_count"] = args.mask_count

    if not args.uncertainty_aware:
        config["data"]["blur_type"] = {"target": "cogcappro.models.inpainting_data.DirectT", "params": {}}

    config["z_dim"] = PRETRAIN_MAP[args.vision_backbone]["z_dim"]
    config["data"]["model_type"] = args.vision_backbone
    config = finalize_runtime_paths(config, args)

    num_devices = len(config["devices"])
    config["data"]["per_gpu_train_batch_size"] = config["data"]["train_batch_size"] // num_devices
    config["data"]["per_gpu_test_batch_size"] = config["data"]["test_batch_size"] // num_devices
    config["data"]["per_gpu_val_batch_size"] = config["data"]["val_batch_size"] // num_devices
    return config


def merge_with_public_config(config: Any, data_type: str = "EEG") -> Any:
    public_config = load_public_runtime_config(data_type)
    return OmegaConf.merge(public_config, config)
