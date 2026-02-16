"""Shared model loading logic for UNet and DiffusionModel."""

import json
import torch
from src.model import Unet
from src.model_diffusion import DiffusionModel


def load_model(model_type, checkpoint_path, config_path=None, model_params=None, device="cuda"):                                                                      
    """        
    Load a UNet or DiffusionModel from a checkpoint.

    Args:
        model_type: "UNet" or "Diffusion"
        checkpoint_path: Path to the .pth checkpoint file
        config_path: (Optional) Path to the JSON config file (for model_params). Deprecated, use model_params instead.
        model_params: (Optional) Dict or DictConfig with model parameters. If provided, takes precedence over config_path.
        device: Device to load model onto

    Returns:
        Loaded model in eval mode
    """
    if model_params is None:
        if config_path is None:
            raise ValueError("Either config_path or model_params must be provided")
        with open(config_path, "r") as f:
            config = json.load(f)
        model_params = config["model_params"]

    # Convert OmegaConf to dict if necessary
    if hasattr(model_params, 'to_container'):
        model_params = model_params.to_container()

    # Keys that shouldn't be passed to model constructors
    NON_MODEL_KEYS = {"type"}

    if model_type == "Diffusion":
        model_config = {k: v for k, v in model_params.items() if k not in NON_MODEL_KEYS}
        model = DiffusionModel(**model_config).to(device)
    elif model_type == "UNet":
        model_kwargs = {k: v for k, v in model_params.items() if k not in NON_MODEL_KEYS}
        model = Unet(use_convnext=True, **model_kwargs).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt)
    else:
        raise ValueError(f"Unknown model type: {model_type!r}. Use 'UNet' or 'Diffusion'.")

    return model.eval()
