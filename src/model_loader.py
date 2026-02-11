"""Shared model loading logic for UNet and DiffusionModel."""

import json
import torch
from src.model import Unet
from src.model_diffusion import DiffusionModel


def load_model(model_type, checkpoint_path, config_path, device="cuda"):
    """
    Load a UNet or DiffusionModel from a checkpoint.

    Args:
        model_type: "UNet" or "Diffusion"
        checkpoint_path: Path to the .pth checkpoint file
        config_path: Path to the JSON config file (for model_params)
        device: Device to load model onto

    Returns:
        Loaded model in eval mode
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    model_params = config["model_params"]

    if model_type == "Diffusion":
        model_config = dict(model_params)
        model_config["checkpoint"] = checkpoint_path
        model_config["load_betas"] = True
        model = DiffusionModel(**model_config).to(device)
    elif model_type == "UNet":
        model = Unet(
            dim=model_params.get("dim", 64),
            channels=model_params.get("channels", 2),
            dim_mults=tuple(model_params.get("dim_mults", [1, 1, 1])),
            use_convnext=True,
            convnext_mult=model_params.get("convnext_mult", 1),
            with_time_emb=model_params.get("with_time_emb", False),
            padding_mode=model_params.get("padding_mode", "circular"),
        )
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt)
        model = model.to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type!r}. Use 'UNet' or 'Diffusion'.")

    return model.eval()
