"""
ML utilities and helpers.
"""

import torch
from pathlib import Path
from loguru import logger

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def download_model(url: str, save_path: Path) -> bool:
    """Download model weights from URL."""
    import requests

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded model to {save_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

def convert_to_onnx(
    model: torch.nn.Module, save_path: Path, input_shape=(1, 3, 224, 224)
):
    """Convert PyTorch model to ONNX format."""
    model.eval()

    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    logger.info(f"Converted model to ONNX: {save_path}")
