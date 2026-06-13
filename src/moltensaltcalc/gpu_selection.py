"""Select the GPU that is supposed to be used by making only that one visible with the environment variable CUDA_VISIBLE_DEVICES."""

import os
import warnings


def select_device(device: str) -> str:
    """Select the GPU device to use by only making the correct one visible."""

    if device.startswith("cuda"):
        changed_device = False
        # Normalize device BEFORE CUDA init
        if ":" in device:
            idx = device.split(":", 1)[1]
            if idx.isdigit():
                os.environ["CUDA_VISIBLE_DEVICES"] = idx
                changed_device = True
            else:
                print(f"Invalid CUDA device index: {idx}, falling back to 'cuda'")
        device = "cuda"

        import torch

        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU.", stacklevel=2)
            device = "cpu"

        if changed_device:
            print("\n\nCHANGED DEVICE:\n")
            print("Visible device count:", torch.cuda.device_count())
            print("Current device index:", torch.cuda.current_device())
            print("Current device name:", torch.cuda.get_device_name(0))
            print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
            print("\n\n")

    return device
