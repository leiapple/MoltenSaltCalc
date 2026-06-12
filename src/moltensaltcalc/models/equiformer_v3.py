"""Implementation of the equiformer_v3 (fairchem) MLIP."""

from moltensaltcalc.registry import register_model

AVAILABLE_MODELS = [
    "omat24_direct",
    "omat24_gradient",
    "omat24-mptrj-salex_gradient",
    "mptrj_gradient",
]
MODEL_HF_ID = "mirror-physics/equiformer_v3"


def clean_model(model_path: str):
    """Cleans the model dict by removing the "_orig_mod.module." from the keys if present and writes the cleaned model to the huggingface cache."""
    import torch  # Somehow doesn't get it from the _build function

    checkpoint = torch.load(model_path, map_location="cpu")
    old_state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in old_state_dict.items():
        new_key = key
        if new_key.startswith("_orig_mod.module."):
            new_key = new_key.replace("_orig_mod.module.", "")
        elif new_key.startswith("module."):
            new_key = new_key.replace("module.", "")
        elif new_key.startswith("_orig_mod."):
            new_key = new_key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value
    checkpoint["state_dict"] = new_state_dict
    torch.save(checkpoint, model_path)  # Overwrite the model in the huggingface cache


@register_model(
    "equiformer_v3",
    metadata={
        "model_path": {
            "type": "str",
            "choices": ["omat24_direct", "omat24_gradient", "omat24-mptrj-salex_gradient", "mptrj_gradient", "..."],
            "description": f"Path to a local file path (e.g. 'models/omat24_direct.pt') or a string specifier ({AVAILABLE_MODELS}). Models can be downloaded from https://huggingface.co/{MODEL_HF_ID}/tree/main/checkpoint, which is also where the string specifiers query to download the model.",
            "default": "omat24_direct",
        },
        "model_revision": {
            "type": "str",
            "description": "Revision of the model to download from HuggingFace.",
            "default": "main",
        },
        "dont_clean_model": {
            "type": "bool",
            "description": "Whether to skip cleaning the model by stripping the '_orig_mod.module.' from the keys.",
            "default": False,
        },
    },
)
def _build(params, device):
    """Import and build the EQUIFORMER_V3 MLIP."""
    import os

    # Normalize device BEFORE CUDA init
    changed_device = False
    if device.startswith("cuda"):
        if ":" in device:
            idx = device.split(":", 1)[1]
            if idx.isdigit():
                os.environ["CUDA_VISIBLE_DEVICES"] = idx
                changed_device = True
            else:
                print(f"Invalid CUDA device index: {idx}, falling back to 'cuda'")
        device = "cuda"

    import numpy as np
    import torch
    from equiformer_v3.core import OCPCalculator
    from huggingface_hub import hf_hub_download

    # Fairchem resets the random seeds when loading the model, so we need to keep it
    rng_seed_before = int(np.random.get_state()[1][0])  # type: ignore

    # Get the pre-trained model from HuggingFace
    model_path = params.get("model_path", "omat24_direct")
    rev = params.get("model_revision", "main")
    if model_path in AVAILABLE_MODELS:
        model_path = hf_hub_download(repo_id=MODEL_HF_ID, filename=f"checkpoint/{model_path}.pt", revision=rev)
    if not params.get("dont_clean_model", False):
        clean_model(model_path)

    calc = OCPCalculator(
        checkpoint_path=model_path,
        cpu=device.startswith("cpu"),
        seed=rng_seed_before,
    )
    print(f"Changed device: {changed_device}")
    if changed_device:
        print("Visible device count:", torch.cuda.device_count())
        print("Current device index:", torch.cuda.current_device())
        print("Current device name:", torch.cuda.get_device_name(0))
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    return calc
