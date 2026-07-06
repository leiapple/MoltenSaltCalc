"""Implementation of the EquFlashV2 (fairchem) MLIP."""

from moltensaltcalc.download_model import figshare_download
from moltensaltcalc.registry import register_model

AVAILABLE_MODELS_DICT = {
    "equflash": "65435004",
    "equflash-omat": "66275099",
    "equflash_v2": "65435007",
    "equflash_v2-omat": "66275102",
}


@register_model(
    "equflash",
    metadata={
        "model_path": {
            "type": "str",
            "choices": list(AVAILABLE_MODELS_DICT.keys()),
            "description": f"A string specifier ({AVAILABLE_MODELS_DICT.keys()}), which leads to model download from figshare or a path to a local checkpoint (.pt) file.",
            "default": "equflash_v2",
        },
    },
)
def _build(params, device):
    """Import and build the EquFlashV2 MLIP."""
    import numpy as np
    from GGNN.common.calculator import UCalculator

    # Fairchem resets the rng seeds when loading the model, so we need to keep it
    rng_seed_before = int(np.random.get_state()[1][0])  # type: ignore
    model_path = params.get("model_path", "equflash_v2")
    if model_path in AVAILABLE_MODELS_DICT:
        # Download the model from figshare
        figshare_id = AVAILABLE_MODELS_DICT[model_path]
        model_path = figshare_download(figshare_id)

    calc = UCalculator(checkpoint_path=str(model_path), seed=rng_seed_before, cpu=device.startswith("cpu"))

    return calc
