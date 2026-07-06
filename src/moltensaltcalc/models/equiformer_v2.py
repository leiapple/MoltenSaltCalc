"""Implementation of the equiformer_v2 (fairchem) MLIP."""

from moltensaltcalc.registry import register_model

AVAILABLE_MODELS = [
    "eqV2_153M_omat",
    "eqV2_153M_omat_mp_salex",
    "eqV2_31M_mp",
    "eqV2_31M_omat",
    "eqV2_31M_omat_mp_salex",
    "eqV2_86M_omat",
    "eqV2_86M_omat_mp_salex",
    "eqV2_dens_153M_mp",
    "eqV2_dens_31M_mp",
    "eqV2_dens_86M_mp",
]
MODEL_HF_ID = "facebook/OMAT24"


@register_model(
    "equiformer_v2",
    metadata={
        "model_path": {
            "type": "str",
            "choices": AVAILABLE_MODELS,
            "description": f"Path to a local file path (e.g. 'models/eqV2_31M_omat.pt') or a string specifier ({AVAILABLE_MODELS}). Models can be downloaded from https://huggingface.co/{MODEL_HF_ID}/tree/main/checkpoint, which is also where the string specifiers query to download the model.",
            "default": "eqV2_31M_omat",
        },
        "model_revision": {
            "type": "str",
            "description": "Revision of the model to download from HuggingFace.",
            "default": "main",
        },
    },
)
def _build(params, device):
    """Import and build the EQUIFORMER_V2 MLIP."""
    import numpy as np
    from equiformer_v2.core import OCPCalculator
    from huggingface_hub import hf_hub_download

    # Fairchem resets the rng seeds when loading the model, so we need to keep it
    rng_seed_before = int(np.random.get_state()[1][0])  # type: ignore

    # Get the pre-trained model from HuggingFace
    model_path = params.get("model_path", "eqV2_31M_omat")
    rev = params.get("model_revision", "main")
    if model_path in AVAILABLE_MODELS:
        model_path = hf_hub_download(repo_id=MODEL_HF_ID, filename=f"{model_path}.pt", revision=rev)

    calc = OCPCalculator(
        checkpoint_path=model_path,
        cpu=device.startswith("cpu"),
        seed=rng_seed_before,
    )

    return calc
