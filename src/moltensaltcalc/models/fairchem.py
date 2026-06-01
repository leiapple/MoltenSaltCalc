"""Implementation of the FAIRCHEM MLIP."""

from moltensaltcalc.registry import register_model


@register_model(
    "fairchem",
    metadata={
        "model_size": {
            "type": "str",
            "choices": ["s", "m"],
            "description": "Size of the FairChem model. Size 'm' is currently (04.2026) only supported for version '1p1'.",
            "default": "s",
        },
        "model_version": {
            "type": "str",
            "choices": ["1p1", "1p2", "1 (for older versions of fairchem-core)"],
            "description": "Version of the pretrained model.",
            "default": "1p2",
        },
        "model_task": {
            "type": "str",
            "choices": ["omc", "omol", "odac", "oc20", "omat"],
            "description": "Task the model is trained for.",
            "default": "omat",
        },
        "InferenceSettings": {
            "type": "fairchem.core.units.mlip_unit.api.inference.InferenceSettings",
            "description": "Settings for the inference of the FAIRCHEM model.",
            "default": "Turbo settings from FAIRCHEM with compile=False",
        },
    },
)
def _build(params, device):
    """Import and build the FAIRCHEM MLIP."""
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

    import random

    import numpy as np
    import torch
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    # Turbo settings but without compile, so it works when the compiler is not available
    turbo_settings = InferenceSettings(
        tf32=True,
        activation_checkpointing=False,
        merge_mole=True,
        compile=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )
    settings = params.get("InferenceSettings", turbo_settings)

    # Fairchem resets the random seeds after loading the model, so we need to keep it
    rng_seed_before = int(np.random.get_state()[1][0])  # type: ignore
    predictor = pretrained_mlip.get_predict_unit(
        f"uma-{params.get('model_size', 's').lower()}-{params.get('model_version', '1p2').lower()}",
        device=device,
        inference_settings=settings,
    )
    np.random.seed(rng_seed_before)
    random.seed(rng_seed_before)
    print(f"Changed device: {changed_device}")
    if changed_device:
        print("Visible device count:", torch.cuda.device_count())
        print("Current device index:", torch.cuda.current_device())
        print("Current device name:", torch.cuda.get_device_name(0))
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    return FAIRChemCalculator(
        predictor,
        task_name=params.get("model_task", "omat").lower(),
    )
