"""Implementation of the FAIRCHEM MLIP."""

from moltensaltcalc.registry import register_model

AVAILABLE_MODELS = [
    "uma-s-1p2",
    "uma-s-1p1",
    "uma-m-1p1",
    "esen-md-direct-all-omol",
    "esen-sm-conserving-all-omol",
    "esen-sm-direct-all-omol",
    "allscaip-md-conserving-all-omol",
    "allscaip-md-direct-all-omol",
    "esen-sm-conserving-all-oc25",
    "esen-md-direct-all-oc25",
    "esen-sm-filtered-odac25",
    "esen-sm-full-odac25",
]


@register_model(
    "fairchem",
    metadata={
        "model_name": {
            "type": "str",
            "choices": AVAILABLE_MODELS,
            "description": "Version of the pretrained model.",
            "default": "uma-s-1p2",
        },
        "model_task": {
            "type": "str",
            "choices": ["omc", "omol", "odac", "oc20", "omat", ""],
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

    import random

    import numpy as np
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

    # Fairchem resets the rng seeds after loading the model, so we need to keep it
    rng_seed_before = int(np.random.get_state()[1][0])  # type: ignore
    predictor = pretrained_mlip.get_predict_unit(
        params.get("model_name", "uma-s-1p2").lower(),
        device=device,
        inference_settings=settings,
    )
    np.random.seed(rng_seed_before)
    random.seed(rng_seed_before)
    task_name = params.get("model_task", "omat").lower()
    return FAIRChemCalculator(
        predictor,
        task_name=task_name if task_name != "" else None,
    )
