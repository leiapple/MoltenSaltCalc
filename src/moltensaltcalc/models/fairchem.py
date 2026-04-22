from moltensaltcalc.registry import register_model


@register_model(
    "fairchem",
    metadata={
        "model_size": {
            "type": "str",
            "choices": ["small", "medium"],
            "description": "Size of the FairChem model. Size 'medium' is currently only supported for version '1p1'.",
        },
        "model_version": {
            "type": "str",
            "choices": ["1p1", "1p2", "1 (for older version of fairchem-core)"],
            "description": "Version of the pretrained model.",
        },
        "model_task": {
            "type": "str",
            "choices": ["omc", "omol", "odac", "oc20", "omat"],
            "description": "Task the model is trained for.",
        },
        "InferenceSettings": {
            "type": "fairchem.core.units.mlip_unit.api.inference.InferenceSettings",
            "description": "Settings for the inference of the FAIRCHEM model.",
            "default": "Turbo settings from FAIRCHEM with compile=False",
        },
    },
)
def build_fairchem(params, device):
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

    size = params.get("model_size", None)

    # Fairchem resets the random seeds after loading the model, so we need to keep it
    rng_seed_before = int(np.random.get_state()[1][0])
    if size == "small":
        predictor = pretrained_mlip.get_predict_unit(
            f"uma-s-{params.get('model_version', None)}",
            device=device,
            inference_settings=settings,
        )
    elif size == "medium":
        predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device=device)
    else:
        raise ValueError(f"Invalid FAIRCHEM model_size: {size}")
    np.random.seed(rng_seed_before)
    random.seed(rng_seed_before)

    return FAIRChemCalculator(
        predictor,
        task_name=params.get("model_task", None),
    )
