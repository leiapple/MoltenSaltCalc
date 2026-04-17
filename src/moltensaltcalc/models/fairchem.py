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
            "choices": ["1p1", "1p2"],
            "description": "Version of the pretrained model.",
        },
        "model_task": {
            "type": "str",
            "choices": ["omc", "omol", "odac", "oc20", "omat"],
            "description": "Task the model is trained for.",
        },
    },
)
def build_fairchem(params, device):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    size = params.get("model_size", None)

    if size == "small":
        predictor = pretrained_mlip.get_predict_unit(
            f"uma-s-{params.get('model_version', None)}", device=device
        )
    elif size == "medium":
        predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device=device)
    else:
        raise ValueError(f"Invalid FAIRCHEM model_size: {size}")

    return FAIRChemCalculator(
        predictor,
        task_name=params.get("model_task", None),
    )
