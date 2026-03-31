from moltensaltcalc.registry import register_model


@register_model(
    "fairchem",
    metadata={
        "model_size": ["small", "medium"],
        "model_version": ["1p1", "1p2"],
        "model_task": ["omc", "omol", "odac", "oc20", "omat"],
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
