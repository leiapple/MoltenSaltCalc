"""Implementation of the Eqnorm MLIP."""

from moltensaltcalc.registry import register_model


@register_model(
    "eqnorm",
    metadata={
        "model_name": {
            "type": "str",
            "choices": [
                "eqnorm",
            ],
            "description": "Name of pretrained model.",
            "default": "eqnorm",
        },
        "model_task": {
            "type": "str",
            "choices": [
                # "eqnorm-omat",
                "eqnorm-mptrj",
                # "eqnorm-max-mptrj",
            ],
            "description": "Task head used by the model (depend on the provided model file).",
            "default": "eqnorm-mptrj",
        },
    },
)
def _build(params, device):
    """Import and build the Eqnorm MLIP."""
    from eqnorm.calculator import EqnormCalculator

    calc = EqnormCalculator(
        model_name=params.get("model_name", "eqnorm"),
        model_variant=params.get("model_task", "eqnorm-mptrj"),
        device=device,
    )

    return calc
