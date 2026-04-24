"""Implementation of the 7net MLIP."""

from moltensaltcalc.registry import register_model


@register_model(
    "7net",
    metadata={
        "model_name": {
            "type": "str",
            "choices": [
                "7net-omni",
                "7net-mf-ompa",
                "7net-omat",
                "7net-l3i5",
                "7net-0",
            ],
            "description": "Name of pretrained models or path to the checkpoint, deployed model or the model itself.",
            "default": "7net-omni",
        },
        "model_task": {
            "type": "str",
            "choices": [
                "mpa",
                "omat24",
                "matpes_pbe",
                "matpes_r2scan",
                "mp_r2scan",
                "oc20",
                "oc22",
                "odac23",
                "omol25_low",
                "omol25_high",
                "spice",
                "qcml",
                "pet_mad",
            ],
            "description": "Task head used by the model (depend on the provided model file).",
            "default": "omat24",
        },
    },
)
def build_(params, device):
    """Import and build the 7net MLIP."""
    from sevenn.calculator import SevenNetCalculator

    calc = SevenNetCalculator(
        model=params.get("model_name", "7net-omni").lower(),
        modal=params.get("model_task", "omat24").lower(),
        device=device,
    )

    return calc
