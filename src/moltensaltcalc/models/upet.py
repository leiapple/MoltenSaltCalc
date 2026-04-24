from moltensaltcalc.registry import register_model


@register_model(
    "upet",
    metadata={
        "model_task": {
            "type": "str",
            "choices": ["omat", "oam", "mad", "omatpes", "omad", "spice"],
            "description": "Task the model is trained for.",
            "default": "omat",
        },
        "model_size": {
            "type": "str",
            "choices": ["xs", "s", "m", "l", "xl"],
            "description": "Size of the UPET model. Availability depends on the task selected. The following combinations are available: omat: xs, s, m, l, xl; oam: l, xl; mad: xs, s; omatpes: l; omad: xs, s, l; spice: s, l.",
            "default": "s",
        },
        "model_version": {
            "type": "str",
            "choices": ["latest", "v0.1.0", "v0.2.0"],
            "description": "Version of the pretrained UPET model. Defaults to 'latest'.",
            "default": "latest",
        },
        "checkpoint_path": {
            "type": "str",
            "description": "Path to a pretrained UPET model checkpoint file. Optional.",
            "default": None,
        },
    },
)
def build_(params, device):
    from upet.calculator import UPETCalculator

    model_name = f"pet-{params.get('model_task', 'omat').lower()}-{params.get('model_size', 's').lower()}"
    calc = UPETCalculator(
        model=model_name,
        version=params.get("model_version", "latest"),
        checkpoint_path=params.get("checkpoint_path", None),
        device=device,
    )

    return calc
