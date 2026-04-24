"""Implementation of the CHGNet MLIP."""

from moltensaltcalc.registry import register_model


@register_model(
    "chgnet",
    metadata={
        "model_name": {
            "type": "str",
            "choices": ["0.3.0", "0.2.0", "r2scan"],
            "description": "Name of the pretrained CHGNet model. Defaults to '0.3.0'.",
        },
    },
)
def build_(params, device):
    """Import and build the CHGNet MLIP."""
    from chgnet.model.dynamics import CHGNetCalculator
    from chgnet.model.model import CHGNet

    chgnet = CHGNet.load(model_name=params.get("model_name", "0.3.0"), use_device=device)
    calc = CHGNetCalculator(chgnet, use_device=device)

    return calc
