"""Implementation of the HIENet MLIP."""

from moltensaltcalc.registry import register_model


@register_model(
    "hienet",
    metadata={
        "model_path": {
            "type": "str",
            "description": "Path to a checkpoint file. For 'HIENet-0', the pre-trained model from https://github.com/daniisler/AIRS/tree/feature-python-package/OpenMat/HIENet/hienet/checkpoints will be used.",
            "default": "HIENet-0",
        }
    },
)
def _build(params, device):
    """Import and build the HIENet MLIP."""
    from hienet.hienet_calculator import HIENetCalculator

    calc = HIENetCalculator(model=params.get("model_path", "HIENet-0"), device=device)

    return calc
