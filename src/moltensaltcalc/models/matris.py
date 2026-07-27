"""Implementation of the MatRIS MLIP."""

from moltensaltcalc.registry import register_model


@register_model(
    "matris",
    metadata={
        "model_name": {
            "type": "str",
            "choices": ["matris_10m_oam", "matris_10m_mp"],
            "description": "The name of the model to use.",
            "default": "matris_10m_oam",
        },
    },
)
def _build(params, device):
    """Import and build the MatRIS MLIP."""
    from matris.applications.base import MatRISCalculator

    model_name = params.get("model_name", "matris_10m_oam")

    calc = MatRISCalculator(model=model_name, task="efs", device=device)

    return calc
