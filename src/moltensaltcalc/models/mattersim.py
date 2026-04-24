"""Implementation of the MatterSim MLIP."""

from moltensaltcalc.registry import register_model


@register_model(
    "mattersim",
    metadata={
        "model_path": {
            "type": "str",
            "description": "Path to a pytorch model file, e.g. 'MatterSim-v1.0.0-5M.pth' that can be downloaded from https://github.com/microsoft/mattersim. Optional.",
            "default": None,
        }
    },
)
def build_(params, device):
    """Import and build the MatterSim MLIP."""
    from mattersim.forcefield import MatterSimCalculator

    calc = MatterSimCalculator(load_path=params.get("model_path", None), device=device)

    return calc
