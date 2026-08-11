"""Implementation of the TACE MLIP."""

from moltensaltcalc.registry import register_model


@register_model(
    "tace",
    metadata={
        "model_name": {
            "type": "str",
            "choices": [
                "TACE-OAM-7M",
                "TACE-OAM-L",
                "TACE-Omat24-7M",
                "TACE-OMat24-L",
                "TACE-OMat24-RRA-1.0",
                "TACE-OMat24-RRA-Preview",
                "TECE-OAM-RRA-1.0",
                "TECE-OMat24-RRA-1.0"
            ],
            "description": "Name of pre-trained model or path to the checkpoint, deployed model or the model itself.",
            "default": "TACE-Omat24-7M",
        },
    },
)
def _build(params, device):
    """Import and build the TACE MLIP."""
    from tace.foundations import tace_foundations
    from tace.interface.ase import TACEAseCalc

    model_name = params.get("model_name", "TACE-Omat24-7M")
    model = tace_foundations[model_name]
    calc = TACEAseCalc(model=model, dtype="float32", device=device)

    return calc
