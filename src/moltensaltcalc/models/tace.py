"""Implementation of the TACE MLIP."""

from moltensaltcalc.registry import register_model


@register_model(
    "tace",
    metadata={
        "model_name": {
            "type": "str",
            "choices": [
                "TACE-v1-OMat24-M",
                "TACE-v1-OAM-M",
                "TACE-v1-LES-REICO-5-PdAgCHO.pt",
            ],
            "description": "Name of pre-trained model or path to the checkpoint, deployed model or the model itself.",
            "default": "TACE-v1-OMat24-M",
        },
    },
)
def _build(params, device):
    """Import and build the TACE MLIP."""
    from tace.foundations import tace_foundations
    from tace.interface.ase import TACEAseCalc

    model_name = params.get("model_name", "TACE-v1-OMat24-M")
    model = tace_foundations[model_name]
    calc = TACEAseCalc(model=model, dtype="float32", device=device)

    return calc
