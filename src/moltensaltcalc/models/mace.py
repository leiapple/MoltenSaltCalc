"""Implementation of the MACE MLIP."""

from moltensaltcalc.registry import register_model


@register_model(
    "mace",
    metadata={
        "model_path": {
            "type": "str",
            "choices": ["small", "medium", "large", "..."],
            "description": "Path to a local file path (e.g. mace_models_custom/mace-mh-0.model), a URL (e.g. https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model?raw=true) or a string specifier ('small', 'medium' or 'large'). Models can be downloaded from https://github.com/ACEsuit/mace-foundations. If either of 'small', 'medium' or 'large' are provided, the respective model is downloaded from figshare.",
            "default": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model?raw=true",
        },
        "model_task": {
            "type": "str",
            "choices": [
                "omat_pbe",
                "omol",
                "spice_wB97M",
                "rgd1_b3lyp",
                "oc20_usemppbe",
                "matpes_r2scan",
                "default",
            ],
            "description": "Task head used by the model (depend on the provided model file, see https://github.com/acesuit/mace#pretrained-foundation-models).",
            "default": "omat_pbe",
        },
    },
)
def _build(params, device):
    """Import and build the MACE MLIP."""
    from mace.calculators import mace_mp

    return mace_mp(
        model=params.get(
            "model_path",
            "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model?raw=true",
        ),
        head=params.get("model_task", "default"),
        device=device,
    )
