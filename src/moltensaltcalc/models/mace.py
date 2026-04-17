from moltensaltcalc.registry import register_model


@register_model(
    "mace",
    metadata={
        "model_path": {
            "type": "str",
            "description": "Path to a MACE model file (e.g. mace_models_custom/mace-mh-0.model). Models can be downloaded from https://github.com/ACEsuit/mace-foundations",
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
            ],
            "description": "Task head used by the model (depend on the provided model file)",
        },
    },
)
def build_mace(params, device):
    from mace.calculators import mace_mp

    return mace_mp(
        model=params.get("model_path", None),
        head=params.get("model_task", None),
        device=device,
    )
