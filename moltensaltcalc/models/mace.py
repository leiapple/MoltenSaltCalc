from moltensaltcalc.registry import register_model


@register_model(
    "mace",
    metadata={
        "model_path": "e.g. mace_models_custom/mace-mh-0.model, which can be downloaded from https://github.com/ACEsuit/mace-foundations",
        "task_name": [
            "omat_pbe",
            "omol",
            "spice_wB97M",
            "rgd1_b3lyp",
            "oc20_usemppbe",
            "matpes_r2scan",
        ],
    },
)
def build_mace(params, device):
    from mace.calculators import mace_mp

    return mace_mp(
        model=params.get("model_path", None),
        head=params.get("model_task", None),
        device=device,
    )
