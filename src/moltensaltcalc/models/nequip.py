"""Implementation of the NequIP MLIP."""

from moltensaltcalc.registry import register_model


@register_model(
    "nequip",
    metadata={
        "model_path": {
            "type": "str",
            "description": "Path to a precompiled NequIP model file (e.g. nequip_models/mir-group__NequIP-OAM-S__0.1.nequip.pth). The filename must end with '.nequip.pth'. A description how to compile the model can be found at https://www.nequip.net/models. Required.",
            "default": None,
        },
    },
)
def _build(params, device):
    """Import and build the NequIP MLIP."""
    from nequip.integrations.ase import NequIPCalculator

    calc = NequIPCalculator.from_compiled_model(
        params.get("model_path", None),
        chemical_species_to_atom_type_map=True,
        device=device,
    )

    return calc
