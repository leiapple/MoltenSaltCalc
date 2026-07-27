"""Implementation of the orbitals MLIP."""

from moltensaltcalc.registry import register_model


@register_model(
    "orbitals",
    metadata={
        "model_task": {
            "type": "str",
            "choices": ["mpa", "omat"],
            "description": "Task the model is trained for.",
            "default": "omat",
        },
        "max_neighbors": {
            "type": "str",
            "choices": ["20", "inf"],
            "description": "Maximum number of neighbors.",
            "default": "20",
        },
        "model_type": {
            "type": "str",
            "choices": ["direct", "conservative"],
            "description": "How inference of the forces is achieved.",
            "default": "conservative",
        },
        "model_version": {
            "type": "str",
            "choices": ["v2", "v3"],
            "description": "Version of the model.",
            "default": "v3",
        },
        "orb_v2_task": {
            "type": "str",
            "choices": ["", "_mptraj_only", "_d3", "_d3_sm", "_d3_xs"],
            "description": "Task the model is trained for, only applied if model_version is v2.",
            "default": "",
        },
    },
)
def _build(params, device):
    """Import and build the orbitals MLIP."""
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    task = params.get("model_task", "omat").lower()
    max_neigh = params.get("max_neighbors", "inf")
    model_type = params.get("model_type", "conservative")
    model_name = f"orb_v3_{model_type}_{max_neigh}_{task}"
    if params.get("model_version", "v3") == "v2":
        orb_v2_task = params.get("orb_v2_task", "")
        model_name = f"orb{orb_v2_task}_v2"
    try:
        model_builder = getattr(pretrained, model_name)
    except AttributeError as e:
        raise ValueError(
            f"Unsupported ORB model '{model_name}'. Available models can be found in orb_models.forcefield.pretrained or at https://github.com/orbital-materials/orb-models/blob/main/MODELS.md."
        ) from e

    orbff, atoms_adapter = model_builder(
        device=device,
        precision="float32-high",
    )
    return ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=device)
