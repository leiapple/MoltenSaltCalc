"""Implementation of the GRACE MLIP."""

from moltensaltcalc.registry import register_model


@register_model(
    "grace",
    metadata={
        "model_size": {
            "type": "str",
            "choices": ["small", "medium", "large"],
            "description": "Size of the GRACE model.",
            "default": "small",
        },
        "num_layers": {
            "type": "int",
            "choices": [1, 2],
            "description": "Number of message-passing layers.",
            "default": 1,
        },
        "model_task": {
            "type": "str",
            "choices": ["oam", "omat"],
            "description": "Task the model is trained for.",
            "default": "omat",
        },
    },
)
def build_grace(params, device):
    """Import and build the GRACE MLIP."""
    from tensorpotential.calculator.foundation_models import (
        GRACEModels,
        grace_fm,
    )

    size = params.get("model_size", "small").lower()
    layers = params.get("num_layers", 1)
    task = params.get("model_task", "omat").lower()

    mapping = {
        ("omat", "small", 1): GRACEModels.GRACE_1L_OMAT,  # type: ignore
        ("omat", "small", 2): GRACEModels.GRACE_2L_OMAT,  # type: ignore
        ("omat", "medium", 1): GRACEModels.GRACE_1L_OMAT_medium_base,  # type: ignore
        ("omat", "medium", 2): GRACEModels.GRACE_2L_OMAT_medium_base,  # type: ignore
        ("omat", "large", 1): GRACEModels.GRACE_1L_OMAT_large_base,  # type: ignore
        ("omat", "large", 2): GRACEModels.GRACE_2L_OMAT_large_base,  # type: ignore
        ("oam", "small", 1): GRACEModels.GRACE_1L_OAM,  # type: ignore
        ("oam", "small", 2): GRACEModels.GRACE_2L_OAM,  # type: ignore
        ("oam", "medium", 1): GRACEModels.GRACE_1L_OMAT_medium_ft_AM,  # type: ignore
        ("oam", "medium", 2): GRACEModels.GRACE_2L_OMAT_medium_ft_AM,  # type: ignore
        ("oam", "large", 1): GRACEModels.GRACE_1L_OMAT_large_ft_AM,  # type: ignore
        ("oam", "large", 2): GRACEModels.GRACE_2L_OMAT_large_ft_AM,  # type: ignore
    }
    try:
        model = mapping[(task, size, layers)]
    except KeyError as e:
        raise ValueError(f"Unknown model parameters: {params}. Known parameters: {list(mapping.keys())}") from e

    return grace_fm(model, device=device)
