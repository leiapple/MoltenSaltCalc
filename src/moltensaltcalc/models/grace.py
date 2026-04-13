from moltensaltcalc.registry import register_model


@register_model(
    "grace",
    metadata={
        "model_size": ["small", "medium", "large"],
        "num_layers": [1, 2],
        "model_task": ["OAM", "OMAT"],
    },
)
def build_grace(params, device):
    from tensorpotential.calculator.foundation_models import (
        GRACEModels,
        grace_fm,
    )

    size = params.get("model_size", None)
    layers = params.get("num_layers", None)
    task = params.get("model_task", None)

    mapping = {
        ("OMAT", "small", 1): GRACEModels.GRACE_1L_OMAT,
        ("OMAT", "small", 2): GRACEModels.GRACE_2L_OMAT,
        ("OMAT", "medium", 1): GRACEModels.GRACE_1L_OMAT_medium_base,
        ("OMAT", "medium", 2): GRACEModels.GRACE_2L_OMAT_medium_base,
        ("OMAT", "large", 1): GRACEModels.GRACE_1L_OMAT_large_base,
        ("OMAT", "large", 2): GRACEModels.GRACE_2L_OMAT_large_base,
        ("OAM", "small", 1): GRACEModels.GRACE_1L_OAM,
        ("OAM", "small", 2): GRACEModels.GRACE_2L_OAM,
        ("OAM", "medium", 1): GRACEModels.GRACE_1L_OMAT_medium_ft_AM,
        ("OAM", "medium", 2): GRACEModels.GRACE_2L_OMAT_medium_ft_AM,
        ("OAM", "large", 1): GRACEModels.GRACE_1L_OMAT_large_ft_AM,
        ("OAM", "large", 2): GRACEModels.GRACE_2L_OMAT_large_ft_AM,
    }

    model = mapping[(task, size, layers)]

    return grace_fm(model)
