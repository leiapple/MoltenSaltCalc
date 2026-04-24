from moltensaltcalc.registry import register_model


@register_model(
    "nequix",
    metadata={
        "model_task": {
            "type": "str",
            "choices": ["mp", "omat", "oam"],
            "description": "Task the model is trained for.",
            "default": "omat",
        },
        "model_path": {
            "type": "str",
            "description": "Provide the path to a Nequix model file. Optional, overrides the model_task.",
        },
        "model_backend": {
            "type": "str",
            "choices": ["torch", "jax"],
            "description": "Backend to use for the Nequix model.",
            "default": "jax",
        },
    },
)
def build_(params, device):
    from nequix.calculator import NequixCalculator

    model_name = params.get("model_task", "omat")
    if model_name is not None:
        model_name = f"nequix-{model_name}-1"
    calc = NequixCalculator(
        model_name,
        model_path=params.get("model_path", None),
        backend=params.get("model_backend", "jax"),
        use_kernel=False,
        use_compile=True,
        device=device,
    )

    return calc
