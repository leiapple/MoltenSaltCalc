from moltensaltcalc.registry import MODEL_METADATA


def format_unknown_model_error(model_name: str, discoverable_models: list[str]) -> str:
    """Returns a formatted error message for an unknown model.

    Args:
        model_name (str): Name of the model (input) that was not found.
        discoverable_models (list[str]): List of available model names.

    Returns:
        str: Formatted error message.
    """
    if discoverable_models:
        models_str = "\n- '" + "'\n- '".join(discoverable_models) + "'"
    else:
        models_str = "\n  (none found)"

    return f"Unknown model '{model_name}'.\n\n\nAvailable models:{models_str}"


def format_model_error(model_name: str, params: dict, error: Exception) -> str:
    """Returns a formatted error message for a model initialization error.

    Args:
        model_name (str): Name of the model (input) that failed to initialize.
        params (dict): Model parameters that were passed to the model.
        error (Exception): Original error that occurred during initialization.

    Returns:
        str: Formatted error message.
    """
    metadata = MODEL_METADATA.get(model_name, {})

    msg = f"""
Model initialization failed.

Model: {model_name}
Parameters: {params}

Error type: {type(error).__name__}
Error message: {error}
""".strip()

    if metadata:
        msg += f"\n\nKnown parameter options for {model_name} (if applicable):\n"
        for key, values in metadata.items():
            msg += f"  - {key}: {values}\n"

    return msg
