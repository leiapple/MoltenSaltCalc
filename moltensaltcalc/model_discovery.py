import pkgutil

import moltensaltcalc.models as models_pkg


def discover_models() -> list[str]:
    """Finds all available models in the models package.

    Returns:
        list[str]: List of available model names
    """

    models = []

    for _, module_name, _ in pkgutil.iter_modules(models_pkg.__path__):
        models.append(module_name)

    return sorted(models)
