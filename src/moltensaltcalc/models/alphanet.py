"""Implementation of the AlphaNet (alfanet) MLIP."""

from moltensaltcalc.download_model import ensure_model_extension, figshare_download
from moltensaltcalc.registry import register_model

AVAILABLE_MODELS_DICT = {
    "AlphaNet-oma-v1": "53851139",
    "AlphaNet-MPtrj-v1": "53851133",
}


@register_model(
    "alphanet",
    metadata={
        "model_path": {
            "type": "str",
            "choices": list(AVAILABLE_MODELS_DICT.keys()),
            "description": f"A string specifier ({AVAILABLE_MODELS_DICT.keys()}), which leads to model download from figshare or a path to a local checkpoint (.ckpt) file.",
            "default": "AlphaNet-oma-v1",
        },
    },
)
def _build(params, device):
    """Import and build the AlphaNet MLIP."""
    from importlib.resources import files

    from alphanet.config import All_Config
    from alphanet.infer.calc import AlphaNetCalculator

    model_path = params.get("model_path", "AlphaNet-oma-v1")
    if model_path in AVAILABLE_MODELS_DICT:
        # Download the model from figshare
        figshare_id = AVAILABLE_MODELS_DICT[model_path]
        model_path = figshare_download(figshare_id)

    model_path = ensure_model_extension(model_path, extension=".ckpt")

    config_path = files("alphanet").joinpath("pretrained/OMA/oma.json")
    config = All_Config().from_json(str(config_path))
    calc = AlphaNetCalculator(str(model_path), device=device, config=config)

    return calc
