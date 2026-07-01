"""Implementation of the matgl MLIPs."""

from moltensaltcalc.registry import register_model

# matgl.get_available_pretrained_models()
AVAILABLE_MODELS = [
    "CHGNet-PES-MatPES-PBE-2025.2.10",
    "CHGNet-PES-MatPES-r2SCAN-2025.2.10",
    "M3GNet-Eform-MP-2018.6.1",
    "M3GNet-PES-ANI-1x-Subset",
    "M3GNet-PES-MatPES-PBE-2025.2",
    "M3GNet-PES-MatPES-r2SCAN-2025.2",
    "MEGNet-BandGap-mfi-MP-2019.4.1",
    "MEGNet-Eform-MP-2018.6.1",
    "QET-PES-MatPES-PBE-2025.2",
    "QET-PES-MatPES-r2SCAN-2025.2",
    "QET-PES-MatQ",
    "SO3Net-PES-ANI-1x-Subset",
    "TensorNet-PES-ANI-1x-Subset",
    "TensorNet-PES-MatPES-PBE-2025.2",
    "TensorNet-PES-MatPES-PBE-2025.2-m",
    "TensorNet-PES-MatPES-r2SCAN-2025.2",
    "TensorNet-PES-MatPES-r2SCAN-2025.2-m",
]


@register_model(
    "matgl",
    metadata={
        "model_name": {
            "type": "str",
            "choices": AVAILABLE_MODELS,
            "description": "The name of the model to use.",
            "default": "QET-PES-MatQ",
        },
    },
)
def _build(params, device):
    """Import and build the matgl MLIPs."""
    from matgl import load_model
    from matgl.ext.ase import PESCalculator

    model_name = params.get("model_name", "QET-PES-MatQ")

    pot = load_model(model_name)

    return PESCalculator(pot, use_voigt=True, stress_unit="eV/A3", device=device)
