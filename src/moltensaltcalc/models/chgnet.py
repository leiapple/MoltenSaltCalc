"""Implementation of the CHGNet MLIP."""

import numpy as np

from moltensaltcalc.registry import register_model


class CHGNetVoigtWrapper:
    """Wrapper for CHGNet calculator to convert stress to Voigt format."""

    def __init__(self, calc):
        """Initialize the wrapper."""
        self.calc = calc

    def calculate(self, atoms=None, properties=None, system_changes=None):
        """Modify the calculator to convert stress to Voigt format."""
        self.calc.calculate(atoms, properties, system_changes)
        if "stress" in self.calc.results:
            stress = self.calc.results["stress"]
            # CHGNet gives (3,3) → convert to Voigt (6,)
            stress_voigt = np.array(
                [
                    stress[0, 0],
                    stress[1, 1],
                    stress[2, 2],
                    stress[0, 1],
                    stress[1, 2],
                    stress[0, 2],
                ]
            )
            self.calc.results["stress"] = stress_voigt

    def __getattr__(self, name):
        """Forward all other calls to the calculator."""
        return getattr(self.calc, name)


@register_model(
    "chgnet",
    metadata={
        "model_name": {
            "type": "str",
            "choices": ["0.3.0", "0.2.0", "r2scan"],
            "description": "Name of the pretrained CHGNet model.",
            "default:": "0.3.0",
        },
    },
)
def _build(params, device):
    """Import and build the CHGNet MLIP."""
    from chgnet.model.dynamics import CHGNetCalculator
    from chgnet.model.model import CHGNet

    chgnet = CHGNet.load(model_name=params.get("model_name", "0.3.0"), use_device=device)
    calc = CHGNetCalculator(chgnet, use_device=device)
    calc = CHGNetVoigtWrapper(calc)

    return calc
