"""Implementation of the CHGNet MLIP."""

from ase.stress import full_3x3_to_voigt_6_stress

from moltensaltcalc.registry import register_model


class CHGNetVoigtWrapper:
    """Wrapper for CHGNet calculator to convert stress to Voigt format. CHGNet gives (3,3) => convert to Voigt (6,)."""

    def __init__(self, calc):
        """Initialize the wrapper."""
        self.calc = calc

    def calculate(self, atoms=None, properties=None, system_changes=None):
        """Modify the calculator to convert stress to Voigt format."""
        self.calc.calculate(atoms, properties, system_changes)
        if "stress" in self.calc.results:
            self.calc.results["stress"] = full_3x3_to_voigt_6_stress(self.calc.results["stress"])

    def get_property(self, name, atoms=None, allow_calculation=True):
        """Modify the calculator to convert stress to Voigt format."""
        result = self.calc.get_property(name, atoms, allow_calculation)
        if name == "stress" and result is not None:
            result = full_3x3_to_voigt_6_stress(result)
        return result

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
    calc = CHGNetCalculator(chgnet, use_device=device, stress_weight=1.0)  # The analyzer expects stress in eV/Å³
    calc = CHGNetVoigtWrapper(calc)

    return calc
