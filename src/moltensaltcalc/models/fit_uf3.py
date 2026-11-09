"""Implementation of the UF3 (ultra-fast force field) MLIP, with an optional
short-range ZBL repulsive wall.

A UF3 model is fitted offline with fit_uf3.py and serialized to a model.json via
``WeightedLinearModel.to_json``. The 2-body B-splines are fit on DFT liquid
configurations and have NO physical repulsion below the sampled range (~2 A): the
splines flatten, so under NPT the barostat can collapse the box. To fix that we
splice a ZBL screened-nuclear-repulsion wall onto the potential, smoothly switched
off above ``repulsion_r_out`` so it only guards the unphysical short-range region
and never touches the fitted structure. UF3 and ZBL are combined with ASE's
SumCalculator (the same way dispersion is added at the simulator level).
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.neighborlist import neighbor_list

from moltensaltcalc.registry import register_model

# ZBL universal screening function constants
_KE = 14.399645  # e^2/(4 pi eps0) in eV*Angstrom
_C = np.array([0.18175, 0.50986, 0.28022, 0.02817])
_D = np.array([3.19980, 0.94229, 0.40290, 0.20162])


class UF3ASECalculator(Calculator):
    """ASE-compliant wrapper around uf3's ``UFCalculator`` (energy/forces/stress)."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, uf_calculator, **kwargs):
        super().__init__(**kwargs)
        self._uf = uf_calculator
        self.elements = [str(k) for k in uf_calculator.solutions if not isinstance(k, tuple)]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy = float(self._uf.get_potential_energy(self.atoms))
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = self._uf.get_forces(self.atoms)
        # UFCalculator.get_stress is numerical -> it re-evaluates energy through
        # atoms.get_potential_energy(), so the atoms it is given must carry a calculator.
        stress_atoms = self.atoms.copy()
        stress_atoms.calc = self._uf
        self.results["stress"] = self._uf.get_stress(stress_atoms)


class ZBLRepulsion(Calculator):
    """Short-range ZBL repulsive wall, smoothly switched off between r_in and r_out.

    Full ZBL screened-nuclear repulsion below r_in, zero above r_out, a smooth
    (C2) switch between. Parameter-free apart from the two cutoffs: the strength
    comes from the atomic numbers. Provides energy, forces and stress (the stress
    is what lets the wall push back against a collapsing NPT box).
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, r_in=4.0, r_out=5.0, **kwargs):
        super().__init__(**kwargs)
        self.r_in = float(r_in)
        self.r_out = float(r_out)

    def _pair(self, d, z1, z2):
        """Return (energy, dE/dd) per pair, vectorized over the pair arrays."""
        a = 0.46850 / (z1**0.23 + z2**0.23)
        x = (d / a)[:, None]
        ex = np.exp(-_D[None, :] * x)
        phi = np.sum(_C[None, :] * ex, axis=1)
        dphi = np.sum(-_C[None, :] * _D[None, :] * ex, axis=1) / a  # d(phi)/dd
        pref = _KE * z1 * z2
        e_zbl = pref / d * phi
        de_zbl = -pref / d**2 * phi + pref / d * dphi
        # smooth switch S: 1 at r_in -> 0 at r_out
        t = np.clip((d - self.r_in) / (self.r_out - self.r_in), 0.0, 1.0)
        s = 1.0 - (6 * t**5 - 15 * t**4 + 10 * t**3)
        ds = -(30 * t**4 - 60 * t**3 + 30 * t**2) / (self.r_out - self.r_in)
        ds = np.where((d > self.r_in) & (d < self.r_out), ds, 0.0)
        e = e_zbl * s
        de = de_zbl * s + e_zbl * ds
        return e, de

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(self.atoms)
        energy = 0.0
        forces = np.zeros((n, 3))
        stress = np.zeros((3, 3))
        i, j, d, D = neighbor_list("ijdD", self.atoms, self.r_out)
        if len(d) > 0:
            Z = self.atoms.numbers
            e, de = self._pair(d, Z[i].astype(float), Z[j].astype(float))
            energy = 0.5 * e.sum()  # each unordered pair counted twice
            fi = (de / d)[:, None] * D  # force on atom i[k] from directed pair k
            np.add.at(forces, i, fi)
            stress = 0.5 * np.einsum("ki,kj->ij", D, fi) / self.atoms.get_volume()
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = 0.5 * (stress + stress.T).flat[[0, 4, 8, 5, 2, 1]]


@register_model(
    "fit_uf3",
    metadata={
        "model_path": {
            "type": "str",
            "description": (
                "Path to a fitted UF3 model JSON (WeightedLinearModel.to_json), e.g. "
                "results/2body_ClCsZr_n200_c8_w0.3_seed42_927efc/model.json. Required."
            ),
            "default": None,
        },
        "repulsion": {
            "type": "bool",
            "description": "Add the ZBL baseline repulsion (must match what the fit subtracted). Default True.",
            "default": True,
        },
        "repulsion_r_in": {
            "type": "float",
            "description": "ZBL full below this radius (A). Default 4.0 (baseline form).",
            "default": 4.0,
        },
        "repulsion_r_out": {
            "type": "float",
            "description": "ZBL switched off above this radius (A), where it is already negligible. Default 5.0.",
            "default": 5.0,
        },
    },
)
def _build(params, device=None):  # pylint: disable=unused-argument
    """Load a fitted UF3 model.json and return an ASE calculator.

    UF3 is CPU-only, so ``device`` is accepted but ignored. No dispersion is added
    here (the DFT training data already includes D3; pass ``dispersion=None``). A
    short-range ZBL repulsive wall is added by default so NPT MD does not collapse.
    """
    from uf3.forcefield.calculator import UFCalculator
    from uf3.regression.least_squares import WeightedLinearModel

    model_path = params.get("model_path")
    if model_path is None:
        raise ValueError(
            "The 'fit_uf3' model requires a 'model_path' pointing to a fitted "
            "model.json produced by fit_uf3.py."
        )

    uf = UF3ASECalculator(UFCalculator(WeightedLinearModel.from_json(model_path)))

    if not params.get("repulsion", True):
        return uf

    wall = ZBLRepulsion(
        r_in=params.get("repulsion_r_in", 4.0),
        r_out=params.get("repulsion_r_out", 5.0),
    )
    combined = SumCalculator([uf, wall])
    combined.elements = uf.elements  # expose for composition-from-model logic
    return combined