"""Fast UF3 backend for 2-body models (drop-in replacement for ``fit_uf3``).

Same ``model.json`` and same physics as ``fit_uf3`` (verified to ~1e-12 in energy, forces
and stress), but evaluated with a cell-list neighbour list plus the model's own pair
B-splines instead of uf3's generic dense routines:

* forces: analytic, from the first spline derivative (no dense (N, 3, n_pairs) arrays),
* stress: analytic virial (``fit_uf3`` uses a 12-evaluation finite-difference stress),
* ONE neighbour list per step, shared by the UF3 pair terms and the ZBL wall,
* uses ``matscipy``'s C neighbour list if installed (much faster), else ASE's.

Only 2-body models are supported (``degree == 2``); for 3-body models use ``fit_uf3``.
The optional ZBL repulsive wall is the same ``ZBLRepulsion`` as in ``fit_uf3``.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.symbols import symbols2numbers

from moltensaltcalc.models.fit_uf3 import ZBLRepulsion
from moltensaltcalc.registry import register_model

try:  # optional: C implementation, several times faster than ASE's numpy one
    from matscipy.neighbours import neighbour_list as _neighbour_list
except ImportError:  # pragma: no cover
    from ase.neighborlist import neighbor_list as _neighbour_list


class UF3FastCalculator(Calculator):
    """Neighbour-list evaluation of a 2-body UF3 model (+ optional ZBL wall).

    Follows uf3's convention E = sum over directed (i, j) pairs of phi(r_ij), i.e. every
    unordered pair is counted twice (the fitted splines absorb this factor). The ZBL
    wall, if given, uses the usual physical convention (each pair once), as in ``fit_uf3``.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, uf_calculator, wall: ZBLRepulsion | None = None, **kwargs):
        super().__init__(**kwargs)
        if uf_calculator.degree != 2:
            raise ValueError(
                f"fit_uf3_fast supports only 2-body UF3 models (degree={uf_calculator.degree}); use 'fit_uf3'."
            )
        self._uf = uf_calculator
        self._wall = wall
        self.elements = [str(e) for e in uf_calculator.element_list]
        self._pairs = []  # (Z_a, Z_b, r_min, r_max, spline)
        for pair in uf_calculator.interactions_map[2]:
            za, zb = symbols2numbers(pair)
            self._pairs.append(
                (
                    int(za),
                    int(zb),
                    max(float(uf_calculator.r_min_map[pair]), 0.0),
                    float(uf_calculator.r_max_map[pair]),
                    uf_calculator.pair_potentials[pair],
                )
            )
        self._rcut = max(p[3] for p in self._pairs)
        if wall is not None:
            self._rcut = max(self._rcut, wall.r_out)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        n = len(atoms)
        Z = atoms.numbers
        # full (directed) list: every pair appears twice, as (i, j) and (j, i); D = r_j - r_i
        i, j, d, D = _neighbour_list("ijdD", atoms, self._rcut)
        Zi, Zj = Z[i], Z[j]

        # ---- UF3 pair terms ----
        energy = float(self._uf.energy_1b(atoms))
        dedd = np.zeros(len(d))  # dE/dd, per directed pair (already including the factor 2 below)
        for za, zb, r_min, r_max, spline in self._pairs:
            if za == zb:
                sel = (Zi == za) & (Zj == za)
            else:
                sel = ((Zi == za) & (Zj == zb)) | ((Zi == zb) & (Zj == za))
            sel &= (d > r_min) & (d < r_max)
            if not sel.any():
                continue
            ds = d[sel]
            energy += float(np.sum(spline(ds)))  # uf3 convention: E = sum over DIRECTED pairs
            dedd[sel] += 2.0 * spline(ds, nus=1)  # each pair enters twice in E

        # ---- optional ZBL wall (physical convention: 1/2 sum over the directed list) ----
        if self._wall is not None:
            w = d < self._wall.r_out
            if w.any():
                e_z, de_z = self._wall._pair(d[w], Z[i][w].astype(float), Z[j][w].astype(float))  # noqa: SLF001
                energy += 0.5 * float(e_z.sum())
                dedd[w] += de_z  # (1/2 * 2)

        # F_i = + sum_j dE/dd * D/d
        fpair = (dedd / np.where(d > 0, d, 1.0))[:, None] * D
        forces = np.zeros((n, 3))
        np.add.at(forces, i, fpair)
        # stress is cheap here (analytic virial), so always provide it: avoids a 2nd neighbour
        # list when MD asks for forces and stress separately.
        virial = 0.5 * np.einsum("ki,kj->ij", D, fpair) / atoms.get_volume()

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = (0.5 * (virial + virial.T)).flat[[0, 4, 8, 5, 2, 1]]


@register_model(
    "fit_uf3_fast",
    metadata={
        "model_path": {
            "type": "str",
            "description": "Path to a fitted 2-body UF3 model JSON (WeightedLinearModel.to_json). Required.",
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
    """Load a fitted 2-body UF3 model.json and return a fast ASE calculator (CPU only)."""
    from uf3.forcefield.calculator import UFCalculator
    from uf3.regression.least_squares import WeightedLinearModel

    model_path = params.get("model_path")
    if model_path is None:
        raise ValueError("The 'fit_uf3_fast' model requires a 'model_path' pointing to a fitted model.json.")

    wall = None
    if params.get("repulsion", True):
        wall = ZBLRepulsion(
            r_in=params.get("repulsion_r_in", 4.0),
            r_out=params.get("repulsion_r_out", 5.0),
        )
    return UF3FastCalculator(UFCalculator(WeightedLinearModel.from_json(model_path)), wall=wall)
