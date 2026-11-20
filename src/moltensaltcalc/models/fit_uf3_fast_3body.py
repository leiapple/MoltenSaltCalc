"""Fast UF3 backend for 2+3-body models (drop-in replacement for fit_uf3 at degree 3).

Reuses uf3's fitted pair and trio B-splines, but evaluates them with a single
neighbour list + an analytic virial instead of uf3's np.unique triplet enumeration
and 12-evaluation numerical stress. ~500x faster than the old 'fit_uf3' backend for
3-body MD; energy/forces/stress validated against it (forces are energy-consistent).
"""
# Numeric kernel: physics names (Z, Dca, DcA, ...) and vectorised methods with many
# locals are intentional, so the usual style/complexity checks are relaxed here.
# pylint: disable=invalid-name,too-many-locals,too-many-statements,too-many-instance-attributes
# pylint: disable=too-few-public-methods,protected-access,consider-using-enumerate,ungrouped-imports
# pylint: disable=missing-function-docstring,unbalanced-tuple-unpacking,dangerous-default-value

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.data import chemical_symbols
from ase.symbols import symbols2numbers
from moltensaltcalc.models.fit_uf3 import ZBLRepulsion
from moltensaltcalc.registry import register_model

try:
    from matscipy.neighbours import neighbour_list as _neighbour_list
except ImportError:
    from ase.neighborlist import neighbor_list as _neighbour_list


class UF3Fast3BodyCalculator(Calculator):
    """Neighbour-list evaluation of a 2+3-body UF3 model (+ optional ZBL wall)."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, uf_calculator, wall: ZBLRepulsion | None = None, **kwargs):
        super().__init__(**kwargs)
        if uf_calculator.degree != 3:
            raise ValueError(
                f"fit_uf3_fast_3body supports only 3-body models (degree={uf_calculator.degree}); "
                "use fit_uf3_fast for 2-body."
            )
        self._uf = uf_calculator
        self._wall = wall
        self.elements = [str(e) for e in uf_calculator.element_list]
        from uf3.data import composition

        # --- 2-body: (Z_a, Z_b, r_min, r_max, pair_spline) per pair ---
        self._pairs = []
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

        # --- 3-body: keyed by (Z_vertex, Z_ligandA, Z_ligandB) -> knot ranges + trio_spline ---
        self._trios = {}
        zset = set()
        for trio in uf_calculator.bspline_config.interactions_map[3]:
            zi, za, zb = (int(z) for z in symbols2numbers(trio))
            km = uf_calculator.bspline_config.knots_map[trio]
            self._trios[(zi, za, zb)] = (
                float(km[0][0]),
                float(km[0][-1]),
                float(km[1][0]),
                float(km[1][-1]),
                float(km[2][0]),
                float(km[2][-1]),
                uf_calculator.trio_potentials[trio],
            )
            zset.update((zi, za, zb))
        self._refX = {z: composition.reference_X[chemical_symbols[z]] for z in zset}

        self._rcut2 = max((p[3] for p in self._pairs), default=0.0)
        self._rcut3 = max((t[1] for t in self._trios.values()), default=0.0)
        self._rcut = max(self._rcut2, self._rcut3)
        if wall is not None:
            self._rcut = max(self._rcut, wall.r_out)

    def _pair_terms(self, i, j, d, Z):
        """2-body energy and per-directed-pair dE/dd.

        uf3's convention: E = sum over DIRECTED pairs (every unordered pair appears
        twice, as (i,j) and (j,i)), and the fitted spline absorbs that. So the energy
        sums spline(d) over the directed list, and each pair's dE/dd carries a factor 2.
        """
        energy = float(self._uf.energy_1b(self.atoms))
        dedd = np.zeros(len(d))
        Zi, Zj = Z[i], Z[j]
        for za, zb, r_min, r_max, spline in self._pairs:
            if za == zb:  # noqa: SIM108
                sel = (Zi == za) & (Zj == za)
            else:
                sel = ((Zi == za) & (Zj == zb)) | ((Zi == zb) & (Zj == za))
            sel &= (d > r_min) & (d < r_max)
            if not sel.any():
                continue
            ds = d[sel]
            energy += float(np.sum(spline(ds)))
            dedd[sel] += 2.0 * spline(ds, nus=1)
        return energy, dedd

    def _triplets(self, i, j, d, D):
        """Every vertex's unordered ligand pairs, once each (no np.unique sort).

        Returns per-triplet arrays: central index c, the two ligand real indices a/b,
        and the two vertex->ligand vectors Dca, Dcb. Ligands are NOT yet ordered by
        electronegativity (that is done in _three_body).
        """
        sel = d <= self._rcut3
        ii, jj, Dij = i[sel], j[sel], D[sel]
        if len(ii) == 0:
            return None
        order = np.argsort(ii, kind="stable")
        ii, jj, Dij = ii[order], jj[order], Dij[order]
        uniq, start = np.unique(ii, return_index=True)
        counts = np.diff(np.append(start, len(ii)))

        c_idx, a_idx, b_idx, Dca_l, Dcb_l = [], [], [], [], []
        for g in range(len(uniq)):
            n = counts[g]
            if n < 2:
                continue
            s = start[g]
            a, b = np.triu_indices(n, k=1)
            c_idx.append(np.full(len(a), uniq[g]))
            a_idx.append(jj[s + a])
            b_idx.append(jj[s + b])
            Dca_l.append(Dij[s + a])
            Dcb_l.append(Dij[s + b])
        if not c_idx:
            return None
        return (
            np.concatenate(c_idx),
            np.concatenate(a_idx),
            np.concatenate(b_idx),
            np.concatenate(Dca_l),
            np.concatenate(Dcb_l),
        )

    def _three_body(self, i, j, d, D, Z, n_atoms, volume):
        energy = 0.0
        forces = np.zeros((n_atoms, 3))
        virial = np.zeros((3, 3))
        trip = self._triplets(i, j, d, D)
        if trip is None:
            return energy, forces, virial
        c, a, b, Dca, Dcb = trip

        # order the two ligands by electronegativity proxy: A = lower, B = higher
        xa = np.array([self._refX[z] for z in Z[a]])
        xb = np.array([self._refX[z] for z in Z[b]])
        swap = xb < xa
        A, B = a.copy(), b.copy()
        A[swap], B[swap] = b[swap], a[swap]
        DcA, DcB = Dca.copy(), Dcb.copy()
        DcA[swap], DcB[swap] = Dcb[swap], Dca[swap]

        rl = np.linalg.norm(DcA, axis=1)  # vertex - A
        rm = np.linalg.norm(DcB, axis=1)  # vertex - B
        DAB = DcB - DcA  # A -> B
        rn = np.linalg.norm(DAB, axis=1)  # A - B
        Zc, ZA, ZB = Z[c], Z[A], Z[B]

        for (zi, za, zb), (rl_lo, rl_hi, rm_lo, rm_hi, rn_lo, rn_hi, spline) in self._trios.items():
            m = (Zc == zi) & (za == ZA) & (zb == ZB)
            m &= (rl > rl_lo) & (rl < rl_hi) & (rm > rm_lo) & (rm < rm_hi) & (rn > rn_lo) & (rn < rn_hi)
            if not m.any():
                continue
            pts = np.vstack([rl[m], rm[m], rn[m]]).T
            energy += float(np.sum(spline(pts)))
            vl = spline(pts, nus=np.array([1, 0, 0]))
            vm = spline(pts, nus=np.array([0, 1, 0]))
            vn = spline(pts, nus=np.array([0, 0, 1]))

            cm, Am, Bm = c[m], A[m], B[m]
            ul = DcA[m] / rl[m][:, None]
            um = DcB[m] / rm[m][:, None]
            un = DAB[m] / rn[m][:, None]
            fl = vl[:, None] * ul
            fm = vm[:, None] * um
            fn = vn[:, None] * un

            np.add.at(forces, cm, fl + fm)
            np.add.at(forces, Am, -fl + fn)
            np.add.at(forces, Bm, -fm - fn)

            virial += np.einsum("k,ki,kj->ij", vl / rl[m], DcA[m], DcA[m])
            virial += np.einsum("k,ki,kj->ij", vm / rm[m], DcB[m], DcB[m])
            virial += np.einsum("k,ki,kj->ij", vn / rn[m], DAB[m], DAB[m])

        return energy, forces, virial / volume

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        n = len(atoms)
        Z = atoms.numbers
        vol = atoms.get_volume()
        i, j, d, D = _neighbour_list("ijdD", atoms, self._rcut)

        energy, dedd = self._pair_terms(i, j, d, Z)
        fpair = (dedd / np.where(d > 0, d, 1.0))[:, None] * D
        forces = np.zeros((n, 3))
        np.add.at(forces, i, fpair)
        virial2 = 0.5 * np.einsum("ki,kj->ij", D, fpair)

        e3, f3, stress3 = self._three_body(i, j, d, D, Z, n, vol)
        energy += e3
        forces += f3

        if self._wall is not None:
            w = d < self._wall.r_out
            if w.any():
                e_z, de_z = self._wall._pair(d[w], Z[i][w].astype(float), Z[j][w].astype(float))
                energy += 0.5 * float(e_z.sum())
                fz = (de_z / d[w])[:, None] * D[w]
                np.add.at(forces, i[w], fz)
                virial2 += 0.5 * np.einsum("ki,kj->ij", D[w], fz)

        stress2 = virial2 / vol
        stress = stress2 + stress3
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = (0.5 * (stress + stress.T)).flat[[0, 4, 8, 5, 2, 1]]


@register_model(
    "fit_uf3_fast_3body",
    metadata={
        "model_path": {
            "type": "str",
            "description": "Path to a fitted 2+3-body UF3 model JSON (WeightedLinearModel.to_json). Required.",
            "default": None,
        },
        "repulsion": {
            "type": "bool",
            "description": "Add the ZBL baseline repulsion (must match what the fit subtracted). Default True.",
            "default": True,
        },
        "repulsion_r_in": {
            "type": "float",
            "description": "ZBL full below this radius (A). Default 4.0.",
            "default": 4.0,
        },
        "repulsion_r_out": {
            "type": "float",
            "description": "ZBL switched off above this radius (A). Default 5.0.",
            "default": 5.0,
        },
    },
)
def _build(params, device=None):  # pylint: disable=unused-argument
    """Load a fitted 2+3-body UF3 model.json and return a fast ASE calculator (CPU only)."""
    from uf3.forcefield.calculator import UFCalculator
    from uf3.regression.least_squares import WeightedLinearModel

    model_path = params.get("model_path")
    if model_path is None:
        raise ValueError("The 'fit_uf3_fast_3body' model requires a 'model_path' pointing to a fitted model.json.")

    wall = None
    if params.get("repulsion", True):
        wall = ZBLRepulsion(
            r_in=params.get("repulsion_r_in", 4.0),
            r_out=params.get("repulsion_r_out", 5.0),
        )
    return UF3Fast3BodyCalculator(UFCalculator(WeightedLinearModel.from_json(model_path)), wall=wall)
