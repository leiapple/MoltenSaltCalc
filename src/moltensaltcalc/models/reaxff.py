"""Implementation of the ReaxFF calculator."""

import warnings

from moltensaltcalc.registry import register_model


@register_model(
    "reaxff",
    metadata={
        "ffield_reax_path": {
            "type": "str",
            "description": (
                "Path to a local ReaxFF force-field file, e.g. 'ffield.reax.082.CHOCsKNaClIFLi', which can be downloaded from https://github.com/by-student-2017/lammps_education_reaxff_win.git."
            ),
            "default": None,
        },
        "elements": {
            "type": "list",
            "description": (
                "Chemical elements corresponding to the LAMMPS atom types. The order must match the pair_coeff command."
            ),
            "default": None,
        },
    },
)
def _build(params, device):
    """Import and build the ReaxFF calculator."""
    from ase.calculators.lammpslib import LAMMPSlib

    if device is not None:
        warnings.warn(
            "ReaxFF does not support specifying a device. The device depends on your LAMMPS installations.",
            stacklevel=2,
        )

    ffield_reax_path = params.get(
        "ffield_reax_path",
        None,
    )
    elements = params.get("elements", None)

    atom_types = {element: i + 1 for i, element in enumerate(elements)}

    lmpcmds = [
        "pair_style reaxff NULL",
        f"pair_coeff * * {ffield_reax_path} {' '.join(elements)}",
        "fix qeq all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff",
    ]

    return LAMMPSlib(
        lmpcmds=lmpcmds,
        atom_types=atom_types,
        lammps_header=[
            "units metal",
            "atom_style charge",
            "atom_modify map array sort 0 0",
        ],
        keep_alive=True,
    )
