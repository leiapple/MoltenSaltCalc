"""Implementation of the VASP DFT calculator."""

from moltensaltcalc.registry import register_model


@register_model(
    "vasp",
    metadata={
        "command": {
            "type": "str",
            "choices": ["mpirun vasp_std", "vasp_std", "vasp_gam", "vasp_ncl", "Disable"],
            "description": "VASP command to use. Alternatively set to 'Disable' to and use the ASE_VASP_COMMAND environment variable.",
            "default": "mpirun vasp_std",
        },
        "xc": {
            "type": "str",
            "choices": ["PBE", "LDA", "revPBE", "RPBE"],
            "description": "Exchange-correlation functional.",
            "default": "PBE",
        },
        "encut": {
            "type": "int",
            "description": "Plane-wave cutoff energy in eV.",
            "default": 400,
        },
        "ivdw": {
            "type": "int",
            "description": "Van der Waals dispersion correction (e.g., 12 for Grimme D3).",
            "default": 12,
        },
        "kpts": {
            "type": "list",
            "description": "K-point grid as a list of 3 integers.",
            "default": [1, 1, 1],
        },
        "nelm": {
            "type": "int",
            "description": "Maximum electronic SC cycles per ionic step.",
            "default": 60,
        },
        "pp_version": {
            "type": "int | str",
            "description": "The pseudopotential version (suffix) to use, such as '', 52, 54, 64.",
            "default": "",
        },
        "prec": {
            "type": "str",
            "choices": ["Accurate", "Normal", "Low"],
            "description": "Accuracy to use for the calculations.",
            "default": "Accurate",
        },
        "ismear": {
            "type": "int",
            "description": "Gaussian smearing (ideal for liquids/molten salts).",
            "default": 0,
        },
        "sigma": {
            "type": "float",
            "description": "Smearing width.",
            "default": 0.05,
        },
        "lreal": {
            "type": "str | bool",
            "choices": ["Auto", "On", True, False],
            "description": "Evaluate projection operators in real space for speed.",
            "default": "Auto",
        },
        "algo": {
            "type": "str",
            "choices": ["VeryFast", "Fast", "Normal"],
            "description": "Electronic minimization algorithm: RMM-DIIS electronic optimization for fast MD steps, hybrid approach or blocked-Davidson.",
            "default": "VeryFast",
        },
    },
)
def _build(params, device=None):  # pylint: disable=unused-argument
    """Import and build the VASP ASE Calculator."""
    from ase.calculators.vasp import Vasp

    command = params.get("command", "mpirun vasp_std")
    if command.lower() == "disable":
        command = None
    xc = params.get("xc", "PBE")
    encut = params.get("encut", 400)
    ivdw = params.get("ivdw", 12)
    kpts = params.get("kpts", [1, 1, 1])
    nelm = params.get("nelm", 60)
    pp_version = params.get("pp_version", "")
    prec = params.get("prec", "Accurate")
    ismear = params.get("ismear", 0)
    sigma = params.get("sigma", 0.05)
    lreal = params.get("lreal", "Auto")
    algo = params.get("algo", "VeryFast")

    calc = Vasp(
        command=command,
        xc=xc,
        encut=encut,
        ivdw=ivdw,
        kpts=kpts,
        pp_version=pp_version,
        prec=prec,
        nelm=nelm,
        ismear=ismear,
        sigma=sigma,
        lreal=lreal,
        algo=algo,
    )

    return calc
