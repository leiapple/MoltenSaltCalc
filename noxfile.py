"""Nox configuration for uMLIP testing in different environments."""

import os
from pathlib import Path

import nox

nox.options.envdir = Path.home() / ".nox"

MODELS = [
    "7net",
    "chgnet",
    "fairchem",
    "grace-nodisp",
    "mace",
    "mattersim",
    "nequip",
    "nequix",
    "upet",
    "equiformer_v3",
]


@nox.session(name="umlip", venv_backend="uv", reuse_venv=True)
@nox.parametrize("model", MODELS)
def test_umlip(session, model):
    """Test uMLIPs specified in the model parameter."""

    if "HF_TOKEN" in os.environ:
        session.env["HF_TOKEN"] = os.environ["HF_TOKEN"]

    session.install("pytest")
    session.install(f".[{model}]")
    session.run(
        "pytest",
        "tests/test_uMLIPs.py",
        f"--model={model}",
        "-m",
        "umlip",
        "--disable-warnings",
    )
