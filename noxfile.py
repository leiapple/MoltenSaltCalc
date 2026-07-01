"""Nox configuration for uMLIP testing in different environments."""

import os
from pathlib import Path

import nox

import moltensaltcalc as msc

nox.options.envdir = Path.home() / ".nox"


@nox.session(name="umlip", venv_backend="uv", reuse_venv=True, python="3.12.13")
@nox.parametrize("model", msc.available_models())
def test_umlip(session, model):
    """Test uMLIPs specified in the model parameter."""

    if "HF_TOKEN" in os.environ:
        session.env["HF_TOKEN"] = os.environ["HF_TOKEN"]

    if model.lower() == "grace":
        model = "grace-nodisp"

    if model.lower() == "vasp":
        return

    # Check the python version
    session.run("python", "-c", "import sys; print(sys.executable); print(sys.version)")

    # Install the dependencies
    # session.install("pytest")
    # session.install(f".[{model}]")
    session.run("uv", "sync", "--active", "--extra", "dev", "--extra", model)

    # Run the short test
    session.run(
        "pytest",
        "tests/test_uMLIPs.py",
        f"--model={model}",
        "-m",
        "umlip",
        "--disable-warnings",
    )
