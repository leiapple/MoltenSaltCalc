"""Nox configuration for uMLIP testing in different environments."""

from pathlib import Path

import nox

nox.options.envdir = Path.home() / ".nox"

MODELS = [
    "7net",
    "chgnet",
    "fairchem",
    "grace",
    "mace",
    "mattersim",
    "nequip",
    "nequix",
    "upet",
]


@nox.session(name="umlip", venv_backend="uv", reuse_venv=True)
@nox.parametrize("model", MODELS)
def test_umlip(session, model):
    """Test uMLIPs specified in the model parameter."""
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
