from pathlib import Path

import pytest
from ase.io import Trajectory

import moltensaltcalc as msc

BASE = Path(__file__).parent


@pytest.mark.umlip
def test_umlip_minimal(request, tmp_path):
    model = request.config.getoption("--model")

    if model is None:
        pytest.skip(
            f"Skipping uMLIP tests, as no model was specified. Run tests for the different uMLIPs with 'nox -s' or 'nox -s \"umlip(model='model_name')\"'."
        )

    # Nequip needs a precompiled model
    params = (
        {"model_path": BASE / "test_uMLIP_precompiled" / "oam-s-0.1.nequip.pth"}
        if model == "nequip"
        else {}
    )

    sim = msc.MoltenSaltSimulator(
        model_name=model,
        model_parameters=params,
        device="cpu",
    )

    atoms = sim.build_system(
        ["Cl"],
        ["Na"],
        [10],
        [10],
        density_guess=2.0,
    )

    traj_file = tmp_path / "traj.traj"
    n_steps = 2
    sim.run_npt_simulation(
        atoms,
        T=1200,
        steps=n_steps,
        write_interval=1,
        traj_file=traj_file,
        logfile=tmp_path / "log.log",
    )

    # Minimal sanity check that an output was produced and the trajectory file is readable
    assert traj_file.exists(), "No trajectory file was produced"
    traj = Trajectory(traj_file)
    actual_n_steps = len(traj) - 1  # The first frame is the initial structure
    assert (
        actual_n_steps == n_steps
    ), f"The trajectory file contains {actual_n_steps} frames instead of {n_steps}"
    atoms = traj[-1]  # type: ignore
    assert atoms.get_potential_energy() is not None, "Potential energy cannot be obtained from the written trajectory from atoms.get_potential_energy()"  # type: ignore
    assert atoms.get_forces() is not None, "Forces cannot be obtained from the written trajectory from atoms.get_forces()"  # type: ignore
    assert atoms.get_stress() is not None, "Stress cannot be obtained from the written trajectory from atoms.get_stress()"  # type: ignore
