import os
import re

import numpy as np
import pytest
from ase.io import Trajectory

import moltensaltcalc as msc
from moltensaltcalc.simulator import (
    MODEL_METADATA,
    format_model_error,
    format_unknown_model_error,
)

rng_seed = 42


def parse_md_print_line(line: str):
    pattern = (
        r"(?P<step>\d+)\s*\|\s*"
        r"T\s*=\s*(?P<T>[0-9.eE+-]+)\s*K\s*\|\s*"
        r"P\s*=\s*(?P<P>[0-9.eE+-]+)\s*bar\s*\|\s*"
        r"V\s*=\s*(?P<V>[0-9.eE+-]+)\s*Å³"
    )

    match = re.search(pattern, line)
    assert match is not None, f"Could not parse MD output line: {line}"

    return {
        "step": int(match.group("step")),
        "T": float(match.group("T")),
        "P": float(match.group("P")),
        "V": float(match.group("V")),
    }


# =========================================================
# Fixtures
# =========================================================


@pytest.fixture
def simulator():
    """Create a lightweight simulator instance."""
    return msc.MoltenSaltSimulator(
        model_name="grace",
        model_parameters={"model_size": "small", "num_layers": 1, "model_task": "OMAT"},
        device="cpu",
    )


@pytest.fixture
def simple_salt():
    """Minimal salt system for testing."""
    return {
        "anions": ["F", "Cl"],
        "cations": ["Na", "K"],
        "n_anions": [10, 10],
        "n_cations": [15, 5],
        "density": 2.0,  # g/cm3
    }


# =========================================================
# Basic functionality and system building tests
# =========================================================


def test_available_models():
    models = msc.available_models()
    assert isinstance(models, list)
    assert "grace" in models


def test_build_system_random(simulator, simple_salt):
    np.random.seed(rng_seed)
    atoms = simulator.build_system(
        simple_salt["anions"],
        simple_salt["cations"],
        simple_salt["n_anions"],
        simple_salt["n_cations"],
        simple_salt["density"],
        lattice="random",
    )

    N_tot, N_tot_ref = len(atoms), 40
    assert (
        N_tot == N_tot_ref
    ), f"Random build system number of atoms is {N_tot} instead of {N_tot_ref}"
    N_Na, N_Na_ref = len([atom for atom in atoms if atom.symbol == "Na"]), 15
    assert (
        N_Na == N_Na_ref
    ), f"Random build system number of Na atoms is {N_Na} instead of {N_Na_ref}"
    x_0, x_0_ref = atoms.get_positions()[0][0], 4.350641283744996
    assert np.isclose(
        x_0, x_0_ref, atol=1e-5
    ), f"Random build system first atom x-coordinate is {x_0:.5f} instead of {x_0_ref:.5f}"
    vol, vol_ref = atoms.get_volume(), 900.6947000334296
    assert np.isclose(
        vol, vol_ref, atol=1e-5
    ), f"Random build system volume is {vol:.5f} instead of {vol_ref:.5f}"


def test_build_system_rocksalt(simulator, simple_salt):
    np.random.seed(rng_seed)
    atoms = simulator.build_system(
        simple_salt["anions"],
        simple_salt["cations"],
        simple_salt["n_anions"],
        simple_salt["n_cations"],
        simple_salt["density"],
        lattice="rocksalt",
    )
    N_tot, N_tot_ref = len(atoms), 40
    assert (
        N_tot == N_tot_ref
    ), f"Rocksalt build system number of atoms is {N_tot} instead of {N_tot_ref}"
    N_Na, N_Na_ref = len([atom for atom in atoms if atom.symbol == "Na"]), 15
    assert (
        N_Na == N_Na_ref
    ), f"Rocksalt build system number of Na atoms is {N_Na} instead of {N_Na_ref}"
    x_2, x_2_ref = atoms.get_positions()[1][0], 2.5550218342426883
    assert np.isclose(
        x_2, x_2_ref, atol=1e-5
    ), f"Rocksalt build system second atom x-coordinate is {x_2:.5f} instead of {x_2_ref:.5f}"
    vol, vol_ref = atoms.get_volume(), 900.6947000334296
    assert np.isclose(
        vol, vol_ref, atol=1e-5
    ), f"Rocksalt build system volume is {vol:.5f} instead of {vol_ref:.5f}"


def test_build_system_rocksalt_random_removal(simulator, simple_salt):
    np.random.seed(rng_seed)
    atoms = simulator.build_system(
        simple_salt["anions"],
        simple_salt["cations"],
        simple_salt["n_anions"],
        simple_salt["n_cations"],
        simple_salt["density"],
        lattice="rocksalt",
        random_removal=True,
    )

    N_tot, N_tot_ref = len(atoms), 40
    assert (
        N_tot == N_tot_ref
    ), f"Rocksalt build system random removal number of atoms is {N_tot} instead of {N_tot_ref}"
    N_Na, N_Na_ref = len([atom for atom in atoms if atom.symbol == "Na"]), 15
    assert (
        N_Na == N_Na_ref
    ), f"Rocksalt build system random removal number of Na atoms is {N_Na} instead of {N_Na_ref}"
    x_6, x_6_ref = atoms.get_positions()[6][0], 7.66507
    assert np.isclose(
        x_6, x_6_ref, atol=1e-5
    ), f"Rocksalt build system second atom x-coordinate is {x_6:.5f} instead of {x_6_ref:.5f}"
    vol, vol_ref = atoms.get_volume(), 900.6947000334296
    assert np.isclose(
        vol, vol_ref, atol=1e-5
    ), f"Rocksalt build system volume is {vol:.5f} instead of {vol_ref:.5f}"


# =========================================================
# File system tests
# =========================================================


def test_create_simulation_folder(simulator, tmp_path):
    npt_dir, nvt_dir = simulator.create_simulation_folder(base_name=str(tmp_path))

    assert os.path.exists(npt_dir)
    assert os.path.exists(nvt_dir)


# =========================================================
# MD integration test (slow)
# =========================================================


@pytest.mark.slow
def test_short_md_run(simulator, simple_salt, capsys, tmp_path):
    np.random.seed(rng_seed)

    npt_dir, nvt_dir = simulator.create_simulation_folder(base_name=str(tmp_path))

    atoms = simulator.build_system(
        simple_salt["anions"],
        simple_salt["cations"],
        simple_salt["n_anions"],
        simple_salt["n_cations"],
        simple_salt["density"],
        lattice="rocksalt",
    )

    T = 1200
    traj_file_npt = os.path.join(npt_dir, "npt_simulation.traj")
    traj_file_nvt = os.path.join(nvt_dir, "nvt_simulation.traj")
    log_file_npt = os.path.join(npt_dir, "npt_run.log")
    log_file_nvt = os.path.join(nvt_dir, "nvt_run.log")

    atoms = simulator.run_npt_simulation(
        atoms,
        T=T,
        steps=5,
        print_interval=1,
        write_interval=1,
        traj_file=str(traj_file_npt),
        logfile=str(log_file_npt),
        timestep_fs=10.0,
    )

    captured = capsys.readouterr()
    npt_lines = [
        l for l in captured.out.splitlines() if "|" in l and "T =" in l and "P =" in l
    ]

    # Assertions for the NPT run
    assert len(npt_lines) > 0, "No MD output found in capsys from the NPT run"
    last_npt = parse_md_print_line(npt_lines[-1])
    # Expected: Step      5 | T = 2603.327386 K | P = 1.736778e-02 bar | V =   997.55 Å³
    final_T, final_T_ref = last_npt["T"], 2603.327386
    assert np.isclose(
        final_T, final_T_ref, atol=1e-1
    ), f"NPT final T = {final_T:.1f} instead of expected {final_T_ref:.1f}"
    final_P, final_P_ref = last_npt["P"], 1.736778e-02
    assert np.isclose(
        final_P, final_P_ref, atol=1e-5
    ), f"NPT final P = {final_P:.5f} instead of expected {final_P_ref:.5f}"
    final_V, final_V_ref = last_npt["V"], 997.55
    assert np.isclose(
        final_V, final_V_ref, atol=1e-1
    ), f"NPT final V = {final_V:.1f} instead of expected {final_V_ref:.1f}"

    # Ensure the trajectory file exists and is readable
    assert os.path.exists(traj_file_npt)

    traj = Trajectory(traj_file_npt)
    N_frames, N_frames_ref = len(traj), 6
    assert (
        N_frames == N_frames_ref
    ), f"NPT Trajectory length is {N_frames} instead of {N_frames_ref}"
    last_atoms = traj[-1]
    # Ensure the timestep is saved and correct
    assert (
        "time_fs" in last_atoms.info
    ), f"NPT time_fs not in last_atoms.info: {last_atoms.info}"
    final_time_fs, final_time_fs_ref = last_atoms.info["time_fs"], 50.0
    assert np.isclose(
        final_time_fs, final_time_fs_ref, atol=1e-5
    ), f"NPT Time of last frame is {final_time_fs:.5f} instead of {final_time_fs_ref:.5f}"

    # Ensure the energy is correct
    final_energy, final_energy_ref = last_atoms.get_total_energy(), -119.87897281091125
    assert np.isclose(
        final_energy, final_energy_ref, atol=1e-5
    ), f"NTP Energy of last frame is {final_energy:.5f} instead of {final_energy_ref:.5f}"

    atoms = simulator.run_nvt_simulation(
        atoms,
        T=T,
        steps=5,
        print_interval=1,
        write_interval=1,
        traj_file=str(traj_file_nvt),
        logfile=str(log_file_nvt),
        timestep_fs=10.0,
    )

    captured = capsys.readouterr()
    nvt_lines = [
        l for l in captured.out.splitlines() if "|" in l and "T =" in l and "P =" in l
    ]

    # Assertions for the NVT run
    assert len(nvt_lines) > 0, "No MD output found in capsys from the NPT run"
    last_nvt = parse_md_print_line(nvt_lines[-1])
    # Expected: 5 | T = 1691.756416 K | P = 1.149296e-02 bar | V =   997.55 Å³
    final_T, final_T_ref = last_nvt["T"], 1691.756416
    assert np.isclose(
        final_T, final_T_ref, atol=1e-1
    ), f"NVT final T = {final_T:.1f} instead of expected {final_T_ref:.1f}"
    final_P, final_P_ref = last_nvt["P"], 1.149296e-02
    assert np.isclose(
        final_P, final_P_ref, atol=1e-5
    ), f"NVT final P = {final_P:.5f} instead of expected {final_P_ref:.5f}"
    final_V, final_V_ref = last_nvt["V"], 997.55
    assert np.isclose(
        final_V, final_V_ref, atol=1e-1
    ), f"NVT final V = {final_V:.1f} instead of expected {final_V_ref:.1f}"

    # Ensure the trajectory file exists and is readable
    assert os.path.exists(traj_file_nvt)

    traj = Trajectory(traj_file_nvt)
    N_frames, N_frames_ref = len(traj), 6
    assert (
        N_frames == N_frames_ref
    ), f"Trajectory length is {N_frames} instead of {N_frames_ref}"
    last_atoms = traj[-1]
    # Ensure the timestep is saved and correct
    assert (
        "time_fs" in last_atoms.info
    ), f"NVT final time_fs not in last_atoms.info: {last_atoms.info}"
    final_time_fs, final_time_fs_ref = last_atoms.info["time_fs"], 50.0
    assert np.isclose(
        final_time_fs, final_time_fs_ref, atol=1e-5
    ), f"NVT Time of last frame is {final_time_fs:.5f} instead of {final_time_fs_ref:.5f}"

    # Ensure the energy is correct
    final_energy, final_energy_ref = last_atoms.get_total_energy(), -127.93872350976946
    assert np.isclose(
        final_energy, final_energy_ref, atol=1e-5
    ), f"NVT Energy of last frame is {final_energy:.5f} instead of {final_energy_ref:.5f}"


# =========================================================
# Error handling tests
# =========================================================


def test_invalid_model():
    with pytest.raises(ValueError) as exc:
        msc.MoltenSaltSimulator(model_name="gace", model_parameters={})

    assert "Unknown model" in str(exc.value)


def test_format_unknown_model_error_basic():
    msg = format_unknown_model_error("foo", ["bar", "baz"])

    assert "Unknown model 'foo'" in msg
    assert "Available models" in msg
    assert "- 'bar'" in msg
    assert "- 'baz'" in msg


def test_format_unknown_model_error_empty():
    msg = format_unknown_model_error("foo", [])

    assert "Unknown model 'foo'" in msg
    assert "Available models" in msg


def test_format_model_error_with_metadata(monkeypatch):
    monkeypatch.setitem(MODEL_METADATA, "test_model", {"param1": [1, 2, 3]})

    err = RuntimeError("fail")
    msg = format_model_error("test_model", {}, err)

    assert "Known parameter options for test_model" in msg
    assert "- param1: [1, 2, 3]" in msg


def test_format_model_error_no_metadata(monkeypatch):
    monkeypatch.setitem(MODEL_METADATA, "empty_model", {})

    err = RuntimeError("fail")
    msg = format_model_error("empty_model", {}, err)

    assert "Model initialization failed" in msg
    assert "Known parameter options" not in msg


def test_build_system_invalid_lengths(simulator):
    with pytest.raises(ValueError):
        simulator.build_system(
            ["Cl"],
            ["Na"],
            [10, 10],  # mismatch
            [10],
            density_guess=2.0,
        )


def test_build_system_unsupported_lattice(simulator):
    with pytest.raises(ValueError) as e:
        simulator.build_system(
            ["Cl"],
            ["Na"],
            [10],
            [10],
            density_guess=2.0,
            lattice="foo",
        )
    assert "Unsupported lattice type" in str(e.value)


@pytest.mark.slow
def test_build_system_invalid_density(simulator):
    with pytest.raises(RuntimeError) as e:
        simulator.build_system(
            ["Cl"],
            ["Na"],
            [10],
            [10],
            lattice="random",
            density_guess=1000.0,  # g/cm³, not possible with min_distance=1.6 Å
        )
    assert "Increase the initial density guess" in str(e.value)
