"""Provides helper functions for the MoltenSaltAnalyzer class."""

import os
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.geometry.rdf import get_rdf
from ase.io import Trajectory
from ase.io.ulm import InvalidULMFileError


def _trajectory_worker(args):
    """Load one trajectory and return it with its computed times and metadata."""
    traj_file, temperature, run_id, calculator, timestep_fs, no_timestep = args
    if not Path(traj_file).exists():
        warnings.warn(f"Trajectory file {traj_file} not found. Skipping.", stacklevel=2)
        return None

    try:
        traj_obj = Trajectory(traj_file)
        traj = []
        for i, atoms in enumerate(traj_obj):  # type: ignore
            try:
                traj.append(atoms)
            except Exception as e:  # pylint: disable=broad-exception-caught
                warnings.warn(
                    f"Skipping frame {i} in {traj_file}: {type(e).__name__}: {e}",
                    stacklevel=2,
                )
    except InvalidULMFileError as e:
        warnings.warn(f"Error loading trajectory file {traj_file}: {e}. Skipping.", stacklevel=2)
        return None

    if all("time_fs" in getattr(atoms, "info", {}) for atoms in traj) and no_timestep:
        times = np.array([atoms.info["time_fs"] for atoms in traj])
    else:
        if no_timestep:
            warnings.warn(
                f"WARNING: No time_fs found in {os.path.basename(traj_file)}, assuming a constant timestep of "
                f"{timestep_fs} fs. Modify with analyzer.recompute_times(timestep_fs).",
                stacklevel=2,
            )
        times = np.arange(len(traj)) * timestep_fs

    if calculator is not None:
        try:
            for atoms in traj:
                atoms.calc = calculator
        except Exception as e:  # pylint: disable=broad-exception-caught
            warnings.warn(
                f"Failed while attaching calculator to {traj_file}: {type(e).__name__}: {e}. Skipping.",
                stacklevel=2,
            )
            return None

    return traj, times, temperature, run_id


def _rdf_worker(args) -> tuple[np.ndarray, np.ndarray]:
    """Worker function for parallel RDF computation."""
    positions, numbers, cell, pbc, rmax, nbins, elements_nr = args
    atoms = Atoms(
        positions=positions,
        numbers=numbers,
        cell=cell,
        pbc=pbc,
    )
    rdf, distances = get_rdf(atoms, rmax, nbins, elements=elements_nr)
    return rdf, distances


def _msd_element_worker(args) -> tuple[str, float, np.ndarray, np.ndarray]:
    """Worker function for parallel MSD computation (for diffusion coeffs)."""
    symbol, pos_species, times, max_lag_fs, fit_min_fs, fit_max_fs = args
    lag_times = times[times <= max_lag_fs]
    msd = np.zeros_like(lag_times)

    for i in range(1, len(lag_times)):
        sq_disp = (pos_species[i:, :, :] - pos_species[:-i, :, :]) ** 2
        msd[i] = np.mean(np.sum(sq_disp, axis=2))

    # Linear fit on the desired diffusive interval (avoid ballistic regime and long-time noisy data)
    fit_mask = (lag_times >= fit_min_fs) & (lag_times <= fit_max_fs)

    slope, _ = np.polyfit(
        lag_times[fit_mask],
        msd[fit_mask],
        1,
    )

    return symbol, slope / 6.0, lag_times, msd


def _get_unwrapped_positions(traj) -> np.ndarray:
    """Reconstruct continuous positions from a periodic ASE trajectory."""
    positions = np.asarray([atoms.get_positions() for atoms in traj])
    unwrapped = np.empty_like(positions)
    unwrapped[0] = positions[0]

    pbc = traj[0].get_pbc()

    for i in range(1, len(traj)):
        delta = positions[i] - positions[i - 1]
        delta_mic, _ = find_mic(
            delta,
            cell=traj[i - 1].cell,
            pbc=pbc,
        )
        unwrapped[i] = unwrapped[i - 1] + delta_mic

    return unwrapped
