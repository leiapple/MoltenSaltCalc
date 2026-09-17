"""MoltenSaltAnalyzer class for analyzing molecular dynamics simulations."""

import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers, chemical_symbols
from ase.geometry.rdf import get_rdf
from ase.io import Trajectory
from ase.io.ulm import InvalidULMFileError
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm


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


class MoltenSaltAnalyzer:
    """
    Class for analyzing molten salt simulation results.
    """

    def __init__(
        self,
        traj_files_npt: list[str] | list[Path] | str | Path | None = None,
        traj_files_nvt: list[str] | list[Path] | str | Path | None = None,
        temperatures_npt: list[float] | list[int] | None = None,
        temperatures_nvt: list[float] | list[int] | None = None,
        ids_npt: list[str] | None = None,
        ids_nvt: list[str] | None = None,
        timestep_fs: int | float | None = None,
        calculator: Calculator | None = None,
        n_workers_load_traj: int | None = None,
    ):
        """Initialize the analyzer with the trajectories and the always used properties

        Args:
            traj_files_npt (list, str, Path, optional): Path to the NPT trajectory file(s). Defaults to None.
            traj_files_nvt (list, str, Path, optional): Path to the NVT trajectory file(s). Defaults to None.
            temperatures_npt (list, optional): List of temperatures in K for the NPT trajectories. Defaults to None.
            temperatures_nvt (list, optional): List of temperatures in K for the NVT trajectories. Defaults to None.
            ids_npt (list, optional): List of identifiers for the NPT trajectories. Defaults to None.
            ids_nvt (list, optional): List of identifiers for the NVT trajectories. Defaults to None.
            timestep_fs (int, float, optional): Constant timestep in fs. Only applies if time_fs is not found in the trajectory files. Defaults to None which is treated as 10.0 later but a warning is issued.
            calculator (ase.calculators.calculator, optional): Calculator to use for the energy and forces predictions (needed in case they are not available from the trajectory files, but leads to a slow initialization). Defaults to None.
            n_workers_load_traj (int, optional): Number of workers to use for loading the trajectory files in parallel. Defaults to None, which means the maximum of len(traj_files_npt) and len(traj_files_nvt) capped by 20.

        Raises:
            ValueError: If the number of trajectory files is not equal to the number of temperatures.
            ValueError: If ids are provided but the number of trajectory files is not equal to the number of ids.
            FileNotFoundError: If the trajectory file(s) does not exist.

        Defines:
            self.trajs_npt (list): List of Trajectory objects for the NPT trajectories.
            self.trajs_nvt (list): List of Trajectory objects for the NVT trajectories.
            self.times_fs_npt (list[np.ndarray]): List of arrays of times in fs for each NPT trajectory.
            self.times_fs_nvt (list[np.ndarray]): List of arrays of times in fs for each NVT trajectory.
            self.temperatures_npt (list): List of temperatures in K for the NPT trajectories.
            self.temperatures_nvt (list): List of temperatures in K for the NVT trajectories.
            self.ids_npt (list): List of identifiers for the NPT trajectories.
            self.ids_nvt (list): List of identifiers for the NVT trajectories.
            self.timestep_fs (float): Constant timestep in fs, only applies if time_fs is not found in the trajectory files.
        """
        self.times_fs_npt, self.times_fs_nvt, self.trajs_npt, self.trajs_nvt = None, None, None, None
        self.ids_npt, self.ids_nvt = ids_npt, ids_nvt
        no_timestep = timestep_fs is None
        if no_timestep:
            timestep_fs = 10.0  # fs
        self.timestep_fs = timestep_fs
        self.temperatures_npt, self.temperatures_nvt = temperatures_npt, temperatures_nvt
        if isinstance(traj_files_npt, (str, Path)):
            traj_files_npt = [Path(traj_files_npt)]
        if isinstance(traj_files_nvt, (str, Path)):
            traj_files_nvt = [Path(traj_files_nvt)]

        if n_workers_load_traj is None:
            n_workers_load_traj = min(
                max(
                    len(traj_files_npt) if traj_files_npt is not None else 0,
                    len(traj_files_nvt) if traj_files_nvt is not None else 0,
                ),
                20,
            )

        if traj_files_npt is not None:
            self.trajs_npt, self.times_fs_npt, self.temperatures_npt, self.ids_npt = self._load_trajectories(
                traj_files_npt,
                temperatures_npt,
                ids_npt,
                calculator,
                "NPT",
                timestep_fs,
                no_timestep,
                n_workers_load_traj,
            )
        if traj_files_nvt is not None:
            self.trajs_nvt, self.times_fs_nvt, self.temperatures_nvt, self.ids_nvt = self._load_trajectories(
                traj_files_nvt,
                temperatures_nvt,
                ids_nvt,
                calculator,
                "NVT",
                timestep_fs,
                no_timestep,
                n_workers_load_traj,
            )

    def _load_trajectories(
        self,
        traj_files: list[str] | list[Path],
        temperatures: list[float] | list[int] | None,
        ids: list[str] | None,
        calculator: Calculator | None,
        id_str: str,
        timestep_fs: float,
        no_timestep: bool,
        n_workers: int = 1,
    ) -> tuple[list | None, list | None, list | None, list | None]:
        """Load the trajectories from the provided files."""
        if temperatures is None or len(traj_files) != len(temperatures):
            raise ValueError(f"Number of {id_str} trajectory files and temperatures_{id_str.lower()} must match.")
        if ids is not None and len(traj_files) != len(ids):
            raise ValueError(f"Number of {id_str} trajectory files and ids_{id_str.lower()} must match.")

        valid_trajs, valid_times_fs, valid_temperatures = [], [], []
        valid_ids = [] if ids is not None else None
        traj_list = list(
            zip(
                traj_files,
                temperatures,
                ids if ids is not None else [None] * len(traj_files),
                strict=True,
            )
        )
        worker_args = [
            (traj_file, temperature, run_id, calculator, timestep_fs, no_timestep)
            for traj_file, temperature, run_id in traj_list
        ]
        if n_workers > 1:
            with Pool(processes=n_workers) as pool:
                loaded = pool.imap(_trajectory_worker, worker_args)
                for result in tqdm(loaded, total=len(worker_args), desc=f"Loading {id_str} trajectories"):
                    if result is None:
                        continue
                    traj, times, temperature, run_id = result
                    valid_trajs.append(traj)
                    valid_times_fs.append(times)
                    valid_temperatures.append(temperature)
                    if valid_ids is not None:
                        valid_ids.append(run_id)
        else:
            for args in worker_args:
                result = _trajectory_worker(args)
                if result is None:
                    continue
                traj, times, temperature, run_id = result
                valid_trajs.append(traj)
                valid_times_fs.append(times)
                valid_temperatures.append(temperature)
                if valid_ids is not None:
                    valid_ids.append(run_id)

        return (
            valid_trajs,
            valid_times_fs,
            valid_temperatures,
            valid_ids,
        )

    def recompute_times(self, timestep_fs: int | float):
        """Sets the times corresponding to the atoms in the trajectories according to the provided constant timestep.

        Args:
            timestep_fs (int, float): Newly chosen timestep in fs.
        """
        self.timestep_fs = timestep_fs
        if self.trajs_npt is not None:
            self.times_fs_npt = [np.arange(len(traj)) * self.timestep_fs for traj in self.trajs_npt]

        if self.trajs_nvt is not None:
            self.times_fs_nvt = [np.arange(len(traj)) * self.timestep_fs for traj in self.trajs_nvt]

    def _select_trajectory(
        self, preferred_type: str, T: int | float | None = None, traj_id: str | None = None
    ) -> tuple[Trajectory, np.ndarray]:  # type: ignore
        """Select trajectory for a given temperature.

        Args:
            preferred_type (str): Preferred ensemble to select if both are available.
                Must be either "npt" or "nvt". If the preferred type is not available
                for the requested temperature, the available trajectory is returned.
            T (int, float, optional): Temperature in K for which the trajectory should be selected. Defaults to None.
            traj_id (str, optional): Identifier for the trajectory to select, overrides T. Defaults to None.

        Raises:
            ValueError: If no trajectory files were provided during initialization.
            ValueError: If preferred_type is not "npt" or "nvt".
            ValueError: If the requested id/temperature is not available in any trajectory.

        Returns:
            Tuple[Trajectory, np.ndarray]: The selected trajectory object and the corresponding simulation times in fs.
        """
        if self.trajs_npt is None and self.trajs_nvt is None:
            raise ValueError("No trajectory files provided.")
        if preferred_type not in ["npt", "nvt"]:
            raise ValueError("preferred_type must be either 'npt' or 'nvt'.")

        candidates = {}

        if traj_id is not None:
            # Select trajectory by ID
            if self.ids_npt is not None and traj_id in self.ids_npt:
                idx = self.ids_npt.index(traj_id)
                candidates["npt"] = (self.trajs_npt[idx], self.times_fs_npt[idx])  # type: ignore
            if self.ids_nvt is not None and traj_id in self.ids_nvt:
                idx = self.ids_nvt.index(traj_id)
                candidates["nvt"] = (self.trajs_nvt[idx], self.times_fs_nvt[idx])  # type: ignore
        else:
            # Select trajectory by temperature
            if self.temperatures_npt is not None and T in self.temperatures_npt:
                idx = self.temperatures_npt.index(T)  # type: ignore
                candidates["npt"] = (self.trajs_npt[idx], self.times_fs_npt[idx])  # type: ignore
            if self.temperatures_nvt is not None and T in self.temperatures_nvt:
                idx = self.temperatures_nvt.index(T)  # type: ignore
                candidates["nvt"] = (self.trajs_nvt[idx], self.times_fs_nvt[idx])  # type: ignore

        if not candidates:
            raise ValueError(f"Id {traj_id} or temperature {T} not found in any trajectories.")

        if preferred_type in candidates:
            return candidates[preferred_type]

        return next(iter(candidates.values()))

    def _get_eq_indices(self, eq_fraction: float, times_fs: np.ndarray) -> np.ndarray:
        """Gets the indices of the simulation times later than 1-eq_fraction of the total simulation time.

        Args:
            eq_fraction (float): Fraction of the total simulation time from the end of the simulation
            times_fs (np.ndarray): Times in femtoseconds

        Returns:
            np.ndarray: Indices of the simulation times later than 1-eq_fraction of the total simulation time.
        """
        eq_indices = np.where(times_fs >= np.max(times_fs) * (1 - eq_fraction))[0]
        return eq_indices

    def compute_temperature_vs_time(
        self, traj_id: str | None = None, T: int | float | None = None, eq_fraction: float = 0.1, ensemble: str = "nvt"
    ):
        """Compute the temperature from the trajectory file.
        Args:
            traj_id (str, optional): Identifier for the trajectory. Defaults to None.
            T (int, float, optional): Temperature in K. The trajectory with the matching temperature is selected if traj_id is None. Defaults to None.
            eq_fraction (float, optional): Final fraction of the simulation time to be considered as equilibrium. Defaults to 0.1.
            ensemble (str, optional): Ensemble to select preferentially. Defaults to "nvt".

        Raises:
            ValueError: If eq_fraction is not between 0 and 1.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Temperatures in K and equilibrium times.
        """
        if eq_fraction > 1.0 or eq_fraction < 0.0:
            raise ValueError("eq_fraction must be between 0 and 1.")

        traj, times = self._select_trajectory(ensemble, T, traj_id)
        eq_indices = self._get_eq_indices(eq_fraction, times)
        traj = [traj[i] for i in eq_indices]
        temperatures = np.array([atoms.get_temperature() for atoms in traj])
        return temperatures, eq_indices

    def compute_density_vs_time(
        self, traj_id: str | None = None, T: int | float | None = None, eq_fraction: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the density from the trajectory file. If both NPT and NVT trajectories are loaded, the density is computed from the NPT trajectory.

        Args:
            traj_id (str, optional): Identifier for the trajectory. Defaults to None.
            T (int, float, optional): Temperature in K. The trajectory with the matching temperature is selected if traj_id is None. Defaults to None.
            eq_fraction (float, optional): Final fraction of the simulation time to be considered as equilibrium. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Densities in g/cm³ and times in fs.
        """
        traj, times = self._select_trajectory("npt", T, traj_id)
        if eq_fraction is not None:
            eq_indices = self._get_eq_indices(eq_fraction, times)
            traj = [traj[i] for i in eq_indices]
            times = times[eq_indices]
        masses = traj[0].get_masses().sum() / units.kg * 1e3  # g
        volumes = np.array([atoms.get_volume() for atoms in traj]) * 1e-24  # cm³
        densities = masses / volumes  # g/cm³
        return densities, times

    def compute_eq_density(
        self, traj_id: str | None = None, T: int | float | None = None, eq_fraction: float = 0.1
    ) -> float:
        """
        Compute the density after equilibration (last x time% of the trajectory). If both NPT and NVT trajectories are loaded, the density is computed from the NPT trajectory.

        Args:
            traj_id (str, optional): Identifier for the trajectory. Defaults to None.
            T (int, float, optional): Temperature in K. The trajectory with the matching temperature is selected if traj_id is None. Defaults to None.
            eq_fraction (float, optional): Final fraction of the simulation time to be considered as equilibrium. Defaults to 0.1.

        Raises:
            ValueError: If eq_fraction is not between 0 and 1.

        Returns:
            float: Density in g/cm³
        """
        if eq_fraction > 1.0 or eq_fraction < 0.0:
            raise ValueError("eq_fraction must be between 0 and 1.")

        densities, _ = self.compute_density_vs_time(traj_id, T, eq_fraction)
        eq_density = np.mean(densities, dtype="float64")  # g/cm³

        return eq_density

    def _select_trajs_multi_temp(self, ids: list[str] | None = None, property_name="") -> tuple[list, list, list]:
        """Select NPT trajectories, times, and temperatures for multiple temperatures.

        Args:
            ids (list[str], optional): List of trajectory identifiers to select. Defaults to None, which selects all available NPT trajectories.
            property_name (str, optional): Name of the property being computed. Used for error messages. Defaults to an empty string.

        Raises:
            ValueError: If no trajectory identifiers are provided but were not initialized.
            ValueError: If at least one of the provided ids is not available in the initialized NPT trajectories.
            ValueError: If no NPT trajectory files are provided.
            ValueError: If no NPT temperatures are provided.
            ValueError: If less than two NPT trajectory files are provided.

        Returns:
            tuple[list, list, list]: Selected NPT trajectories, times, and temperatures.
        """
        if ids is not None:
            if self.ids_npt is None:
                raise ValueError(
                    "Trajectory identifiers were not initialized, cannot select by id. Please omit the ids argument."
                )
            missing_ids = [id for id in ids if id not in self.ids_npt]
            if missing_ids:
                raise ValueError(f"The ids: {missing_ids} are not available in the initialized NPT trajectories.")
            selected_trajs = []
            selected_times = []
            selected_temps = []
            for traj_id in ids:
                if traj_id in self.ids_npt:
                    idx = self.ids_npt.index(traj_id)
                    selected_trajs.append(self.trajs_npt[idx])  # type: ignore
                    selected_times.append(self.times_fs_npt[idx])  # type: ignore
                    selected_temps.append(self.temperatures_npt[idx])  # type: ignore
        else:
            selected_trajs = self.trajs_npt
            selected_times = self.times_fs_npt
            selected_temps = self.temperatures_npt
        if selected_trajs is None:
            raise ValueError(f"No NPT trajectory files provided. The {property_name} cannot be computed.")
        if selected_temps is None:
            raise ValueError(f"No NPT temperatures provided. The {property_name} cannot be computed.")
        if len(selected_trajs) < 2:
            raise ValueError(f"At least two NPT trajectory files are required for the {property_name}.")

        return selected_trajs, selected_times, selected_temps  # type: ignore

    def compute_thermal_expansion(self, eq_fraction: float = 0.1, ids: list[str] | None = None) -> dict:
        """Compute the thermal expansion coefficient from the initialized trajectory files and NPT temperatures.

        Args:
            eq_fraction (float, optional): Final fraction of the simulation time to be considered as equilibrium. Defaults to 0.1.
            ids (list, optional): Identifiers for the trajectories to be used. Defaults to None.

        Raises:
            ValueError: If insufficient NPT trajectory files or temperatures are available for the computation.

        Returns:
            dict:  Thermal expansion results:
                - "temperatures": List of temperatures used
                - "eq_vols": Equilibrium volumes in Å³ for each temperature
                - "eq_vols_norm": Equilibrium volumes normalized to the mean volume
                - "fit": Fit parameters
                - "fit_line": Fit line
                - "thermal_expansion": Thermal expansion coefficient in 1/K
        """

        if not 0.0 <= eq_fraction <= 1.0:
            raise ValueError("eq_fraction must be between 0 and 1.")

        # Select the NPT trajectories, times, and temperatures
        selected_trajs, selected_times, selected_temps = self._select_trajs_multi_temp(ids, "thermal expansion")

        # Get the equilibrium volumes for each trajectory file
        eq_vols = np.zeros(len(selected_trajs))
        for i, (traj, times) in enumerate(zip(selected_trajs, selected_times, strict=False)):  # type: ignore
            eq_indices = self._get_eq_indices(eq_fraction, times)
            traj = [traj[i] for i in eq_indices]
            volumes = np.array([atoms.get_volume() for atoms in traj])  # Å³
            eq_vol = np.mean(volumes)  # Å³
            eq_vols[i] = eq_vol

        # Fit linear thermal expansion to the volumes normalized by the mean volume
        eq_vols_norm = eq_vols / np.mean(eq_vols)
        fit = np.polyfit(selected_temps, eq_vols_norm, 1)
        fit_line = np.polyval(fit, selected_temps)

        return {
            "temperatures": selected_temps,
            "eq_vols": eq_vols,
            "eq_vols_norm": eq_vols_norm,
            "fit": fit,
            "fit_line": fit_line,
            "thermal_expansion": fit[0],
        }

    def _mean_enthalpy(self, traj, times: np.ndarray, T: float, eq_fraction: float, pressure: float) -> float:
        """Compute the mean enthalpy of a trajectory.

        Args:
            traj (Trajectory): The trajectory to analyze.
            times (np.ndarray): Array of time points corresponding to the trajectory frames.
            T (float): Temperature of the simulation in K.
            eq_fraction (float): Fraction of the simulation time to be considered as equilibrium.
            pressure (float): External pressure in the same units as the volume.

        Returns:
            float: Mean enthalpy of the trajectory.
        """
        eq_indices = self._get_eq_indices(eq_fraction, times)
        traj = [traj[i] for i in eq_indices]
        H = np.array([atoms.get_total_energy() + pressure * atoms.get_volume() for atoms in traj])  # eV
        H_mean = np.mean(H)  # eV
        if traj[0].get_kinetic_energy() == 0:
            H_mean += 1.5 * len(traj[0]) * T * units.kB  # eV
        return H_mean

    def compute_heat_capacity_cp(
        self,
        T: int | float,
        ids: list[str] | None = None,
        eq_fraction: float = 0.1,
        p_ext_bar: float = 1.01325,
    ) -> float:
        """Compute heat capacity from finite differences of the mean enthalpy using NPT trajectories.

        Args:
            T (int, float): Central temperature in K. The two trajectories closest to this temperature are used.
            ids (list[str], optional): List of trajectory identifiers to select. Defaults to None, which selects all available NPT trajectories.
            eq_fraction (float, optional): Final fraction of the simulation time to be considered as equilibrium. Defaults to 0.1.
            p_ext_bar (float, optional): External pressure in bar. Defaults to 1.01325 = 1 atm.

        Raises:
            ValueError: If the specified temperature T is outside the range of available NPT temperatures.
            ValueError: If insufficient NPT trajectory files or temperatures are available for the computation.

        Returns:
            float: Heat capacity in J/g/K
        """

        if not 0.0 <= eq_fraction <= 1.0:
            raise ValueError("eq_fraction must be between 0 and 1.")

        # Select the NPT trajectories, times, and temperatures
        selected_trajs, selected_times, selected_temps = self._select_trajs_multi_temp(ids, "heat capacity")

        # Sort trajectories by temperature
        data = sorted(
            zip(selected_temps, selected_trajs, selected_times, strict=True),
            key=lambda item: item[0],
        )

        # Find the two temperatures surrounding T
        temperatures = np.array([item[0] for item in data])
        idx = np.searchsorted(temperatures, T)
        if idx == 0 or idx == len(temperatures):
            raise ValueError(f"T={T} K must lie between two available NPT temperatures.")

        T1, traj1, times1 = data[idx - 1]
        T2, traj2, times2 = data[idx]

        if T1 == T2:
            raise ValueError("The two selected NPT trajectories must have different temperatures.")

        if any(traj[0].get_kinetic_energy() == 0 for traj in [traj1, traj2]):
            warnings.warn(
                "Kinetic energy of the first frame is zero, which may indicate an issue "
                "with the trajectory. Proceeding with 3N/2*kB*T added to the enthalpy.",
                stacklevel=2,
            )

        pressure = p_ext_bar * units.bar  # Pa
        H1 = self._mean_enthalpy(traj1, times1, T1, eq_fraction, pressure)  # eV
        H2 = self._mean_enthalpy(traj2, times2, T2, eq_fraction, pressure)  # eV

        heat_capacity = (H2 - H1) / ((T2 - T1) * units.J)  # J/K

        mass_g = traj1[0].get_masses().sum() / units.kg * 1e3  # g
        return heat_capacity / mass_g  # J/g/K

    def compute_heat_capacity_cv(self, T: int | float, traj_id: str | None = None, eq_fraction: float = 0.1) -> float:
        """Compute heat capacity from total energy fluctuations. If both NPT and NVT trajectories are loaded, the heat capacity is computed from the NVT trajectory.

        Args:
            T (int, float): Temperature in K. The trajectory with the matching temperature is selected.
            traj_id (str, optional): Identifier for the trajectory, overrides T. Defaults to None.
            eq_fraction (float, optional): Final fraction of the simulation time to be considered as equilibrium. Defaults to 0.1.

        Returns:
            float: Heat capacity in J/g/K
        """
        # Can only select based on temperature if the traj temperatures are provided
        traj, times = self._select_trajectory("nvt", T, traj_id)
        eq_indices = self._get_eq_indices(eq_fraction, times)
        traj = [traj[i] for i in eq_indices]
        U = np.array([atoms.get_total_energy() for atoms in traj])
        # Compute the variation and get the approximate heat capacity C
        var_U = np.var(U, ddof=1) / units.J**2  # J²
        m_tot = traj[0].get_masses().sum() / units.kg * 1e3  # g
        C = var_U / (units.kB / units.C * T**2 * m_tot)  # J/g/K
        return C

    def compute_diffusion_coefficient(self, T: int | float, traj_id: str | None = None) -> dict:
        """Compute diffusion coefficients of each atom species present in the trajectory from the mean squared displacement. If both NPT and NVT trajectories are loaded, the diffusion coefficient is computed from the NVT trajectory, unless selected otherwise with traj_id.

        Args:
            T (int, float): Temperature in K.  The trajectory with the matching temperature is selected.
            traj_id (str, optional): Identifier for the trajectory, overrides T to select the trajectory. Defaults to None.

        Returns:
            dict: Diffusion coefficients in Å²/fs for each element, in the form {element: diffusion coefficient}
        """
        traj, times = self._select_trajectory("nvt", T, traj_id)
        symbols = np.array(traj[0].get_chemical_symbols())
        unique_elements = np.unique(symbols)
        # Get the positions relative to the center of mass
        positions = np.array([atoms.get_positions() - atoms.get_center_of_mass() for atoms in traj])  # Å
        r0 = positions[0]  # Å
        # Get the diffusion coefficients for each atom species
        D = {}
        for symbol in unique_elements:
            mask = symbols == symbol
            # Compute the mean square displacements without variation of time origins
            msd = np.mean(np.sum((positions[:, mask, :] - r0[mask, :]) ** 2, axis=2), axis=1)  # Å²
            slope, _ = np.polyfit(times, msd, 1)
            D[symbol] = slope / 6.0  # Å²/fs

        return D

    def fit_arrhenius(self, temperatures: list[int] | list[float], diffusion_coeffs: list[float]) -> dict:
        """Fit Arrhenius law to diffusion coefficients and temperatures.

        Args:
            temperatures (list): Temperatures in K.
            diffusion_coeffs (list): Diffusion coefficients corresponding to the temperatures in Å²/fs.

        Returns:
            dict: Arrhenius parameters:
                - "Ea": Activation energy in J/mol
                - "D0": Exponential pre-factor of the Arrhenius law in Å²/fs
                - "slope": Slope of the Arrhenius law
                - "intercept": Intercept of the Arrhenius law
        """

        # Linearize: ln(D) = ln(D0) - Ea/(R*T)
        x = 1.0 / np.array(temperatures)  # 1/K
        y = np.log(diffusion_coeffs)  # ln(Å²/fs)

        m, b = np.polyfit(x, y, 1)  # m = slope, b = intercept
        Ea = -m * units.kB / units.C * units.mol  # J/mol
        D0 = np.exp(b)  # Å²/fs

        return {"Ea": Ea, "D0": D0, "slope": m, "intercept": b}

    def compute_rdf(
        self,
        T: float,
        traj_id: str | None = None,
        max_num_frames: int | None = None,
        rmax: float = 6.0,
        nbins: int = 100,
        pairs: list[tuple[int, int]] | list[tuple[str, str]] | None = None,
        cell_constraints: list[tuple[float, float]] | None = None,
        n_workers: int = 1,
    ) -> dict:
        """Compute radial distribution functions. If both NPT and NVT trajectories are loaded, the RDF is computed from the NVT trajectory.

        Args:
            T (float): Temperature in K. The trajectory with the matching temperature is selected.
            traj_id (str, optional): Identifier for the trajectory, overrides T. Defaults to None.
            max_num_frames (int, optional): Maximum number of trajectory frames to compute the RDF for and average over. The frames are selected from the end of the simulation. Defaults to None which means all frames are considered.
            rmax (float, optional): Maximum distance (Å) to consider. Defaults to 6.0.
            nbins (int, optional): Number of bins for the RDF. Defaults to 100.
            pairs (list[tuple] | None, optional): Atom pairs in terms of atomic numbers or symbols to compute the RDF for. Defaults to None which means all unique pairs in the system are analyzed.
            cell_constraints (list[tuple] | None, optional): Whether to compute the RDF only for a subpart of the cell, given by the list of cell constraints. Each constraint is a tuple of the form (min, max) for the x, y, and z coordinates. The boundaries are inclusive. Defaults to None which means all atoms are included.
            n_workers (int, optional): Number of workers to use for parallel RDF computation. Defaults to 1.

        Raises:
            ValueError: If no pairs are specified.

        Returns:
            dict: Dictionary with RDF results:
                - "(atomic number, atomic number)": (distances, avg_rdf, std_rdf) for each pair. Distances are in Å and avg_rdf is unitless (normalized).
        """
        traj, _ = self._select_trajectory("nvt", T, traj_id)

        # Get all unique atomic pairs if not specified
        if pairs is None:
            atm_nums = traj[0].get_atomic_numbers()
            unique_elements = sorted(set(atm_nums))
            pairs_numbers = [
                (a, b)  # ase < 3.28.0 does not support symbols for the get_rdf filter
                for i, a in enumerate(unique_elements)
                for b in unique_elements[i:]
            ]
            pairs = [(chemical_symbols[a], chemical_symbols[b]) for a, b in pairs_numbers]
        else:
            pairs_numbers = []
            for pair in pairs:
                a = atomic_numbers[pair[0]] if isinstance(pair[0], str) else pair[0]
                b = atomic_numbers[pair[1]] if isinstance(pair[1], str) else pair[1]
                pairs_numbers.append((a, b))

        if len(pairs_numbers) == 0:
            raise ValueError("No pairs specified.")

        # Select the last max_num_frames frames
        if max_num_frames is None:
            atoms_list = traj
        else:
            n = len(traj)
            atoms_list = [traj[i] for i in range(max(0, n - max_num_frames), n)]

        # Compute the RDF for each of the selected pairs
        rdf_results = {}
        with Pool(processes=n_workers) as pool:
            for elements_nr, pair in zip(pairs_numbers, pairs, strict=False):
                tasks = []
                for atoms in atoms_list:
                    positions = atoms.get_positions()
                    # Apply the cell constraints from the input argument
                    if cell_constraints is not None:
                        # Validate cell_constraints format
                        if (
                            not isinstance(cell_constraints, list)
                            or len(cell_constraints) != 3
                            or not all(isinstance(t, tuple) and len(t) == 2 for t in cell_constraints)
                        ):
                            raise ValueError(
                                "cell_constraints must be a list of three (min, max) tuples, one for each coordinate (x, y, z)."
                            )
                        # Apply the constraints
                        selected_atoms = np.array(
                            [
                                all(cell_constraints[i][0] <= pos[i] <= cell_constraints[i][1] for i in range(3))
                                for pos in positions
                            ]
                        )
                    else:
                        selected_atoms = np.ones(len(positions), dtype=bool)
                    tasks += [
                        (
                            positions[selected_atoms],
                            atoms.get_atomic_numbers()[selected_atoms],
                            atoms.get_cell(),
                            atoms.get_pbc(),
                            rmax,
                            nbins,
                            elements_nr,
                        )
                    ]
                results = pool.map(_rdf_worker, tasks) if n_workers > 1 else [_rdf_worker(task) for task in tasks]
                avg_rdf = np.mean([res[0] for res in results if not np.isnan(res[0]).any()], axis=0)
                std_rdf = np.std([res[0] for res in results if not np.isnan(res[0]).any()], axis=0)
                # Distances are the same for all frames, so they can be taken from the final frame
                rdf_results[pair] = (results[-1][1], avg_rdf, std_rdf)

        return rdf_results

    def _autocorr_fft(self, x: np.ndarray, nmax: int) -> np.ndarray:
        """Compute the autocorrelation function of a signal using FFT.

        Args:
            x (np.ndarray): Signal to compute the autocorrelation of.
            nmax (int): Maximum distance to compute the autocorrelation for.

        Returns:
            np.ndarray: Normalized autocorrelation function
        """
        n = len(x)
        nmax = min(nmax, n)
        f = np.fft.fft(x, n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[:nmax].real  # type: ignore
        norm = np.arange(n, n - nmax, -1)
        return acf / norm

    def compute_viscosity(
        self,
        T: float,
        eq_fraction: float = 0.1,
        traj_id: str | None = None,
        tmax_fs: list[int] | np.ndarray | int = 20000,
    ) -> tuple[np.ndarray | float, tuple[np.ndarray, np.ndarray]]:
        """Compute shear viscosity using the Green-Kubo relation. The timestep between frames has to be constant.

        Args:
            T (float): Temperature in K. The trajectory with the matching temperature is selected.
            traj_id (str, optional): Identifier for the trajectory, overrides T. Defaults to None.
            eq_fraction (float, optional): Fraction of the trajectory to consider as equilibrated. Defaults to 0.1.
            tmax_fs (int, optional): Maximum correlation time in femtoseconds. Defaults to 20000.

        Raises:
            ValueError: If the timestep between the frames is not constant.
            ValueError: If any of the tmax_fs values are non-positive.

        Returns:
            Tuple[float, Tuple[np.ndarray, np.ndarray]]: Viscosity in Pa s and the autocorrelation function and times:
                - "eta": Viscosity in Pa s
                - "(autocorrelation, times)": (autocorrelation function, times) in eV²/Å⁶ fs and fs
        """

        # Can only select based on temperature if the traj temperatures are provided
        traj, times = self._select_trajectory("nvt", T, traj_id)

        # Check only the equilibrated parts
        eq_indices = self._get_eq_indices(eq_fraction, times)
        times = times[eq_indices]

        # Ensure a constant timestep
        dt = times[1] - times[0]
        if not np.allclose(np.diff(times), dt):
            raise ValueError(
                f"The timestep between the frames is not constant ({np.unique(np.round(np.diff(times), 8))} fs occur)."
            )

        # Convert tmax_fs to an array
        tmax_fs = np.asarray(tmax_fs, dtype=float)

        if np.any(tmax_fs <= 0):
            raise ValueError("All tmax_fs values must be positive.")

        # Get the maximum difference in number of frames to compute the autocorrelation for (largest tmax_fs)
        nmax = len(times[times - times[0] <= np.max(tmax_fs)])

        # Get the stress tensors and extract the shear stress components
        traj = [traj[i] for i in eq_indices]
        stress_ts = np.array([atoms.get_stress() for atoms in traj], dtype=float)  # eV/Å³
        shear_stress = stress_ts[:, 3:]  # eV/Å³
        # Remove means to isolate equilibrium fluctuations
        shear_stress -= np.mean(shear_stress, axis=0)  # eV/Å³

        # Compute the average of the autocorrelation of the shear stress components
        ac_mean = np.mean(
            [self._autocorr_fft(shear_stress[:, i], nmax) for i in range(shear_stress.shape[1])],
            axis=0,
        )  # eV²/Å⁶
        ac_times = np.arange(ac_mean.size) * dt  # fs

        # Get the viscosity coefficient by integrating the autocorrelation function
        cumulative_integral = np.concatenate([[0.0], cumulative_trapezoid(ac_mean, ac_times)])  # eV²/Å⁶ fs
        V = np.mean([atoms.get_volume() for atoms in traj])  # Å³

        # Convert cumulative integral to viscosity for different tmax_fs (max autocorrelation times)
        eta_running = V * cumulative_integral / (units.kB / units.C * T)  # eV²/(Å³ J/K K) fs = eV²/(Å³ J) fs
        eta_running /= units.J**2 * 1e-15  # J/m³ s = Pa s
        eta = np.interp(
            tmax_fs,
            ac_times,
            eta_running,
        )
        return (eta, (ac_mean, ac_times + times[0]))

    def viscosity_vs_tmax_find_plateau(
        self,
        tmax_fs_list: list[float],
        eta_Pa_s_list: list[float],
        min_window_size_fs: float = 1000,
        std_threshold_Pa_s: float = 100,
        slope_threshold: float = 5e-2,
        min_consecutive_windows: int = 1,
    ) -> tuple:
        """Given different values for the viscosity (eta) from different maximal integration times (tmax_fs), finds the longest plateau of a minimum size (min_window_size_fs) and computes the mean and std on that plateau.

        Args:
            tmax_fs_list (list[float]): List of upper limits for the ACF integration times in femtoseconds.
            eta_Pa_s_list (list[float]): List of viscosity values corresponding to the tmax_fs_list in Pa s.
            min_window_size_fs (float): Minimum size of the window for local statistics in femtoseconds. Defaults to 1000.
            std_threshold_Pa_s (float): Threshold for the standard deviation of the viscosity within a window in Pa s to consider it to a plateau. Defaults to 100.
            slope_threshold (float): Maximum allowed slope (absolute value) of the viscosity in Pa s per femtosecond. Defaults to 5e-2.
            min_consecutive_windows (int, optional): Minimum number of consecutive windows that must satisfy the std and slope criteria to consider a plateau. Defaults to 1.

        Raises:
            ValueError: If tmax_fs_list or eta_Pa_s_list are not 1D arrays.
            ValueError: If tmax_fs_list and eta_Pa_s_list have different lengths.
            ValueError: If less than two data points are provided.
            ValueError: If tmax_fs_list is not regularly spaced.
            ValueError: If min_window_size_fs is smaller than two data points.
            ValueError: If min_window_size_fs is longer than the available tmax range.
            RuntimeError: If no plateau satisfying the criteria is found.

        Returns:
            tuple: A tuple containing:
            - eta_mean (float): Mean viscosity value within the plateau.
            - eta_std (float): Standard deviation of the viscosity within the plateau.
            - plateau_t (list[float]): List of tmax values corresponding to the plateau.
        """
        tmax = np.asarray(tmax_fs_list, dtype=float)  # fs
        eta = np.asarray(eta_Pa_s_list, dtype=float)  # Pa s

        # Input validations
        if tmax.ndim != 1 or eta.ndim != 1:
            raise ValueError("tmax_fs_list and eta_Pa_s_list must be 1D arrays.")
        if len(tmax) != len(eta):
            raise ValueError("tmax_fs_list and eta_Pa_s_list must have the same length.")
        if len(tmax) < 2:
            raise ValueError("At least two data points are required for tmax_list.")

        order = np.argsort(tmax)
        tmax = tmax[order]  # fs
        eta = eta[order]  # Pa s
        dt = tmax[1] - tmax[0]  # fs
        window_n = int(np.ceil(min_window_size_fs / dt))

        if not np.allclose(np.diff(tmax), dt):
            raise ValueError(f"tmax_list must be regularly spaced ({np.unique(np.round(np.diff(tmax), 8))} fs occur).")
        if window_n < 2:
            raise ValueError(f"min_window_size_fs={min_window_size_fs} must contain at least two data points.")
        if window_n > len(tmax):
            raise ValueError(
                f"min_window_size_fs={min_window_size_fs} is longer than the available tmax range ({tmax[-1] - tmax[0]} fs)."
            )
        if min_consecutive_windows < 1:
            raise ValueError(f"min_consecutive_windows={min_consecutive_windows} must be at least 1.")

        # Calculate all rolling statistics at once. Since tmax is regularly spaced, the least-squares slope has a constant denominator for every window
        windows = np.lib.stride_tricks.sliding_window_view(eta, window_n)
        window_means = windows.mean(axis=1)
        window_stds = windows.std(axis=1, ddof=1)
        offsets = np.arange(window_n, dtype=float)
        slope_denominator = np.sum((offsets - offsets.mean()) ** 2)
        window_slopes = (windows - window_means[:, None]) @ (offsets - offsets.mean()) / (dt * slope_denominator)
        acceptable = (window_stds <= std_threshold_Pa_s) & (np.abs(window_slopes) <= slope_threshold)
        # Find the first plateau
        consecutive = (
            np.convolve(
                acceptable.astype(int),
                np.ones(min_consecutive_windows, dtype=int),
                mode="valid",
            )
            == min_consecutive_windows
        )
        plateau_starts = np.flatnonzero(consecutive)
        if not plateau_starts.size:
            raise RuntimeError("No viscosity plateau satisfying the specified criteria was found.")
        start_window = plateau_starts[0]
        # Extend the plateau, as long as the local window remains acceptable
        end_window = start_window
        while end_window + 1 < len(acceptable) and acceptable[end_window + 1]:
            end_window += 1
        # Final statistics over the entire selected plateau
        plateau_t = tmax[start_window : end_window + window_n]
        plateau_eta = eta[start_window : end_window + window_n]
        plateau_mean = np.mean(plateau_eta)
        plateau_std = np.std(plateau_eta, ddof=1)
        return (plateau_mean, plateau_std, plateau_t)


# Example usage
if __name__ == "__main__":  # pragma: no cover
    print(
        "\nMinimalistic examples of the MoltenSaltAnalyzer class for a very short simulation of molten NaCl. The results are printed to the console:\n"
    )

    # Assumes the NPT and NVT trajectories have already been generated with the simulator (generate with simulator.py)
    base_dir = os.path.join("demo", "demo_simulation_results", "GRACE_1L_NaCl_super_short")
    npt_dir = os.path.join(base_dir, "NPT")
    nvt_dir = os.path.join(base_dir, "NVT")
    temps = [1100, 1150, 1200]
    npt_trajs = [os.path.join(npt_dir, f"npt_NaCl_{T}K.traj") for T in temps]
    nvt_trajs = [os.path.join(nvt_dir, f"nvt_NaCl_{T}K.traj") for T in temps]

    # Typically 0.1, but since the example trajectories contain only 10 frames, 0.6 is used
    EQ_FRAC = 0.6

    # Can be used for all calculations that use the trajectory files at 1100 K
    analyzer = MoltenSaltAnalyzer(npt_trajs, nvt_trajs, temps, temps)

    # ===================================================================================
    #   Equilibrium Density
    # ===================================================================================
    for temp in temps:
        density = analyzer.compute_eq_density(eq_fraction=EQ_FRAC, T=temp)
        print(f"Density of NaCl at {temp} K: {density:.3f} g/cm³")

    # ===================================================================================
    #   Thermal Expansion
    # ===================================================================================
    thm_exp_results = analyzer.compute_thermal_expansion(EQ_FRAC)
    print(f"Thermal expansion:  β = {thm_exp_results['thermal_expansion']:.6e} K⁻¹")

    # ===================================================================================
    #   Heat Capacity
    # ===================================================================================
    for temp in temps:
        heat_cap = analyzer.compute_heat_capacity_cv(T=temp, eq_fraction=EQ_FRAC)
        print(f"Heat capacity at {temp} K: c_v = {heat_cap:.6e} J/g/K")

    # ===================================================================================
    #   Diffusion Coefficient
    # ===================================================================================
    diff_coeffs = []
    for temp in temps:
        # Set up the analyzer for each of the NVT trajectories to get the diffusion coefficient there
        diff_coeff = analyzer.compute_diffusion_coefficient(T=temp)
        print(f"Diffusion coefficient at {temp} K: D = {diff_coeff:.6e} Å²/fs")
        diff_coeffs.append(diff_coeff)
    # Get the activation energy
    diffusion_results = analyzer.fit_arrhenius(temps, diff_coeffs)
    print(
        f"Arrhenius parameters for the self-diffusion of NaCl: Ea = {diffusion_results['Ea']:.6e} J/mol, D0 = {diffusion_results['D0']:.6e} Å²/fs"
    )

    # ===================================================================================
    #   RDF
    # ===================================================================================
    for temp in temps:
        rdf_data = analyzer.compute_rdf(T=temp, max_num_frames=10, pairs=[(11, 11)], nbins=10)
        print(
            f"Radial distribution function for Na-Na at {temp} K: g(r) = {np.round(rdf_data[(11, 11)][1], 2)}... at distances {rdf_data[(11, 11)][0]}... Å"
        )

    # ===================================================================================
    #   Viscosity
    # ===================================================================================
    for temp in temps:
        viscosity, (autocorr_mean, autocorr_times) = analyzer.compute_viscosity(T=temp)
        # autocorr_mean and autocorr_times can be used to check that the plateau of the autocorrelation function reaches tmax_fs
        print(f"Viscosity at {temp} K: η = {viscosity:.6e} Pa·s")
