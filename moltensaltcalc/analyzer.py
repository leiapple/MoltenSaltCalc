import os
from typing import Tuple

import numpy as np
from ase import units
from ase.geometry.rdf import get_rdf
from ase.io import Trajectory


class MoltenSaltAnalyzer:
    """
    Class for analyzing molten salt simulation results.
    """

    def __init__(
        self,
        traj_files_npt: list[str] | str | None = None,
        traj_files_nvt: list[str] | str | None = None,
        temperatures: list[float] | None = None,
    ):
        """Initialize the analyzer with the trajectories and the always used properties

        Args:
            traj_files_npt (list, str, optional): Path to the NPT trajectory file(s). Defaults to None.
            traj_files_nvt (list, str, optional): Path to the NVT trajectory file(s). Defaults to None.
            temperatures (list, optional): List of temperatures in K. Must be provided if multiple trajectory files are provided. Defaults to None.

        Raises:
            ValueError: If the number of trajectory files is not equal to the number of temperatures.
            FileNotFoundError: If the trajectory file(s) does not exist.

        Defines:
            self.trajs_npt (list): List of Trajectory objects for the NPT trajectories
            self.trajs_nvt (list): List of Trajectory objects for the NVT trajectories
            self.times_fs_npt (np.ndarray): Array of times in fs for the NPT trajectories
            self.times_fs_nvt (np.ndarray): Array of times in fs for the NVT trajectories
            self.temperatures (list): List of temperatures in K
        """
        self.trajs_npt = None
        self.trajs_nvt = None
        self.times_fs_npt = None
        self.times_fs_nvt = None
        self.temperatures = temperatures

        if isinstance(traj_files_npt, str):
            traj_files_npt = [traj_files_npt]
        if isinstance(traj_files_nvt, str):
            traj_files_nvt = [traj_files_nvt]

        if traj_files_npt is not None:
            if temperatures is None or len(traj_files_npt) != len(temperatures):
                raise ValueError(
                    f"Number of trajectory files and temperatures must be equal."
                )
            self.trajs_npt, self.times_fs_npt = [], []
            for traj_file in traj_files_npt:
                if not os.path.exists(traj_file):
                    raise FileNotFoundError(f"Trajectory file {traj_file} not found.")
                traj = Trajectory(traj_file)
                self.trajs_npt.append(traj)
                times = np.array([atoms.info["time_fs"] for atoms in traj])
                self.times_fs_npt.append(times)

        if traj_files_nvt is not None:
            if temperatures is None or len(traj_files_nvt) != len(temperatures):
                raise ValueError(
                    f"Number of trajectory files and temperatures must be equal."
                )
            self.trajs_nvt, self.times_fs_nvt = [], []
            for traj_file in traj_files_nvt:
                if not os.path.exists(traj_file):
                    raise FileNotFoundError(f"Trajectory file {traj_file} not found.")
                traj = Trajectory(traj_file)
                self.trajs_nvt.append(traj)
                times = np.array([atoms.info["time_fs"] for atoms in traj])
                self.times_fs_nvt.append(times)

    def _select_trajectory(
        self, preferred_type: str, T: float | None = None
    ) -> Tuple[Trajectory, np.ndarray]:
        """Selects the trajectory based on the preferred type.

        Args:
            preferred_type (str): Trajectory type to be used if both are available. Either "npt" or "nvt".
            T (float, optional): Temperature in K. If provided, the trajectory with the matching temperature is selected. Defaults to None, which means the first trajectory is selected.

        Raises:
            ValueError: If no trajectory files are provided from the initializer.
            ValueError: If the temperature is not in the list self.temperatures.

        Returns:
            Tuple[Trajectory, np.ndarray]: Trajectory and simulation times in fs.
        """
        if self.trajs_npt is None and self.trajs_nvt is None:
            raise ValueError("No trajectory files provided.")
        if T is not None and T not in self.temperatures:
            raise ValueError(f"Temperature {T} not found in the list of temperatures.")

        # Select the trajectory with the matching temperature if provided, otherwise the first
        traj_idx = 0 if T is None else self.temperatures.index(T)

        # Select the trajectory based on availability and preferred type
        if self.trajs_npt is not None and (
            self.trajs_nvt is None or preferred_type == "npt"
        ):
            traj = self.trajs_npt[traj_idx]
            times = self.times_fs_npt[traj_idx]
        else:
            traj = self.trajs_nvt[traj_idx]
            times = self.times_fs_nvt[traj_idx]

        return traj, times

    def _get_eq_times(self, eq_fraction: float, times_fs: np.ndarray) -> np.ndarray:
        """Gets the indices of the simulation times later than 1-eq_fraction of the total simulation time.

        Args:
            eq_fraction (float): Fraction of the total simulation time from the end of the simulation
            times_fs (np.ndarray): Times in femtoseconds

        Returns:
            np.ndarray: Indices of the simulation times later than 1-eq_fraction of the total simulation time.
        """
        eq_times = np.where(times_fs >= np.max(times_fs) * (1 - eq_fraction))[0]
        return eq_times

    def compute_density_vs_time(
        self, T: float | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the density from the trajectory file. If both NPT and NVT trajectories are loaded, the density is computed from the NPT trajectory.

        Args:
            T (float, optional): Temperature in K. If provided, the density is computed from the trajectory with the matching temperature. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Densities in g/cm³ and times in fs
        """
        traj, times = self._select_trajectory("npt", T)
        masses = traj[0].get_masses().sum() * units._amu * 1e3  # g
        volumes = np.array([atoms.get_volume() for atoms in traj]) * 1e-24  # cm³
        densities = masses / volumes  # g/cm³
        return densities, times

    def compute_eq_density(
        self, eq_fraction: float = 0.1, T: float | None = None
    ) -> float:
        """
        Compute the density after equilibration (last x time% of the trajectory). If both NPT and NVT trajectories are loaded, the density is computed from the NPT trajectory.

        Args:
            eq_fraction (float, optional): Final fraction of the simulation time to be considered as equilibrium. Defaults to 0.1.
            T (float, optional): Temperature in K. If provided, the density is computed from the trajectory with the matching temperature. Defaults to None.

        Returns:
            float: Density in g/cm³
        """
        if eq_fraction > 1.0 or eq_fraction < 0.0:
            raise ValueError("eq_fraction must be between 0 and 1.")

        # Get the densities
        densities, times_fs = self.compute_density_vs_time(T)
        eq_times = self._get_eq_times(eq_fraction, times_fs)
        eq_density = np.mean(densities[eq_times])

        return eq_density

    def compute_thermal_expansion(self, eq_fraction: float = 0.1) -> dict:
        """Compute the thermal expansion coefficient from the initialized trajectory files and temperatures.

        Args:
            eq_fraction (float, optional): Final fraction of the simulation time to be considered as equilibrium. Defaults to 0.1.

        Returns:
            dict:  Thermal expansion results:
                - "temperatures": List of temperatures
                - "eq_vols": Equilibrium volumes in Å³ for each temperature
                - "eq_vols_norm": Equilibrium volumes normalized to the mean volume
                - "fit": Fit parameters
                - "fit_line": Fit line
                - "thermal_expansion": Thermal expansion coefficient in 1/K
        """
        # Get the equilibrium volumes for each trajectory file
        eq_vols = np.zeros(len(self.trajs_npt))
        for i, traj in enumerate(self.trajs_npt):
            volumes = np.array([atoms.get_volume() for atoms in traj])  # Å³
            times = np.array([atoms.info["time_fs"] for atoms in traj])  # fs
            eq_times = self._get_eq_times(eq_fraction, times)
            eq_vol = np.mean(volumes[eq_times])  # Å³
            eq_vols[i] = eq_vol

        # Fit linear thermal expansion to the volumes normalized by the mean volume
        eq_vols_norm = eq_vols / np.mean(eq_vols)
        fit = np.polyfit(self.temperatures, eq_vols_norm, 1)
        fit_line = np.polyval(fit, self.temperatures)

        return {
            "temperatures": self.temperatures,
            "eq_vols": eq_vols,
            "eq_vols_norm": eq_vols_norm,
            "fit": fit,
            "fit_line": fit_line,
            "thermal_expansion": fit[0],
        }

    def compute_heat_capacity(self, T: float, eq_fraction: float = 0.1) -> float:
        """Compute heat capacity from total energy fluctuations. If both NPT and NVT trajectories are loaded, the heat capacity is computed from the NVT trajectory.

        Args:
            T (float): Temperature in K.
            eq_fraction (float, optional): Final fraction of the simulation time to be considered as equilibrium. Defaults to 0.1.

        Returns:
            float: Heat capacity in J/g/K
        """
        # Can only select based on temperature if the traj temperatures are provided
        traj, times = self._select_trajectory(
            "nvt", None if self.temperatures is None else T
        )
        eq_times = self._get_eq_times(eq_fraction, times)
        U = np.array([atoms.get_total_energy() for atoms in traj])[eq_times]
        # Compute the variation and get the approximate heat capacity C
        var_U = np.var(U, ddof=1) * units._e**2  # J²
        m_tot = traj[0].get_masses().sum() * units._amu * 1e3  # g
        C = var_U / (units._k * T**2 * m_tot)  # J/g/K
        return C

    def compute_diffusion_coefficient(self, T: float | None = None) -> float:
        """Compute diffusion coefficient from mean squared displacement. If both NPT and NVT trajectories are loaded, the diffusion coefficient is computed from the NVT trajectory.

        Args:
            T (float, optional): Temperature in K. If provided, the diffusion coefficient is computed from the trajectory with the matching temperature. Defaults to None.

        Returns:
            float: Diffusion coefficient in Å²/fs
        """
        traj, times = self._select_trajectory("nvt", T)
        # Get the positions relative to the center of mass
        positions = np.array(
            [atoms.get_positions() - atoms.get_center_of_mass() for atoms in traj]
        )  # Å
        r0 = positions[0]  # Å
        # Compute the mean square displacements without variation of time origins
        msd = np.mean(np.sum((positions - r0) ** 2, axis=2), axis=1)  # Å²
        slope, _ = np.polyfit(times, msd, 1)
        D = slope / 6.0  # Å²/fs

        return D

    def fit_arrhenius(self, temperatures: list, diffusion_coeffs: list) -> dict:
        """Fit Arrhenius law to diffusion coefficients and temperatures.

        Args:
            temperatures (list): Temperatures in K.
            diffusion_coeffs (list): Diffusion coefficients corresponding to the temperatures in Å²/fs.

        Returns:
            dict: Arrhenius parameters:
                - "Ea": Activation energy in J/mol
                - "D0": Exponential prefactor of the Arrhenius law in Å²/fs
                - "slope": Slope of the Arrhenius law
                - "intercept": Intercept of the Arrhenius law
        """

        # Linearize: ln(D) = ln(D0) - Ea/(R*T)
        x = 1.0 / np.array(temperatures)  # 1/K
        y = np.log(diffusion_coeffs)  # ln(Å²/fs)

        m, b = np.polyfit(x, y, 1)  # m = slope, b = intercept
        Ea = -m * units._k * units.mol  # J/mol
        D0 = np.exp(b)  # Å²/fs

        return {"Ea": Ea, "D0": D0, "slope": m, "intercept": b}

    def compute_rdf(
        self,
        max_num_frames: int = 10000,
        rmax: float = 6.0,
        nbins: int = 100,
        pairs: list[tuple[int, int]] | None = None,
        T: float | None = None,
    ) -> dict:
        """Compute radial distribution functions. If both NPT and NVT trajectories are loaded, the RDF is computed from the NVT trajectory.

        Args:
            max_num_frames (int, optional): Maximum number of trajectory frames to analyze and average over. The frames are selected from the end of the simulation. Defaults to 1000.
            rmax (float, optional): Maximum distance (Å) to consider. Defaults to 6.0.
            nbins (int, optional): Number of bins for the RDF. Defaults to 100.
            pairs (list[tuple] | None, optional): Atom pairs in terms of atomic numbers to compute the RDF for. Defaults to None which means all unique pairs in the system are analyzed.
            T (float, optional): Temperature in K. If provided, the RDF is computed from the trajectory with the matching temperature. Defaults to None.

        Raises:
            ValueError: If no pairs are specified.

        Returns:
            dict: Dictionary with RDF results:
                - "(atomic number, atomic number)": (distances, avg_rdf) for each pair. Distances are in Å and avg_rdf is unitless (normalized).
        """
        traj, _ = self._select_trajectory("nvt", T)

        # Get all unique atomic pairs if not specified
        if pairs is None:
            atm_nums = traj[0].get_atomic_numbers()
            unique_elements = sorted(set(atm_nums))
            pairs = [
                (a, b)  # ase < 3.28.0 does not support symbols for the get_rdf filter
                for i, a in enumerate(unique_elements)
                for b in unique_elements[i:]
            ]
        if len(pairs) == 0:
            raise ValueError("No pairs specified.")

        # Select the last max_num_frames frames
        n = len(traj)
        atoms_list = [traj[i] for i in range(max(0, n - max_num_frames), n)]

        # Compute the RDF for each of the selected pairs
        rdf_results = {}
        for pair in pairs:
            rdfs = []
            for atoms in atoms_list:
                rdf, distances = get_rdf(atoms, rmax, nbins, elements=pair)
                rdfs.append(rdf)
            if rdfs:
                avg_rdf = np.mean(rdfs, axis=0)
                # Distances are the same for all frames, so they can be taken from the final frame
                rdf_results[pair] = (distances, avg_rdf)

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
        f = np.fft.fft(x, n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[:nmax].real
        norm = np.arange(n, n - nmax, -1)
        return acf / norm

    def compute_viscosity(
        self,
        T: float | None,
        tmax_fs: int = 20000,
    ) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
        """Compute shear viscosity using Green-Kubo relation. The timestep between frames has to be constant.

        Args:
            T (int): Temperature in K.
            tmax_fs (int, optional): Maximum correlation time in femtoseconds. Defaults to 20000.

        Raises:
            ValueError: If the timestep between the frames is not constant.

        Returns:
            Tuple[float, Tuple[np.ndarray, np.ndarray]]: Viscosity in Pa s and the autocorrelation function and times:
                - "eta": Viscosity in Pa s
                - "(autocorrelation, times)": (autocorrelation function, times) in eV²/Å⁶ fs and fs
        """

        # Can only select based on temperature if the traj temperatures are provided
        traj, times = self._select_trajectory(
            "nvt", None if self.temperatures is None else T
        )

        # Ensure a constant timestep
        dt = traj[1].info["time_fs"] - traj[0].info["time_fs"]  # fs
        if not np.allclose(np.diff(times), dt):
            raise ValueError(
                f"The timestep between the frames is not constant ({np.unique(np.round(np.diff(times), 8))} fs occur)."
            )
        # Get the maximum difference in number of frames to compute the autocorrelation for
        nmax = min(len(times), int(np.ceil(tmax_fs / dt)))

        # Get the stress tensors and extract the shear stress components
        stress_ts = np.array(
            [atoms.get_stress() for atoms in traj], dtype=float
        )  # eV/Å³
        shear_stress = stress_ts[:, 3:]  # eV/Å³
        # Remove means to isolate equilibrium fluctuations
        shear_stress -= np.mean(shear_stress, axis=0)  # eV/Å³

        # Compute the average of the autocorrelation of the shear stress components
        ac_mean = np.mean(
            [
                self._autocorr_fft(shear_stress[:, i], nmax)
                for i in range(shear_stress.shape[1])
            ],
            axis=0,
        )  # eV²/Å⁶
        ac_times = np.arange(ac_mean.size) * dt  # fs

        # Get the viscosity coefficient by integrating the autocorrelation function
        integral = np.trapezoid(ac_mean, ac_times)  # eV²/Å⁶ fs
        V = np.mean([atoms.get_volume() for atoms in traj])  # Å³
        eta = V * integral / (units._k * T)  # eV²/(Å³ J/K K) fs = eV²/(Å³ J) fs
        eta *= units._e**2 / 1e-15  # J/m³ s = Pa s

        return (eta, (ac_mean, ac_times))


# Example usage
if __name__ == "__main__":  # pragma: no cover
    print(
        "\nMinimalistic examples of the MoltenSaltAnalyzer class for a very short simulation of molten NaCl. The results are printed to the console:\n"
    )

    # Assumes the NPT and NVT trajectories have already been generated with the simulator (generate with simulator.py)
    base_dir = os.path.join(
        "demo", "demo_simulation_results", "GRACE_1L_NaCl_super_short"
    )
    npt_dir = os.path.join(base_dir, "NPT")
    nvt_dir = os.path.join(base_dir, "NVT")
    temperatures = [1100, 1150, 1200]
    npt_trajs = [os.path.join(npt_dir, f"npt_NaCl_{T}K.traj") for T in temperatures]
    nvt_trajs = [os.path.join(nvt_dir, f"nvt_NaCl_{T}K.traj") for T in temperatures]

    # Typically 0.1, but since the example trajectories contain only 10 frames, 0.6 is used
    eq_frac = 0.6

    # Can be used for all calculations that use the trajectory files at 1100 K
    analyzer = MoltenSaltAnalyzer(npt_trajs, nvt_trajs, temperatures)

    # ===================================================================================
    #   Equilibrium Density
    # ===================================================================================
    for T in temperatures:
        density = analyzer.compute_eq_density()
        print(f"Density of NaCl at {T} K: {density:.3f} g/cm³")

    # ===================================================================================
    #   Thermal Expansion
    # ===================================================================================
    thm_exp_results = analyzer.compute_thermal_expansion(eq_frac)
    print(f"Thermal expansion:  β = {thm_exp_results['thermal_expansion']:.6e} K⁻¹")

    # ===================================================================================
    #   Heat Capacity
    # ===================================================================================
    for T in temperatures:
        heat_cap = analyzer.compute_heat_capacity(T, eq_frac)
        print(f"Heat capacity at {T} K: C = {heat_cap:.6e} J/g/K")

    # ===================================================================================
    #   Diffusion Coefficient
    # ===================================================================================
    diffusion_coeffs = []
    for T in temperatures:
        # Set up the analyzer for each of the NVT trajectories to get the diffusion coefficient there
        diff_coeff = analyzer.compute_diffusion_coefficient(T)
        print(f"Diffusion coefficient at {T} K: D = {diff_coeff:.6e} Å²/fs")
        diffusion_coeffs.append(diff_coeff)
    # Get the activation energy
    diffusion_results = analyzer.fit_arrhenius(temperatures, diffusion_coeffs)
    print(
        f"Arrhenius parameters for the self-diffusion of NaCl: Ea = {diffusion_results['Ea']:.6e} J/mol, D0 = {diffusion_results['D0']:.6e} Å²/fs"
    )

    # ===================================================================================
    #   RDF
    # ===================================================================================
    for T in temperatures:
        rdf_data = analyzer.compute_rdf(10, pairs=[(11, 11)], nbins=10, T=T)
        print(
            f"Radial distribution function for Na-Na at {T} K: g(r) = {np.round(rdf_data[(11, 11)][1], 2)}... at distances {rdf_data[(11, 11)][0]}... Å"
        )

    # ===================================================================================
    #   Viscosity
    # ===================================================================================
    for T in temperatures:
        viscosity, (ac_mean, ac_times) = analyzer.compute_viscosity(T)
        # ac_mean and ac_times can be used to check that the plateau of the autocorrelation function reaches tmax_fs
        print(f"Viscosity at {T} K: η = {viscosity:.6e} Pa·s")
