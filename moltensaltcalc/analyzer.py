import os

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms, units
from ase.geometry.analysis import Analysis
from ase.io import Trajectory


class MoltenSaltAnalyzer:
    """
    Class for analyzing molten salt simulation results.
    """

    def __init__(self):
        """Initialize the analyzer."""
        pass

    def compute_density(self, traj_file):
        """
        Compute density from trajectory file.

        Parameters:
        -----------
        traj_file : str
            Path to trajectory file

        Returns:
        --------
        density : float
            Density in g/cm³
        """
        traj = Trajectory(traj_file)

        # Use last 10% for equilibrium
        volumes = [atoms.get_volume() for atoms in traj]
        equilibrium_volume = np.mean(volumes[-int(len(volumes) * 0.1) :])

        masses = traj[0].get_masses().sum() * units._amu * 1e3  # g
        density = masses / (equilibrium_volume * 1e-24)  # g/cm³

        return density

    def compute_plot_density_vs_time(self, traj_file, title, fig_path):
        """
        Compute and plot density evolution during simulation.

        Parameters:
        -----------
        traj_file : str
            Path to trajectory file
        title : str
            Plot title
        fig_path : str
            Filename for saving the plot

        Returns:
        -----------
        densities: list
            List of densities at each time step
        """
        traj = Trajectory(traj_file)

        volumes = [atoms.get_volume() for atoms in traj]
        masses = traj[0].get_masses().sum() * units._amu * 1e3  # g

        densities = masses / (np.array(volumes) * 1e-24)

        # Plot density evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(densities)
        ax.set_title(title)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Density (g/cm³)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_path)

        return densities

    def compute_thermal_expansion(self, npt_dir, salt_name, temperatures):
        """
        Compute thermal expansion coefficient.

        Parameters:
        -----------
        npt_dir : str
            Directory with NPT trajectories
        salt_name : str
            Name of the salt
        temperatures : list
            List of temperatures

        Returns:
        --------
        dict
            Dictionary with thermal expansion results
        """
        box_lengths = []
        densities = []

        for T in temperatures:
            traj_file = os.path.join(npt_dir, f"npt_{salt_name}_{T}K.traj")
            if not os.path.exists(traj_file):
                continue

            traj = Trajectory(traj_file)
            volumes = [atoms.get_volume() for atoms in traj]
            eq_vol = np.mean(volumes[-int(0.1 * len(volumes)) :])
            box_lengths.append(eq_vol ** (1 / 3))
            densities.append(self.compute_density(traj_file))

        box_lengths = np.array(box_lengths)
        densities = np.array(densities)
        temperatures = np.array(temperatures)

        # Fit linear thermal expansion
        fit = np.polyfit(temperatures, box_lengths / box_lengths[0], 1)
        T_fit = np.linspace(min(temperatures), max(temperatures), 100)
        fit_line = np.polyval(fit, T_fit)

        return {
            "temperatures": temperatures,
            "box_ratios": box_lengths / box_lengths[0],
            "fit": fit,
            "T_fit": T_fit,
            "fit_line": fit_line,
            "thermal_expansion": fit[0],
        }

    def compute_heat_capacity(self, traj_file, T):
        """
        Compute heat capacity from enthalpy fluctuations.

        Parameters:
        -----------
        traj_file : str
            Path to trajectory file
        T : float
            Temperature (K)

        Returns:
        --------
        Cp : float
            Heat capacity in J/g/K
        """
        traj = Trajectory(traj_file)

        # Compute enthalpy: H = U + PV
        H = np.array(
            [
                atoms.get_kinetic_energy()
                + atoms.get_potential_energy()
                + units.bar
                * atoms.get_volume()
                * 1e-30
                / units.eV  # TODO: Check if this is correct: units.eV is 1, but he probably wanted a conversion factor here?
                # TODO: Where is P here? I only see units.bar, but not the actual pressure?
                for atoms in traj
            ]
        )

        # Use last 10% for equilibrium
        H_equil = H[-int(0.1 * len(H)) :]
        var_H = np.var(H_equil, ddof=1) * (units._e**2)  # Convert to J²

        mass_total = traj[0].get_masses().sum() * units._amu * 1e3  # g

        Cp = var_H / (units._k * T**2 * mass_total)  # J/g/K

        return Cp

    def compute_diffusion_coefficient(self, traj_file):
        """
        Compute diffusion coefficient from mean squared displacement.

        Parameters:
        -----------
        traj_file : str
            Path to trajectory file

        Returns:
        --------
        D : float
            Diffusion coefficient in Å²/fs
        """
        traj = Trajectory(traj_file)

        positions = np.array([atoms.get_positions() for atoms in traj])
        nsteps, natoms, _ = positions.shape

        r0 = positions[0]
        msd = np.mean(np.sum((positions - r0) ** 2, axis=2), axis=1)

        times = np.arange(nsteps) * 1.0 * units.fs
        slope, _ = np.polyfit(times, msd, 1)

        D = slope / 6.0  # Å²/fs

        return D

    def fit_arrhenius(self, temperatures, diffusion_coeffs):
        """
        Fit Arrhenius law to diffusion coefficients.

        Parameters:
        -----------
        temperatures : list
            List of temperatures
        diffusion_coeffs : list
            List of diffusion coefficients

        Returns:
        --------
        dict
            Dictionary with Arrhenius parameters
        """

        temp = np.array(temperatures)
        D = np.array(diffusion_coeffs)

        # Linearize: ln(D) = ln(D0) - Ea/(R*T)
        x = 1.0 / temp
        y = np.log(D)

        m, b = np.polyfit(x, y, 1)
        Ea = -m * units._k * units.mol  # J/mol
        D0 = np.exp(b)  # Å²/fs

        return {"Ea": Ea, "D0": D0, "slope": m, "intercept": b}

    def compute_rdf(self, traj_file, rmax=6, nbins=100, pairs=None):
        """
        Compute radial distribution functions.

        Parameters:
        -----------
        traj_file : str
            Path to trajectory file
        rmax : float
            Maximum distance (Å)
        nbins : int
            Number of bins
        pairs : list of tuples
            Pairs to compute RDF for

        Returns:
        --------
        dict
            Dictionary with RDF results
        """
        traj = Trajectory(traj_file)
        symbols = traj[0].get_chemical_symbols()

        atoms_list = [
            Atoms(
                symbols=symbols,
                positions=atoms.get_positions(),
                cell=atoms.get_cell(),
                pbc=True,
            )
            for atoms in traj
        ]

        ana = Analysis(atoms_list)

        unique_elements = sorted(set(symbols))
        if pairs is None:
            pairs = [
                (a, b)
                for i, a in enumerate(unique_elements)
                for b in unique_elements[i:]
            ]

        rdf_results = {}
        for pair in pairs:
            rdfs = ana.get_rdf(rmax=rmax, nbins=nbins, elements=pair)
            if rdfs:
                avg_rdf = np.mean(rdfs, axis=0)
                distances = np.linspace(0, rmax, nbins)
                rdf_results[pair] = (distances, avg_rdf)

        return rdf_results

    def plot_rdf(self, traj_file, title="Radial Distribution Function"):
        """
        Plot radial distribution functions.

        Parameters:
        -----------
        traj_file : str
            Path to trajectory file
        title : str
            Plot title
        """
        rdf_data = self.compute_rdf(traj_file)

        colors = ["midnightblue", "darkorange", "crimson", "green", "purple"]

        plt.figure(figsize=(8, 6))

        for i, ((pair, (distances, avg_rdf)), color) in enumerate(
            zip(rdf_data.items(), colors[: len(rdf_data)])
        ):
            peak_index = np.argmax(avg_rdf)
            peak_r = distances[peak_index]

            plt.plot(
                distances,
                avg_rdf,
                label=f"{pair[0]}-{pair[1]} (peak: {peak_r:.2f} Å)",
                linewidth=2,
                color=color,
            )
            plt.axvline(peak_r, linestyle="--", color=color, alpha=0.5)

        plt.xlabel("Distance (Å)")
        plt.ylabel("g(r)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def compute_viscosity(self, traj_file, T, tmax_fs=None):
        """
        Compute viscosity using Green-Kubo relation.

        Parameters:
        -----------
        traj_file : str
            Path to trajectory file
        T : float
            Temperature (K)
        tmax_fs : float
            Maximum correlation time (fs)

        Returns:
        --------
        eta : float
            Viscosity in Pa·s
        """
        traj = Trajectory(traj_file)

        stress_ts = np.array(
            [atoms.get_stress(voigt=False) for atoms in traj], dtype=float
        )
        nframes = stress_ts.shape[0]

        # Extract shear components
        p_xy = stress_ts[:, 0, 1].copy()
        p_xz = stress_ts[:, 0, 2].copy()
        p_yz = stress_ts[:, 1, 2].copy()

        # Remove means
        for comp in [p_xy, p_xz, p_yz]:
            comp -= np.mean(comp)

        # Convert eV/A^3 to Pa = J/m³
        conv = units._e / 1e-30
        p_xy *= conv
        p_xz *= conv
        p_yz *= conv

        def autocorr(x, nmax):
            n = len(x)
            corr = np.correlate(x, x, mode="full")[n - 1 : n - 1 + nmax]
            norm = np.arange(n, n - nmax, -1)
            return corr / norm

        if tmax_fs is None:
            nmax = nframes
        else:
            nmax = min(nframes, int(np.ceil(tmax_fs / 1e-15)))

        ac_mean = (
            autocorr(p_xy, nmax) + autocorr(p_xz, nmax) + autocorr(p_yz, nmax)
        ) / 3.0

        dt = 1.0 * 1e-15  # TODO: This does not necessarily hold!
        times = np.arange(ac_mean.size) * dt

        V = np.mean([atoms.get_volume() for atoms in traj]) * 1e-30
        integral = np.trapz(ac_mean, times)

        eta = V * integral / (units._k * T)

        return eta

    def analyze_multiple_runs(self, base_dir, temperatures):
        """
        Analyze multiple simulation runs.

        Parameters:
        -----------
        base_dir : str
            Base directory with simulations
        temperatures : list
            List of temperatures

        Returns:
        --------
        dict
            Dictionary with all analysis results
        """
        results = {
            "temperatures": temperatures,
            "densities": [],
            "heat_capacities": [],
            "diffusion_coeffs": [],
            "viscosities": [],
        }

        nvt_dir = os.path.join(base_dir, "NVT")

        for T in temperatures:
            traj_file = os.path.join(nvt_dir, f"nvt_{T}K.traj")

            if os.path.exists(traj_file):
                results["densities"].append(self.compute_density(traj_file))
                results["heat_capacities"].append(
                    self.compute_heat_capacity(traj_file, T)
                )
                results["diffusion_coeffs"].append(
                    self.compute_diffusion_coefficient(traj_file)
                )
                results["viscosities"].append(self.compute_viscosity(traj_file, T))
            else:
                results["densities"].append(np.nan)
                results["heat_capacities"].append(np.nan)
                results["diffusion_coeffs"].append(np.nan)
                results["viscosities"].append(np.nan)

        return results


# Example usage
if __name__ == "__main__":
    # Assumes the NPT and NVT trajectories have already been generated with the simulator
    run_folder = os.path.join("test_sim", "GRACE_1L_NaCl_long")
    npt_dir = os.path.join(os.getcwd(), run_folder, "NPT")
    nvt_dir = os.path.join(os.getcwd(), run_folder, "NVT")
    salts = {"NaCl": [1100, 1150, 1200][:1]}

    analyzer = MoltenSaltAnalyzer()

    # Ensure the plot directory exists
    os.makedirs(os.path.join("test_sim", "plots"), exist_ok=True)

    for salt_name, temps in salts.items():
        for T in temps:
            npt_traj = os.path.join(npt_dir, f"npt_{salt_name}_{T}K.traj")
            nvt_traj = os.path.join(nvt_dir, f"nvt_{salt_name}_{T}K.traj")

            # ===================================================================================
            #   Density vs. Time
            # ===================================================================================
            densities = analyzer.compute_plot_density_vs_time(
                npt_traj,
                title=f"Density Evolution — {salt_name} — {T}K",
                fig_path=os.path.join(
                    "test_sim", "plots", f"density_evolution_{salt_name}_{T}K.png"
                ),
            )

        # ===================================================================================
        #   Thermal Expansion
        # ===================================================================================
        result = analyzer.compute_thermal_expansion(npt_dir, salt_name, temps)
        print(f"Thermal expansion:  β = {result['thermal_expansion']:.6e} 1/K")
