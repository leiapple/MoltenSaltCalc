"""MoltenSaltSimulator class for building and running molecular dynamics simulations."""

import importlib
import os
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.data import atomic_masses, atomic_numbers
from ase.io import Trajectory
from ase.md.andersen import Andersen
from ase.md.bussi import Bussi
from ase.md.langevin import Langevin

# from ase.md.nose_hoover_chain import MaskedMTKNPT
from ase.md.melchionna import MelchionnaNPT
from ase.md.nose_hoover_chain import MTKNPT, NoseHooverChainNVT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from scipy.spatial.distance import cdist

from moltensaltcalc.model_discovery import discover_models
from moltensaltcalc.model_errors import (
    format_model_error,
    format_unknown_model_error,
)
from moltensaltcalc.registry import MODEL_REGISTRY


class MoltenSaltSimulator:
    """Class for building molten salt systems and running molecular dynamics simulations supported by energy estimates from uMLIPs."""

    def __init__(
        self,
        model_name: str,
        model_parameters: dict | None = None,
        device: str = "cuda",
    ):
        """Initialize the simulator with a specific ML potential.

        Args:
            model_name (str): Which MLIP to use.
            model_parameters (dict | None, optional): Parameters for the MLIP. Defaults to None.
            device (str, optional): Which device to use for the calculations, select from "cpu" and "cuda". Defaults to "cuda".
        """
        self.device = device
        model_name = model_name.lower()
        self.calc = None
        self._set_calculator(model_name, model_parameters)

    def _lazy_import_model(self, model_name: str):
        """Imports the model module, triggering its registration into MODEL_REGISTRY.

        Args:
            model_name (str): Input model name.

        Raises:
            ImportError: If the model name is not integrated in the package.
            ImportError: If the model could not be imported.
        """
        try:
            importlib.import_module(f"moltensaltcalc.models.{model_name}")
        except ModuleNotFoundError as e:
            raise ImportError(f"Model module 'moltensaltcalc.models.{model_name}' not found.\n") from e

        except ImportError as e:
            raise ImportError(
                f"Model '{model_name}' could not be imported.\nThis may be due to missing dependencies.\nOriginal error: {repr(e)}"
            ) from e

    def _set_calculator(self, model_name: str, model_parameters: dict | None):
        """Sets the uMLIP calculator for the energy and forces prediction.

        Args:
            model_name (str): Name of the model.
            model_parameters (dict | None): Parameters to be passed to the model.

        Raises:
            ValueError: If the model name provided is not available.
            ValueError: If the calculator could not be setup (e.g. unavailable GPU).
            RuntimeError: If the calculator doesn't contain the model (e.g. wrong parameters).
        """
        model_name = model_name.lower()
        model_parameters = dict(model_parameters or {})

        discoverable_models = discover_models()
        try:
            self._lazy_import_model(model_name)
        except ImportError as e:
            raise ValueError(format_unknown_model_error(model_name, discoverable_models)) from e

        # Registry check (system integrity)
        if model_name not in MODEL_REGISTRY:
            raise RuntimeError(f"Model '{model_name}' was imported but did not register itself.")

        # Instantiate
        try:
            calc = MODEL_REGISTRY[model_name](model_parameters, device=self.device)
        except ImportError as e:
            # Dependency issue
            raise RuntimeError(
                f"Missing dependency for model '{model_name}'.\n\n{e}\n\n=> Please install the required package (version)."
            ) from e

        except ValueError as e:
            # Parameter error
            raise ValueError(format_model_error(model_name, model_parameters, e)) from e
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Most likely CUDA not available => fallback to CPU
            if "cuda" not in str(e).lower() and self.device.lower() == "cuda":
                raise ValueError(format_model_error(model_name, model_parameters, e)) from e
            warnings.warn("CUDA not available, falling back to CPU.", stacklevel=2)
            calc = MODEL_REGISTRY[model_name](model_parameters, device="cpu")

        if calc is None:
            raise RuntimeError(f"Builder for '{model_name}' returned None")

        self.calc = calc

    def create_simulation_folder(self, base_name: Path | str = "simulation") -> tuple[str, str]:
        """Create a folder structure for simulation outputs.


        Args:
            base_name (Path | str, optional): Name of the base folder to be created in the current working directory. Defaults to "simulation".

        Returns:
            Tuple[str, str]: Tuple of the folders where the NPT and NVT trajectories will be stored.
        """
        run_dir = os.path.join(os.getcwd(), base_name)
        os.makedirs(run_dir, exist_ok=True)

        npt_dir = os.path.join(run_dir, "NPT")
        nvt_dir = os.path.join(run_dir, "NVT")
        os.makedirs(npt_dir, exist_ok=True)
        os.makedirs(nvt_dir, exist_ok=True)

        print(f"Simulation folders created in: {run_dir}")

        return npt_dir, nvt_dir

    def build_system(
        self,
        salt_anion: list[str],
        salt_cation: list[str],
        n_anions: list[int],
        n_cations: list[int],
        density_guess: float,
        lattice: str = "random",
        random_removal: bool = False,
        random_min_distance: float = 1.6,
        random_max_attempts: int = 100000,
    ) -> Atoms:
        """Build a molten salt system with random or rocksalt initial positions.

        Args:
            salt_anion (list[str]): Chemical symbols for anions
            salt_cation (list[str]): Chemical symbols for cations
            n_anions (list[int]): Number of atoms for each anion type
            n_cations (list[int]): Number of atoms for each cation type
            density_guess (float): Initial density guess in g/cm³
            lattice (str, optional): Initial lattice type ("random" or "rocksalt"). Defaults to "random".
            random_removal (bool, optional):  If True and lattice is "rocksalt", randomly remove excess atoms to match the desired composition. If False, simply take the first N positions from the generated lattice. Defaults to False.
            random_min_distance (float, optional): Minimum distance between atoms in the random lattice in Å. Defaults to 1.6.
            random_max_attempts (int, optional): Maximum number of attempts to place atoms at random positions. Defaults to 100000.

        Raises:
            ValueError: If the number of distinct ions and the number of amounts of those ions do not match.
            ValueError: If the lattice type is not supported.
            RuntimeError: If the initial box size is too small for the requested lattice (100'000 attempts were not enough to place an atom at a random position with a distance of 1.6 Å to every other atom).

        Returns:
            Atoms: The constructed system
        """

        if (len(salt_anion), len(salt_cation)) != (
            len(n_anions),
            len(n_cations),
        ):
            raise ValueError(
                f"The number of distinct ions {(len(salt_anion), len(salt_cation))} and the length of the list of atoms {(len(n_anions), len(n_cations))} must be equal"
            )

        # Construct the symbols array by spreading anions and cations evenly, shuffled within their groups
        cations = np.random.permutation(np.repeat(salt_cation, n_cations))
        anions = np.random.permutation(np.repeat(salt_anion, n_anions))
        n_tot = len(cations) + len(anions)
        idx = np.linspace(0, n_tot - 1, len(cations), dtype=int)
        mask = np.zeros(n_tot, dtype=bool)
        mask[idx] = True
        symbols = np.empty(n_tot, dtype="<U2")
        symbols[mask] = cations
        symbols[~mask] = anions

        # Calculate initial box size from density guess
        mass = sum(atomic_masses[atomic_numbers[sym]] for sym in symbols)  # amu
        # The density_guess needs to be converted from g/cm³ to amu/Å³
        density_guess_au = density_guess * 1e3 * units.kg / units.m**3
        volume_guess = mass / density_guess_au  # Å³

        if lattice == "random":
            # Place atoms with minimum distance constraint
            initial_box_size = volume_guess ** (1 / 3)  # Å
            positions_atoms = np.zeros((len(symbols), 3))  # Å
            for i in range(len(symbols)):
                for _ in range(random_max_attempts):
                    new_pos = np.random.rand(3) * initial_box_size
                    if i == 0:
                        positions_atoms[i] = new_pos
                        break
                    distances = cdist([new_pos], positions_atoms[:i])
                    if np.all(distances > random_min_distance):
                        positions_atoms[i] = new_pos
                        break
                else:
                    raise RuntimeError(
                        f"The density {density_guess} g/cm³ could not be achieved while maintaining a distance of {random_min_distance} Å to every other atom. Increase the initial density guess."
                    )

            atoms = Atoms(
                symbols=symbols,
                positions=positions_atoms,
                cell=[initial_box_size] * 3,
                pbc=True,
            )

        elif lattice == "rocksalt":
            # Generate an rocksalt lattice with arbitrary symbols and lattice constant
            atoms = bulk("XY", "rocksalt", a=1.0)
            cells_per_side = int(np.ceil((len(symbols) / 2) ** (1 / 3)))
            # Generate enough lattice positions to accommodate all atoms
            atoms = atoms.repeat((cells_per_side, cells_per_side, cells_per_side))
            # Remove excess positions
            if len(atoms) > len(symbols):
                if random_removal:
                    # Randomly select the respective amount of anion and cation positions to remove
                    num_an_positions_to_remove = int(len(atoms) / 2 - len(anions))
                    cat_indices_to_remove = np.random.choice(
                        np.arange(0, len(atoms), 2),
                        size=num_an_positions_to_remove,
                        replace=False,
                    )
                    num_cat_positions_to_remove = int(len(atoms) / 2 - len(cations))
                    an_indices_to_remove = np.random.choice(
                        np.arange(1, len(atoms), 2),
                        size=num_cat_positions_to_remove,
                        replace=False,
                    )
                    indices_to_remove = np.sort(np.concatenate((cat_indices_to_remove, an_indices_to_remove)))
                    atoms = atoms[np.setdiff1d(np.arange(len(atoms)), indices_to_remove)]
                else:
                    atoms = atoms[: len(symbols)]

            # Populate the lattice with the correct chemical symbols
            atoms.set_chemical_symbols(symbols)

            # Rescale the lattice to match the density guess
            scale = (volume_guess / atoms.get_volume()) ** (1 / 3)
            atoms.set_cell(atoms.get_cell() * scale, scale_atoms=True)
        else:
            raise ValueError(f"Unsupported lattice type: {lattice}")

        atoms.calc = self.calc

        return atoms

    def _print_status(self, dyn, atoms):
        """Helper function to print the status of the simulation."""
        step = dyn.get_number_of_steps()
        pressure = -np.sum(atoms.get_stress()[:3]) / 3
        print(
            f"Step {step:6d} | T = {atoms.get_temperature():0f} K | P = {pressure:.6e} bar | V = {atoms.get_volume():8.2f} Å³"
        )

    def _select_npt_dynamics(
        self,
        npt_dyn: str,
        atoms: Atoms,
        timestep_fs: float,
        T: float,
        taut_fs: float,
        taup_fs: float,
        pressure_bar: float,
        compressibility_per_bar: float,
        tchain: int,
        pchain: int,
        tloop: int,
        ploop: int,
        print_interval: int,
        logfile: str | Path | None,
    ) -> NoseHooverChainNVT | NPTBerendsen | MelchionnaNPT:
        """_summary_

        Args:
            npt_dyn (str): NPT dynamics to use. Choices: "nptberendsen", "mtknpt", "melchionna".
            atoms (Atoms): System to simulate
            T (float): Temperature in K.
            timestep_fs (float): Time step dt for the simulation in fs.
            taut_fs (float): Time constant for the NPT temperature coupling in fs.
            taup_fs (float): Time constant for the NPT pressure coupling in fs.
            pressure_bar (float): Pressure in bar.
            compressibility_per_bar (float): Compressibility of the system per bar in 1/bar.
            tchain (int): The number of thermostat variables in the Nose-Hoover thermostat. Only applies if npt_din is "mtknpt".
            pchain (int): The number of barostat variables in the MTK barostat. Only applies if npt_din is "mtknpt".
            tloop (int): The number of sub-steps in thermostat integration. Only applies if npt_din is "mtknpt".
            ploop (int): The number of sub-steps in barostat integration. Only applies if npt_din is "mtknpt".
            print_interval (int): Interval for printing status.
            logfile (str | Path | None): Logfile for the NPT dynamics simulation, "-" for stdout, None for no logfile.

        Raises:
            ValueError: If the NPT specified with npt_dyn is not supported.

        Returns:
            NoseHooverChainNVT | NPTBerendsen | MelchionnaNPT: _description_
        """
        if npt_dyn.lower() == "nptberendsen":
            dyn = NPTBerendsen(
                atoms,
                timestep=timestep_fs * units.fs,
                temperature_K=T,
                taut=taut_fs * units.fs,
                pressure_au=pressure_bar * units.bar,
                taup=taup_fs * units.fs,
                compressibility_au=compressibility_per_bar / units.bar,
                logfile=str(logfile),
                loginterval=print_interval,
            )
        # MelchionnaNPT can so far only operate on lists of atoms where the computational box is a triangular matrix
        # elif npt_dyn.lower() == "melchionnanpt":
        #     dyn = MelchionnaNPT(
        #         atoms,
        #         timestep=timestep_fs * units.fs,
        #         temperature_K=T,
        #         externalstress=pressure_bar * units.bar,
        #         ttime=taut_fs * units.fs,
        #         pfactor=(taup_fs * units.fs)**2 / compressibility_per_bar * units.bar,
        #         trajectory=None,
        #         logfile=str(logfile),
        #     )
        elif npt_dyn.lower() == "mtknpt":
            dyn = MTKNPT(
                atoms,
                timestep=timestep_fs * units.fs,
                temperature_K=T,
                pressure_au=pressure_bar * units.bar,
                tdamp=taut_fs * units.fs,
                pdamp=taup_fs * units.fs,
                tchain=tchain,
                pchain=pchain,
                tloop=tloop,
                ploop=ploop,
            )
        else:
            raise ValueError(f"Unsupported NPT dynamics: {npt_dyn}")
        return dyn  # type: ignore

    def run_npt_simulation(
        self,
        atoms: Atoms,
        T: float | int,
        npt_dyn: str = "nptberendsen",
        steps: int = 1000,
        timestep_fs: float = 1.0,
        taut_fs: float = 100.0,
        taup_fs: float = 1000.0,
        compressibility_per_bar: float = 5e-6,
        pressure_bar: float = 1.01325,
        tchain: int = 3,
        pchain: int = 3,
        tloop: int = 1,
        ploop: int = 1,
        print_interval: int = 100,
        write_interval: int = 10,
        traj_file: str | Path = "npt_simulation.traj",
        print_status: bool = True,
        logfile: str | Path | None = "npt_run.log",
    ) -> Atoms:
        """Run NPT (constant particles, pressure, temperature) molecular dynamics simulation.

        Args:
            atoms (Atoms): System to simulate.
            T (float | int): Temperature in K.
            npt_dyn (str, optional): NPT dynamics to use. Defaults to "nptberendsen". Choices: "nptberendsen", "mtknpt". Defaults to "nptberendsen".
            steps (int, optional): Number of MD steps. Defaults to 1000.
            timestep_fs (float, optional): Time step dt for the simulation in fs. Defaults to 1.0.
            taut_fs (float, optional): Time constant for the NPT temperature coupling in fs. Defaults to 100.0.
            taup_fs (float, optional): Time constant for the NPT pressure coupling in fs. Defaults to 1000.0.
            compressibility_per_bar (float, optional): Compressibility of the system per bar in 1/bar. Defaults to 5e-6.
            pressure_bar (float, optional): Pressure in bar. Defaults to 1.01325.
            tchain (int, optional): The number of thermostat variables in the Nose-Hoover thermostat. Only applies if npt_din is "mtknpt". Defaults to 3.
            pchain (int, optional): The number of barostat variables in the MTK barostat. Only applies if npt_din is "mtknpt". Defaults to 3.
            tloop (int, optional): The number of sub-steps in thermostat integration. Only applies if npt_din is "mtknpt". Defaults to 1.
            ploop (int, optional): The number of sub-steps in barostat integration. Only applies if npt_din is "mtknpt". Defaults to 1.
            print_interval (int, optional): Interval for printing status. Defaults to 100.
            write_interval (int, optional): Interval for writing trajectory frames. Defaults to 10.
            traj_file (str | Path, optional): Output trajectory file path. Defaults to "npt_simulation.traj".
            print_status (bool, optional): Whether to print simulation status. Defaults to True.
            logfile (str | Path | None, optional): Logfile for the NPT dynamics simulation, "-" for stdout, None for no logfile. Defaults to "npt_run.log".

        Returns:
            Atoms: ASE atoms object of the equilibrated system
        """

        # Set up the atomic momenta at the given temperature and remove center of mass motion
        MaxwellBoltzmannDistribution(atoms, temperature_K=T, force_temp=True)
        Stationary(atoms)
        ZeroRotation(atoms)

        # Run the NPT dynamics simulation
        dyn = self._select_npt_dynamics(
            npt_dyn,
            atoms,
            timestep_fs,
            T,
            taut_fs,
            taup_fs,
            pressure_bar,
            compressibility_per_bar,
            tchain,
            pchain,
            tloop,
            ploop,
            print_interval,
            logfile,
        )

        # Write the initial atoms to the trajectory file with the time set to 0 fs
        atoms.info.update({"time_fs": 0.0})
        trajectory_npt = Trajectory(traj_file, "w", atoms, properties=["energy", "forces", "stress"])
        # Attach the trajectory writer and time updater to the dynamics simulation
        dyn.attach(
            lambda: atoms.info.update({"time_fs": dyn.get_time() / units.fs}),
            interval=write_interval,
        )
        dyn.attach(trajectory_npt.write, interval=write_interval)  # type: ignore

        if print_status:
            dyn.attach(lambda: self._print_status(dyn, atoms), interval=print_interval)

        # Run the simulation
        dyn.run(steps)

        # Close the trajectory file
        trajectory_npt.close()
        print(f"NPT simulation finished, trajectory saved to {traj_file}")

        return atoms

    def _select_nvt_dynamics(
        self,
        nvt_dyn: str,
        atoms: Atoms,
        T: float,
        timestep_fs: float,
        tdamp_fs: float,
        print_interval: int,
        logfile: str | Path | None,
    ) -> NVTBerendsen | NoseHooverChainNVT | Langevin | Bussi | Andersen:
        """_summary_

        Args:
            nvt_dyn (str): NVT dynamics to use. Choices: "nvtberendsen", "nosehoover", "langevin", "bussi", "andersen".
            atoms (Atoms): System to simulate.
            T (float): Temperature in K.
            timestep_fs (float): Time step dt for the simulation in fs.
            tdamp_fs (float): Characteristic time scale for thermostat in fs, typically 100*timestep_fs.
            print_interval (int): Interval for printing status.
            logfile (str | Path | None): Logfile for the NVT dynamics simulation, "-" for stdout, None for no logfile.

        Raises:
            ValueError: If the NVT dynamics specified with nvt_dyn is not supported.

        Returns:
            NVTBerendsen | NoseHooverChainNVT | Langevin | Bussi | Andersen: _description_
        """
        if nvt_dyn.lower() == "nvtberendsen":
            dyn = NVTBerendsen(
                atoms,
                timestep=timestep_fs * units.fs,
                temperature_K=T,
                taut=tdamp_fs * units.fs,
                logfile=str(logfile),
            )
        elif nvt_dyn.lower() == "nosehoover":
            dyn = NoseHooverChainNVT(
                atoms,
                timestep=timestep_fs * units.fs,
                temperature_K=T,
                tdamp=tdamp_fs * units.fs,
                logfile=logfile,
                loginterval=print_interval,
            )
        elif nvt_dyn.lower() == "langevin":
            dyn = Langevin(
                atoms,
                timestep=timestep_fs * units.fs,
                temperature_K=T,
                friction=1 / (tdamp_fs * units.fs),
                logfile=logfile,
            )
        elif nvt_dyn.lower() == "bussi":
            dyn = Bussi(
                atoms,
                timestep=timestep_fs * units.fs,
                temperature_K=T,
                taut=tdamp_fs * units.fs,
                logfile=logfile,
            )
        elif nvt_dyn.lower() == "andersen":
            dyn = Andersen(
                atoms,
                timestep=timestep_fs * units.fs,
                temperature_K=T,
                andersen_prob=1 / (tdamp_fs * units.fs),
                logfile=logfile,
            )
        else:
            raise ValueError(f"Unsupported NVT dynamics: {nvt_dyn}")

        return dyn  # type: ignore

    def run_nvt_simulation(
        self,
        atoms: Atoms,
        T: float | int,
        nvt_dyn: str = "nosehoover",
        steps: int = 1000,
        timestep_fs: float = 1.0,
        tdamp_fs: float = 100.0,
        print_interval: int = 100,
        write_interval: int = 10,
        traj_file: str | Path = "nvt_simulation.traj",
        print_status: bool = True,
        logfile: str | Path | None = "nvt_run.log",
    ):
        """Run NVT (constant particles, volume, temperature) molecular dynamics simulation.

        Args:
            atoms (Atoms): System to simulate.
            T (float | int): Temperature in K.
            nvt_dyn (str, optional): NVT dynamics to use. Choices: "nvtberendsen", "nosehoover", "langevin", "bussi", "andersen". Defaults to "nosehoover".
            steps (int, optional): Number of MD steps. Defaults to 1000.
            timestep_fs (float, optional): Time step dt for the simulation in fs. Defaults to 1.0.
            tdamp_fs (float, optional): Characteristic time scale for thermostat in fs, typically 100*timestep_fs. Defaults to 100.0.
            print_interval (int, optional): Interval for printing status. Defaults to 100.
            write_interval (int, optional): Interval for writing trajectory frames. Defaults to 10.
            traj_file (str | Path, optional): Output trajectory file path. Defaults to "nvt_simulation.traj".
            print_status (bool, optional): Whether to print simulation status. Defaults to True.
            logfile (str | Path | None, optional): Logfile for the NoseHooverChainNVT dynamics simulation, "-" for stdout, None for no logfile. Defaults to "nvt_run.log".
        """

        # Set up the atomic momenta at the given temperature and remove center of mass motion
        MaxwellBoltzmannDistribution(atoms, temperature_K=T, force_temp=True)
        Stationary(atoms)
        ZeroRotation(atoms)

        # Setup the Nose-Hoover chain NVT dynamics simulation
        dyn = self._select_nvt_dynamics(nvt_dyn, atoms, T, timestep_fs, tdamp_fs, print_interval, logfile)

        # Write the initial atoms to the trajectory file
        atoms.info.update({"time_fs": 0.0})
        trajectory_nvt = Trajectory(traj_file, "w", atoms, properties=["energy", "forces", "stress"])

        # Attach the trajectory writer and time updater to the dynamics simulation
        dyn.attach(
            lambda: atoms.info.update({"time_fs": dyn.get_time() / units.fs}),
            interval=write_interval,
        )
        dyn.attach(trajectory_nvt.write, interval=write_interval)  # type: ignore

        if print_status:
            dyn.attach(lambda: self._print_status(dyn, atoms), interval=print_interval)

        # Run the simulation
        dyn.run(steps)
        # Close the trajectory file
        trajectory_nvt.close()
        print(f"NVT Simulation finished, trajectory saved to {traj_file}")


# Example usage
if __name__ == "__main__":  # pragma: no cover
    np.random.seed(42)  # Ensure reproducibility (initial random placements)
    # Setup the MS simulator class with the desired model and parameters
    sim = MoltenSaltSimulator(
        model_name="grace",
        device="cpu",
        model_parameters={
            "model_task": "OAM",
            "model_size": "small",
            "num_layers": 1,
        },
    )
    N_STEPS = 10  # In practice this needs to be ~100'000 steps
    N_STEPS_OUTPUT = 2
    N_WRITE = 2
    TIMESTEP = 10.0  # Quite long, in practice usually 1 fs
    # Define salts to simulate like:   "salt_name": ([anions], [cations], amount_of_anions, amount_of_cations)
    salts = {"NaCl": (["Cl"], ["Na"], [150], [150])}
    # Define at which temperatures you want to calculate the properties per salt
    temperatures = {"NaCl": [1100, 1150, 1200]}
    # Define what density you guess the salt to have at the corresponding temperatures
    initial_densities = {"NaCl": [1.542, 1.515, 1.488]}
    # Run the simulation
    for salt_name, (an, cat, n_an, n_cat) in salts.items():
        print(f"\nRunning NPT simulations for {salt_name}...\n")

        # Create folders to store the trajectories
        npt, nvt = sim.create_simulation_folder(base_name=os.path.join("test_sim", f"GRACE_1L_{salt_name}_super_short"))

        # Pair each temperature with its corresponding density guess
        for temp, initial_density in zip(temperatures[salt_name], initial_densities[salt_name], strict=False):
            system = sim.build_system(an, cat, n_an, n_cat, initial_density, lattice="rocksalt")
            traj_file_npt = os.path.join(npt, f"npt_{salt_name}_{temp}K.traj")
            traj_file_nvt = os.path.join(nvt, f"nvt_{salt_name}_{temp}K.traj")
            sim.run_npt_simulation(
                system,
                temp,
                steps=N_STEPS,
                print_interval=N_STEPS_OUTPUT,
                write_interval=N_WRITE,
                traj_file=traj_file_npt,
                print_status=True,
                timestep_fs=TIMESTEP,
            )
            sim.run_nvt_simulation(
                system,
                temp,
                steps=N_STEPS,
                print_interval=N_STEPS_OUTPUT,
                write_interval=N_WRITE,
                traj_file=traj_file_nvt,
                print_status=True,
                timestep_fs=TIMESTEP,
            )
