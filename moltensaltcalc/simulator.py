import os
from typing import Tuple

import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.data import atomic_masses, atomic_numbers
from ase.io import Trajectory
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from scipy.spatial.distance import cdist


class MoltenSaltSimulator:
    """Class for building molten salt systems and running molecular dynamics simulations supported by energy estimates from uMLIPs."""

    def __init__(
        self,
        model_name: str = "GRACE",
        model_parameters: dict | None = None,
        device: str = "cuda",
    ):
        """Initialize the simulator with a specific ML potential.

        Args:
            model_name (str, optional): Which MLIP to use, select from "FAIRCHEM", "MACE" and "GRACE". Defaults to "GRACE".
            model_parameters (dict | None, optional): Parameters for the MLIP. Defaults to None which means {"model_size": "medium", "layer": 1}.
            device (str, optional): Which device to use for the calculations, select from "cpu" and "cuda". Defaults to "cuda".
        """
        if model_parameters is None:
            model_parameters = {"model_size": "medium", "layer": 1}
        self.device = device
        self.calc = None
        self._set_calculator(model_name, model_parameters)

    def _set_calculator(self, model_name: str, model_parameters: dict | None = None):
        """Sets the calculator based on the chosen ML potential.

        Args:
            model_name (str): Which MLIP to use, select from "FAIRCHEM", "MACE" and "GRACE". Defaults to "GRACE".
            model_parameters (dict | Nones, optional): _description_. Defaults to None.

        Raises:
            ValueError: In case no match was found for the model name and parameters.
        """
        if model_parameters is None:
            model_parameters = {}

        # Raises an exception at the end of the function if no match was found
        calculator_unavailable = False

        if model_name == "fairchem" or model_name == "FAIRCHEM":
            from fairchem.core import FAIRChemCalculator, pretrained_mlip

            if model_parameters.get("model_size") == "small":
                predictor = pretrained_mlip.get_predict_unit(
                    "uma-s-1", device=self.device
                )
                self.calc = FAIRChemCalculator(
                    predictor, task_name=model_parameters.get("model_task")
                )
            elif model_parameters.get("model_size") == "medium":
                predictor = pretrained_mlip.get_predict_unit(
                    "uma-m-1p1", device=self.device
                )
                self.calc = FAIRChemCalculator(
                    predictor, task_name=model_parameters.get("model_task")
                )
            else:
                calculator_unavailable = True

        elif model_name == "MACE" or model_name == "mace":
            from mace.calculators import mace_mp

            if model_parameters.get("model_type") == "mace-mh-1":
                self.calc = mace_mp(
                    model="mace-mh-1.model", default_dtype="float64", device=self.device
                )
            else:
                calculator_unavailable = True

        elif model_name == "GRACE" or model_name == "grace":
            from tensorpotential.calculator.foundation_models import (
                GRACEModels,
                grace_fm,
            )

            if (
                model_parameters.get("model_size") == "small"
                and model_parameters.get("layer") == 1
            ):
                self.calc = grace_fm(GRACEModels.GRACE_1L_OMAT)
            elif (
                model_parameters.get("model_size") == "medium"
                and model_parameters.get("layer") == 1
            ):
                self.calc = grace_fm(GRACEModels.GRACE_1L_OMAT_medium_base)
            else:
                calculator_unavailable = True

        else:
            raise ValueError(f"Model '{model_name}' has no available calculators")

        if calculator_unavailable:
            raise ValueError(
                f"The model {model_name} has no calculator with the parameters: {model_parameters}"
            )

    def create_simulation_folder(
        self, base_name: str = "simulation"
    ) -> Tuple[str, str]:
        """Create a folder structure for simulation outputs.


        Args:
            base_name (str, optional): Name of the base folder to be created in the current working directory. Defaults to "simulation".

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
        anion_Natoms: list[int],
        cation_Natoms: list[int],
        density_guess: float,
        lattice: str = "random",
        random_removal: bool = False,
    ) -> Atoms:
        """Build a molten salt system with random or rocksalt initial positions.

        Args:
            salt_anion (list[str]): Chemical symbols for anions
            salt_cation (list[str]): Chemical symbols for cations
            anion_Natoms (list[int]): Number of atoms for each anion type
            cation_Natoms (list[int]): Number of atoms for each cation type
            density_guess (float): Initial density guess (g/cm³)
            lattice (str, optional): Initial lattice type ("random" or "rocksalt"). Defaults to "random".
            random_removal (bool, optional):  If True and lattice is "rocksalt", randomly remove excess atoms to match the desired composition. If False, simply take the first N positions from the generated lattice. Defaults to False.

        Raises:
            ValueError: If the number of distinct ions and the number of amounts of those ions do not match.
            ValueError: If the lattice type is not supported.
            RuntimeError: If the initial box size is too small for the requested lattice (100'000 attempts were not enough to place an atom at a random position with a distance of 1.6 Å to every other atom).

        Returns:
            Atoms: The constructed system
        """

        if (len(salt_anion), len(salt_cation)) != (
            len(anion_Natoms),
            len(cation_Natoms),
        ):
            raise ValueError(
                f"The number of distinct ions {(len(salt_anion), len(salt_cation))} and the length of the list of atoms {(len(anion_Natoms), len(cation_Natoms))} must be equal"
            )

        # Construct the symbols array by spreading anions and cations evenly, shuffled within their groups
        cations = np.random.permutation(np.repeat(salt_cation, cation_Natoms))
        anions = np.random.permutation(np.repeat(salt_anion, anion_Natoms))
        Ntot = len(cations) + len(anions)
        idx = np.linspace(0, Ntot - 1, len(cations), dtype=int)
        mask = np.zeros(Ntot, dtype=bool)
        mask[idx] = True
        symbols = np.empty(Ntot, dtype="<U2")
        symbols[mask] = cations
        symbols[~mask] = anions

        # Calculate initial box size from density guess
        mass = sum(atomic_masses[atomic_numbers[sym]] for sym in symbols)  # amu
        # The density_guess needs to be converted from g/cm³ to amu/Å³
        density_guess_au = density_guess * 1e3 / (units._amu * units.m**3)
        volume_guess = mass / density_guess_au  # Å³

        if lattice == "random":
            initial_box_size = volume_guess ** (1 / 3)  # Å
            # Place atoms with minimum distance constraint
            min_distance = 1.6  # Å
            positions_atoms = np.zeros((len(symbols), 3))
            for i in range(len(symbols)):
                max_attempts = 100000
                for attempt in range(max_attempts):
                    new_pos = np.random.rand(3) * initial_box_size
                    if i == 0:
                        positions_atoms[i] = new_pos
                        break
                    distances = cdist([new_pos], positions_atoms[:i])
                    if np.all(distances > min_distance):
                        positions_atoms[i] = new_pos
                        break
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"The density {density_guess} g/cm³ could not be achieved while maintaining a distance of {min_distance} Å to every other atom. Increase the initial density guess."
                    )

                # Create ASE Atoms object
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
                    num_an_positions_to_remove = len(atoms) / 2 - len(anions)
                    cat_indices_to_remove = np.random.choice(
                        np.arange(0, len(atoms), 2),
                        size=num_an_positions_to_remove,
                        replace=False,
                    )
                    num_cat_positions_to_remove = len(atoms) / 2 - len(cations)
                    an_indices_to_remove = np.random.choice(
                        np.arange(1, len(atoms), 2),
                        size=num_cat_positions_to_remove,
                        replace=False,
                    )
                    indices_to_remove = np.sort(
                        np.concatenate((cat_indices_to_remove, an_indices_to_remove))
                    )
                    atoms = atoms[
                        np.setdiff1d(np.arange(len(atoms)), indices_to_remove)
                    ]
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

    def run_npt_simulation(
        self,
        atoms: Atoms,
        T: float | int,
        steps: int = 1000,
        timestep_fs: float = 1.0,
        taut_fs: float = 100.0,
        taup_fs: float = 1000.0,
        compressibility_per_bar: float = 4.0e-5,
        pressure_bar: float = 1.01325,
        print_interval: int = 100,
        write_interval: int = 10,
        traj_file: str = "npt_simulation.traj",
        print_status: bool = True,
        logfile: str = "npt_equili.log",
    ) -> Atoms:
        """Run NPT (constant particles, pressure, temperature) molecular dynamics simulation.
        Args:
            atoms (Atoms): System to simulate
            T (float | int): Temperature in K
            steps (int, optional): Number of MD steps. Defaults to 1000.
            timestep_fs (float, optional): Time step dt for the simulation in fs. Defaults to 1.0.
            taut_fs (float, optional): Time constant for Berendsen temperature coupling in fs. Defaults to 100.0.
            taup_fs (float, optional): Time constant for Berendsen pressure coupling in fs. Defaults to 1000.0.
            compressibility_per_bar (float, optional): Compressibility of the system per bar in 1/bar. Defaults to 4.0e-5.
            pressure_bar (float, optional): Pressure in bar. Defaults to 1.01325.
            print_interval (int, optional): Interval for printing status. Defaults to 100.
            write_interval (int, optional): Interval for writing trajectory frames. Defaults to 10.
            traj_file (str, optional): Output trajectory file path. Defaults to "npt_simulation.traj".
            print_status (bool, optional): Whether to print simulation status. Defaults to True.
            logfile (str, optional): Logfile for the NPTBerendsen dynamics simulation, "-" for stdout. Defaults to "npt_equili.log".

        Returns:
            Atoms: ASE atoms object of the equilibrated system
        """

        # Set up the atomic momenta to a maxwell-boltzmann distribution at the given temperature
        MaxwellBoltzmannDistribution(atoms, temperature_K=T)

        # Run the NPT dynamics simulation
        dyn = NPTBerendsen(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=T,
            taut=taut_fs * units.fs,
            pressure_au=pressure_bar * units.bar,
            taup=taup_fs * units.fs,
            compressibility_au=compressibility_per_bar / units.bar,
            logfile=logfile,
            loginterval=print_interval,
        )
        # Write the initial atoms to the trajectory file with the time set to 0 fs
        atoms.info.update({"time_fs": 0.0})
        trajectory_npt = Trajectory(traj_file, "w", atoms)
        # Attach the trajectory writer and time updater to the dynamics simulation
        dyn.attach(
            lambda: atoms.info.update({"time_fs": dyn.get_time() / units.fs}),
            interval=write_interval,
        )
        dyn.attach(trajectory_npt.write, interval=write_interval)

        if print_status:
            dyn.attach(lambda: self._print_status(dyn, atoms), interval=print_interval)

        # Run the simulation
        dyn.run(steps)

        # Close the trajectory file
        trajectory_npt.close()
        print(f"NPT simulation finished, trajectory saved to {traj_file}")

        return atoms

    def run_nvt_simulation(
        self,
        atoms: Atoms,
        T: float | int,
        steps: int = 1000,
        timestep_fs: float = 1.0,
        tdamp_fs: float = 100.0,
        print_interval: int = 100,
        write_interval: int = 10,
        traj_file: str = "nvt_simulation.traj",
        print_status: bool = True,
        logfile: str = "nvt_run.log",
    ):
        """Run NVT (constant particles, volume, temperature) molecular dynamics simulation.

        Args:
            atoms (Atoms): System to simulate
            T (float | int): Temperature in K
            steps (int, optional): Number of MD steps. Defaults to 1000.
            timestep_fs (float, optional): Time step dt for the simulation in fs. Defaults to 1.0.
            tdamp_fs (float, optional): Characteristic time scale for thermostat in fs, typically 100*timestep_fs. Defaults to 100.0.
            print_interval (int, optional): Interval for printing status. Defaults to 100.
            write_interval (int, optional): Interval for writing trajectory frames. Defaults to 10.
            traj_file (str, optional): Output trajectory file path. Defaults to "nvt_simulation.traj".
            print_status (bool, optional): Whether to print simulation status. Defaults to True.
            logfile (str, optional): Logfile for the NoseHooverChainNVT dynamics simulation, "-" for stdout. Defaults to "nvt_run.log".
        """

        # Set up the atomic momenta to a maxwell-boltzmann distribution at the given temperature
        MaxwellBoltzmannDistribution(atoms, temperature_K=T)

        # Setup the Nose-Hoover chain NVT dynamics simulation
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=T,
            tdamp=tdamp_fs * units.fs,
            logfile=logfile,
            loginterval=print_interval,
        )

        # Write the initial atoms to the trajectory file
        atoms.info.update({"time_fs": 0.0})
        trajectory_nvt = Trajectory(traj_file, "w", atoms)

        # Attach the trajectory writer and time updater to the dynamics simulation
        dyn.attach(
            lambda: atoms.info.update({"time_fs": dyn.get_time() / units.fs}),
            interval=write_interval,
        )
        dyn.attach(trajectory_nvt.write, interval=write_interval)

        if print_status:
            dyn.attach(lambda: self._print_status(dyn, atoms), interval=print_interval)

        # Run the simulation
        dyn.run(steps)
        # Close the trajectory file
        trajectory_nvt.close()
        print(f"NVT Simulation finished, trajectory saved to {traj_file}")


# Example usage
if __name__ == "__main__":
    np.random.seed(42)  # Ensure reproducibility (initial random placements)
    # Setup the MS simulator class with the desired model and parameters
    sim = MoltenSaltSimulator(
        model_name="GRACE", model_parameters={"model_size": "small", "layer": 1}
    )
    n_steps = 10  # In practice this needs to be ~100'000 steps
    n_steps_output = 2
    write_interval = 2
    timestep_fs = 10.0  # Quite long, in practice usually 1 fs
    # Define salts to simulate like:   "salt_name": ([anions], [cations], amount_of_anions, amount_of_cations)
    salts = {"NaCl": (["Cl"], ["Na"], [150], [150])}
    # Define at which temperatures you want to calculate the properties per salt
    temperatures = {"NaCl": [1100, 1150, 1200]}
    # Define what density you guess the salt to have at the corresponding temperatures
    density_guesses = {"NaCl": [1.542, 1.515, 1.488]}
    # Run the simulation
    for salt_name, (anions, cations, n_anions, n_cations) in salts.items():
        print(f"\nRunning NPT simulations for {salt_name}...\n")

        # Create folders to store the trajectories
        npt_dir, nvt_dir = sim.create_simulation_folder(
            base_name=os.path.join("test_sim", f"GRACE_1L_{salt_name}_super_short")
        )

        # Pair each temperature with its corresponding density guess
        for T, density_guess in zip(
            temperatures[salt_name], density_guesses[salt_name]
        ):
            atoms = sim.build_system(
                anions, cations, n_anions, n_cations, density_guess, lattice="rocksalt"
            )
            traj_file_npt = os.path.join(npt_dir, f"npt_{salt_name}_{T}K.traj")
            traj_file_nvt = os.path.join(nvt_dir, f"nvt_{salt_name}_{T}K.traj")
            atoms = sim.run_npt_simulation(
                atoms,
                T,
                steps=n_steps,
                print_interval=n_steps_output,
                write_interval=write_interval,
                traj_file=traj_file_npt,
                print_status=True,
                timestep_fs=timestep_fs,
            )
            sim.run_nvt_simulation(
                atoms,
                T,
                steps=n_steps,
                print_interval=n_steps_output,
                timestep_fs=timestep_fs,
                write_interval=write_interval,
                traj_file=traj_file_nvt,
                print_status=True,
            )
