import os

import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.data import atomic_masses, atomic_numbers
from ase.io import Trajectory, write
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from scipy.spatial.distance import cdist


class MoltenSaltSimulator:
    """
    Class for building molten salt systems and running molecular dynamics simulations.
    """

    def __init__(self, model_name="GRACE", model_parameters=None, device="cuda"):
        """Initialize the simulator with a specific ML potential."""
        self.device = device
        self.calc = None
        self.set_calculator(model_name, model_parameters)

    def set_calculator(self, model_name, model_parameters=None):
        """Set up the calculator based on the chosen ML potential."""
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
            else:
                calculator_unavailable = True

        else:
            raise ValueError(f"Model '{model_name}' has no available calculators")

        if calculator_unavailable:
            raise ValueError(
                f"The model {model_name} has no calculator with the parameters: {model_parameters}"
            )

        return self.calc

    def create_simulation_folder(self, base_name="simulation"):
        """Create a folder structure for simulation outputs."""
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
        salt_anion,
        salt_cation,
        anion_Natoms,
        cation_Natoms,
        density_guess,
        lattice="random",
    ):
        """
        Build a molten salt system with random or rocksalt initial positions.

        Parameters:
        -----------
        salt_anion : list of str
            Chemical symbols for anions
        salt_cation : list of str
            Chemical symbols for cations
        anion_Natoms : list of int
            Number of atoms for each anion type
        cation_Natoms : list of int
            Number of atoms for each cation type
        density_guess : float
            Initial density guess (g/cm³)
        lattice : str
            Initial lattice type ("random" or "rocksalt")

        Returns:
        --------
        atoms : ASE Atoms object
            The constructed system
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
                # TODO: Not so nice, this could lead to an infinite loop in case the box is too small. But we need to replace this anyways with a better initial placement
                while True:
                    new_pos = np.random.rand(3) * initial_box_size
                    if i == 0:
                        positions_atoms[i] = new_pos
                        break
                    distances = cdist([new_pos], positions_atoms[:i])
                    if np.all(distances > min_distance):
                        positions_atoms[i] = new_pos
                        break

                # Create ASE Atoms object
                atoms = Atoms(
                    symbols=symbols,
                    positions=positions_atoms,
                    cell=[initial_box_size] * 3,
                    pbc=True,
                )

        elif lattice == "rocksalt":
            # Generate an rocksalt lattice with arbitrary symbols and lattice constant of 1 A
            atoms = bulk("XY", "rocksalt", a=1.0)
            cells_per_side = int(
                np.ceil((len(symbols) / 2) ** (1 / 3))
            )  #  Two atoms per rocksalt unit cell
            # Generate enough lattice positions to accommodate all atoms
            atoms = atoms.repeat((cells_per_side, cells_per_side, cells_per_side))
            # Randomly remove excess positions
            if len(atoms) > len(symbols):
                num_positions_to_remove = len(atoms) - len(symbols)
                cat_indices_to_remove = np.random.choice(
                    np.arange(0, len(atoms), 2),
                    size=num_positions_to_remove // 2,
                    replace=False,
                )
                an_indices_to_remove = np.random.choice(
                    np.arange(1, len(atoms), 2),
                    size=num_positions_to_remove // 2,
                    replace=False,
                )
                indices_to_remove = np.sort(
                    np.concatenate((cat_indices_to_remove, an_indices_to_remove))
                )
                atoms = atoms[np.setdiff1d(np.arange(len(atoms)), indices_to_remove)]
            # Populate with the correct chemical symbols
            atoms.set_chemical_symbols(symbols)

            # Rescale the lattice to match the density guess
            scale = (volume_guess / atoms.get_volume()) ** (1 / 3)
            atoms.set_cell(atoms.get_cell() * scale, scale_atoms=True)

            # TODO: Remove these lines, just for testing
            write("scaled_str.png", atoms)
            write("scaled_an.png", atoms[::2])
            write("scaled_cat.png", atoms[1::2])
            # Calculate the initial density
            density_guess_calc = (
                atoms.get_masses().sum()
                * units._amu
                * 1e3
                / (atoms.get_volume() * 1e-24)
            )
            print(
                f"Initial density guess after calculation: {density_guess_calc:.3f} g/cm3"
            )

        else:
            raise ValueError(f"Unsupported lattice type: {lattice}")

        if self.calc:
            atoms.calc = self.calc
        else:
            raise RuntimeError("Calculator not set. Use set_calculator() first.")

        return atoms

    def run_npt_simulation(
        self,
        atoms,
        T,
        steps=1000,
        print_interval=100,
        traj_file="npt_simulation.traj",
        print_status=True,
    ):
        """
        Run NPT (constant pressure, temperature) molecular dynamics.

        Parameters:
        -----------
        atoms : ASE Atoms object
            System to simulate
        T : float
            Temperature (K)
        steps : int
            Number of MD steps
        print_interval : int
            Interval for printing status
        traj_file : str
            Output trajectory file
        print_status : bool
            Whether to print simulation status

        Returns:
        --------
        atoms : ASE Atoms object
            The equilibrated system
        """
        MaxwellBoltzmannDistribution(atoms, temperature_K=T)

        dyn = NPTBerendsen(
            atoms,
            timestep=1.0 * units.fs,
            temperature_K=T,
            taut=100 * units.fs,
            pressure_au=1.01325 * units.bar,
            taup=1000 * units.fs,
            compressibility_au=4.0e-5 / units.bar,
            logfile="npt_equili.log",
        )

        trajectory_npt = Trajectory(traj_file, "w", atoms)
        dyn.attach(
            trajectory_npt.write, interval=10
        )  # TODO: As only every tenth frame is recorded, we only record every 10 fs => Check if Max's simulations were 200 ps or 2000 ps?

        if print_status:

            def print_status_func():
                step = dyn.get_number_of_steps()
                stress_tensor = atoms.get_stress(voigt=False) * 1 / units.bar
                pressure = -np.trace(stress_tensor) / 3
                # TODO: Why do we print the pressure when it's supposed to be constant?
                print(
                    f"Step {step:6d} | P = {pressure:.6e} bar | V = {atoms.get_volume():8.2f} Å³"
                )

            dyn.attach(print_status_func, interval=print_interval)

        dyn.run(steps)
        trajectory_npt.close()
        print(f"NPT trajectory saved to {traj_file}")

        return atoms

    def run_nvt_simulation(
        self,
        atoms,
        T,
        steps=1000,
        print_interval=100,
        traj_file="nvt_simulation.traj",
        print_status=True,
    ):
        """
        Run NVT (constant volume, temperature) molecular dynamics.

        Parameters:
        -----------
        atoms : ASE Atoms object
            System to simulate
        T : float
            Temperature (K)
        steps : int
            Number of MD steps
        print_interval : int
            Interval for printing status
        traj_file : str
            Output trajectory file
        print_status : bool
            Whether to print simulation status

        Returns:
        --------
        None
        """
        MaxwellBoltzmannDistribution(atoms, temperature_K=T)

        dyn = NoseHooverChainNVT(
            atoms,
            timestep=1.0 * units.fs,
            temperature_K=T,
            tdamp=100 * units.fs,
            logfile="nvt_run.log",
        )

        trajectory_nvt = Trajectory(traj_file, "w", atoms)
        dyn.attach(trajectory_nvt.write, interval=10)

        if print_status:

            def print_status_func():
                step = dyn.get_number_of_steps()
                stress_tensor = atoms.get_stress(voigt=False) * 1 / units.bar
                pressure = -np.trace(stress_tensor) / 3
                print(
                    f"Step {step:6d} | P = {pressure:.6e} bar | V = {atoms.get_volume():8.2f} Å³"
                )

            dyn.attach(print_status_func, interval=print_interval)

        dyn.run(steps)
        trajectory_nvt.close()
        print(f"NVT trajectory saved to {traj_file}")


# Example usage
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility of the initial random placements
    # Setup the MS simulator class with the desired model and parameters
    sim = MoltenSaltSimulator(
        model_name="GRACE", model_parameters={"model_size": "small", "layer": 1}
    )
    n_steps = 100  # 1 step is 1 fs, so to get the 200 ps, we need 200000 steps, but for testing it can be lower
    n_steps_output = 10
    # Define salts to simulate like:   "salt_name": ([anions], [cations], amount_of_anions, amount_of_cations)
    salts = {
        "NaCl": (["Cl"], ["Na"], [150], [150]),
        # "0.3NaCl-0.2KCl-0.5MgCl2": (["Cl"], ["K", "Mg", "Na"], [150], [20, 50, 30]),
    }  # To test a size of like 20 ions for each is appropriate
    # Define at which temperatures you want to calculate the properties per salt
    temperatures = {
        "NaCl": [1100, 1125, 1150, 1175, 1200][:1],
        "0.3NaCl-0.2KCl-0.5MgCl2": [700, 800, 900, 1000, 1100],
    }  # For testing 3 is enough
    # Define what density you guess the salt to have at the corresponding temperatures
    density_guesses = {
        "NaCl": [1.542, 1.528, 1.515, 1.501, 1.488],
        "0.3NaCl-0.2KCl-0.5MgCl2": [1.761, 1.719, 1.677, 1.635, 1.593],
    }  # For testing 3 is enough

    # Run the simulation
    for salt_name, (anions, cations, n_anions, n_cations) in salts.items():
        print(f"Running NPT simulations for {salt_name}...")

        # Create folders to store the trajectories
        npt_dir, nvt_dir = sim.create_simulation_folder(
            base_name=os.path.join("test_sim", f"GRACE_1L_{salt_name}_test")
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
            sim.run_npt_simulation(
                atoms,
                T,
                steps=n_steps,
                print_interval=n_steps_output,
                traj_file=traj_file_npt,
                print_status=True,
            )
            sim.run_nvt_simulation(
                atoms,
                T,
                steps=n_steps,
                print_interval=n_steps_output,
                traj_file=traj_file_nvt,
                print_status=True,
            )
