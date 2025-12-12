import numpy as np
from ase import Atoms, units
from ase.io import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.data import atomic_masses, atomic_numbers
from scipy.spatial.distance import cdist
import os


class MoltenSaltSimulator:
    """
    Class for building molten salt systems and running molecular dynamics simulations.
    """
    
    def __init__(self, model_name="uma-s-1", device="cuda"):
        """Initialize the simulator with a specific ML potential."""
        self.device = device
        self.calc = None
        self.set_calculator(model_name, {})
    
    def set_calculator(self, model_name, model_parameters=None):
        """Set up the calculator based on the chosen ML potential."""
        if model_parameters is None:
            model_parameters = {}
        
        if model_name == "fairchem" or model_name == "FAIRCHEM":
            from fairchem.core import pretrained_mlip, FAIRChemCalculator
            if model_parameters.get("model_size") == "small":
                predictor = pretrained_mlip.get_predict_unit("uma-s-1", device=self.device)
                self.calc = FAIRChemCalculator(predictor, task_name=model_parameters.get("model_task"))
            elif model_parameters.get("model_size") == "medium":
                predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device=self.device)
                self.calc = FAIRChemCalculator(predictor, task_name=model_parameters.get("model_task"))
            else:
                raise ValueError("This calculator type is not included in this package")
        
        elif model_name == "MACE" or model_name == "mace":
            from mace.calculators import mace_mp
            if model_parameters.get("model_type") == "mace-mh-1":
                self.calc = mace_mp(model="mace-mh-1.model", default_dtype="float64", device=self.device)
            else:
                raise ValueError("This calculator type is not included in this package")
        
        elif model_name == "GRACE" or model_name == "grace":
            from tensorpotential.calculator.foundation_models import grace_fm, GRACEModels
            if model_parameters.get("model_size") == "small" and model_parameters.get("layer") == 1:
                self.calc = grace_fm(GRACEModels.GRACE_1L_OMAT)
            else:
                raise ValueError("This calculator type is not included in this package")
        
        else:
            raise ValueError(f"Calculator '{model_name}' is not supported")
        
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
    
    def build_system(self, salt_anion, salt_cation, anion_Natoms, cation_Natoms, density_guess):
        """
        Build a molten salt system with random initial positions.
        
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
        
        Returns:
        --------
        atoms : ASE Atoms object
            The constructed system
        """
        if len(salt_anion) != len(anion_Natoms) or len(salt_cation) != len(cation_Natoms):
            raise ValueError("The number of salts and their number of atoms should be the same")
        
        # Create symbols list
        symbols = []
        for element, amount_of_atoms in zip(salt_anion, anion_Natoms):
            symbols += [element] * amount_of_atoms
        for element, amount_of_atoms in zip(salt_cation, cation_Natoms):
            symbols += [element] * amount_of_atoms
        
        # Calculate initial box size from density guess
        mass = sum(atomic_masses[atomic_numbers[sym]] for sym in symbols) * 1.66054e-24  # g
        volume_guess = mass / density_guess  # cm³
        initial_box_size = (volume_guess * 1e24) ** (1/3)  # Å
        
        # Place atoms with minimum distance constraint
        min_distance = 1.6  # Å
        positions_atoms = np.zeros((len(symbols), 3))
        
        for i in range(len(symbols)):
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
            pbc=True
        )
        
        if self.calc:
            atoms.calc = self.calc
        else:
            raise RuntimeError("Calculator not set. Use set_calculator() first.")
        
        return atoms
    
    def run_npt_simulation(self, atoms, T, steps=1000, print_interval=100, 
                          traj_file="npt_simulation.traj", print_status=True):
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
            logfile='npt_equili.log'
        )
        
        trajectory_npt = Trajectory(traj_file, "w", atoms)
        dyn.attach(trajectory_npt.write, interval=10)
        
        if print_status:
            def print_status_func():
                step = dyn.get_number_of_steps()
                stress_tensor = atoms.get_stress(voigt=False) * 1.60218e6
                pressure = -np.trace(stress_tensor) / 3
                p_xy, p_xz, p_yz = stress_tensor[0, 1], stress_tensor[0, 2], stress_tensor[1, 2]
                print(f"Step {step:6d} | P = {pressure:.6e} bar | V = {atoms.get_volume():8.2f} Å³")
            
            dyn.attach(print_status_func, interval=print_interval)
        
        dyn.run(steps)
        trajectory_npt.close()
        print(f"NPT trajectory saved to {traj_file}")
        
        return atoms
    
    def run_nvt_simulation(self, atoms, T, steps=1000, print_interval=100,
                          traj_file="nvt_simulation.traj", print_status=True):
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
            logfile='nvt_run.log'
        )
        
        trajectory_nvt = Trajectory(traj_file, "w", atoms)
        dyn.attach(trajectory_nvt.write, interval=10)
        
        if print_status:
            def print_status_func():
                step = dyn.get_number_of_steps()
                stress_tensor = atoms.get_stress(voigt=False) * 1.60218e6
                pressure = -np.trace(stress_tensor) / 3
                print(f"Step {step:6d} | P = {pressure:.6e} bar | V = {atoms.get_volume():8.2f} Å³")
            
            dyn.attach(print_status_func, interval=print_interval)
        
        dyn.run(steps)
        trajectory_nvt.close()
        print(f"NVT trajectory saved to {traj_file}")