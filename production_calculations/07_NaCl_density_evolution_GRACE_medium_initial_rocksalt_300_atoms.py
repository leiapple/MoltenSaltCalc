# The program was killed after 2 hours (~1000 steps = 1 ps on my laptop) and produced the plot test_sim/plots/density_evolution_NaCl_1100K_1ps

import os

import numpy as np

import moltensaltcalc as msc

np.random.seed(42)  # For reproducibility of initial random placements

# ========================================================================
# Generate the trajectories
# ========================================================================

# Setup the MS simulator class with the desired model and parameters
identifier = "initial_rocksalt_300_atoms"
n_steps = 200000  # 1 step is 1 fs, so to get the 200 ps, we need 200000 steps
print_interval = 1000
# Define salts to simulate like:   "salt_name": ([anions], [cations], amount_of_anions, amount_of_cations)
salts = {
    "NaCl": (["Cl"], ["Na"], [150], [150]),
}
# Define at which temperatures you want to calculate the properties per salt
temperatures = {
    "NaCl": [1100, 1125, 1150, 1175, 1200],
}
# Define what density you guess the salt to have at the corresponding temperatures
density_guesses = {
    "NaCl": [1.542, 1.528, 1.515, 1.501, 1.488],
}

# ========================================================================
# Run the simulations
# ========================================================================
print("Loading the simulator module...")
sim = msc.MoltenSaltSimulator(
    model_name="GRACE", model_parameters={"model_size": "medium", "layer": 1}
)
print("\n\n\nStarting the simulations...\n")
for salt_name, (anions, cations, n_anions, n_cations) in salts.items():
    print(f"\n\nRunning NPT simulations for {salt_name}...\n")

    # Create folders to store the trajectories
    npt_dir, nvt_dir = sim.create_simulation_folder(
        base_name=os.path.join(
            "simulation_results", identifier, f"GRACE_1L_{salt_name}"
        )
    )

    # Pair each temperature with its corresponding density guess
    for T, density_guess in zip(temperatures[salt_name], density_guesses[salt_name]):
        atoms = sim.build_system(
            anions, cations, n_anions, n_cations, density_guess, lattice="rocksalt"
        )
        traj_file_npt = os.path.join(npt_dir, f"npt_{salt_name}_{T}K.traj")
        sim.run_npt_simulation(
            atoms,
            T,
            steps=n_steps,
            print_interval=print_interval,
            traj_file=traj_file_npt,
            print_status=True,
        )
        traj_file_nvt = os.path.join(nvt_dir, f"nvt_{salt_name}_{T}K.traj")
        sim.run_nvt_simulation(
            atoms,
            T,
            steps=n_steps,
            print_interval=print_interval,
            traj_file=traj_file_nvt,
            print_status=True,
        )

    # ========================================================================
    # Analyze the trajectories
    # ========================================================================

    print("\n\nAnalyzing the results of the Simulation...\n")

    analyzer = msc.MoltenSaltAnalyzer()

    # Ensure the plot directory exists
    os.makedirs(os.path.join("plots", identifier), exist_ok=True)

    for salt_name in salts.keys():
        for T in temperatures[salt_name]:
            npt_traj = os.path.join(npt_dir, f"npt_{salt_name}_{T}K.traj")

            # ===================================================================================
            #   Density vs. Time
            # ===================================================================================
            densities = analyzer.compute_plot_density_vs_time(
                npt_traj,
                title=f"Density Evolution — {salt_name} — {T}K",
                fig_path=os.path.join(
                    "plots", identifier, f"density_evolution_{salt_name}_{T}K.png"
                ),
            )
