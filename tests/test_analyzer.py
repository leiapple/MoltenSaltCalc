from pathlib import Path

import numpy as np
import pytest

import moltensaltcalc as msc

BASE = Path(__file__).parent
EQ_FRAC = 0.6


# =========================================================
# Fixtures
# =========================================================


@pytest.fixture
def analyzer():
    """Create a lightweight analyzer instance."""
    return msc.MoltenSaltAnalyzer(
        traj_files_npt=[
            BASE / "test_analyzer_trajectories" / "npt_NaCl_1100K.traj",
            BASE / "test_analyzer_trajectories" / "npt_NaCl_1150K.traj",
            BASE / "test_analyzer_trajectories" / "npt_NaCl_1200K.traj",
        ],
        traj_files_nvt=[
            BASE / "test_analyzer_trajectories" / "nvt_NaCl_1100K.traj",
            BASE / "test_analyzer_trajectories" / "nvt_NaCl_1150K.traj",
            BASE / "test_analyzer_trajectories" / "nvt_NaCl_1200K.traj",
        ],
        temperatures_npt=[1100, 1150, 1200],
        temperatures_nvt=[1100, 1150, 1200],
    )


# =========================================================
# Basic functionality tests
# =========================================================


def test_select_trajectory_preference(analyzer):
    traj, _ = analyzer._select_trajectory("npt", T=1100)
    assert traj is analyzer.trajs_npt[0]

    traj, _ = analyzer._select_trajectory("nvt", T=1100)
    assert traj is analyzer.trajs_nvt[0]


def test_select_trajectory_only_nvt():
    analyzer = msc.MoltenSaltAnalyzer(
        traj_files_npt=[BASE / "test_analyzer_trajectories" / "npt_NaCl_1200K.traj"],
        traj_files_nvt=[BASE / "test_analyzer_trajectories" / "nvt_NaCl_1100K.traj"],
        temperatures_npt=[1200],
        temperatures_nvt=[1100],
    )
    traj, _ = analyzer._select_trajectory("npt", T=1100)
    assert traj is analyzer.trajs_nvt[0]


def test_init_string_input():
    analyzer = msc.MoltenSaltAnalyzer(
        traj_files_npt=str(BASE / "test_analyzer_trajectories" / "npt_NaCl_1100K.traj"),
        traj_files_nvt=str(BASE / "test_analyzer_trajectories" / "nvt_NaCl_1100K.traj"),
        temperatures_npt=[1100],
        temperatures_nvt=[1100],
    )
    assert len(analyzer.trajs_npt) == 1
    assert len(analyzer.trajs_nvt) == 1


def test_get_eq_times():
    analyzer = msc.MoltenSaltAnalyzer()
    times = np.array([0, 10, 20, 30, 40])
    idx = analyzer._get_eq_times(0.5, times)
    assert np.array_equal(idx, np.array([2, 3, 4]))


def test_trajectory_without_time_fs():
    # Prepare the traj
    with pytest.warns(UserWarning) as w:
        analyzer = msc.MoltenSaltAnalyzer(
            traj_files_npt=[
                BASE / "test_analyzer_trajectories" / "npt_NaCl_1100K_no_time_fs.traj"
            ],
            temperatures_npt=[1100],
        )
    assert len(w) == 1
    assert "WARNING: No time_fs found in" in str(w[0].message)
    assert np.allclose(analyzer.timestep_fs, np.diff(analyzer.times_fs_npt))
    analyzer.recompute_times(timestep_fs=5.0)
    assert np.isclose(analyzer.timestep_fs, 5.0)
    assert np.allclose(analyzer.timestep_fs, np.diff(analyzer.times_fs_npt))


def test_compute_density_vs_time(analyzer):
    densities, times = analyzer.compute_density_vs_time(1100)
    assert isinstance(densities, np.ndarray), "Densities is not a numpy array"
    assert isinstance(times, np.ndarray), "Times is not a numpy array"
    assert len(densities) == len(times), "Length of densities and times do not match"
    ini_density, ini_density_ref = densities[0], 1.54200
    assert np.isclose(
        ini_density, ini_density_ref, atol=1e-5
    ), f"Initial density is {ini_density:.5f} instead of {ini_density_ref:.5f}"
    final_density, final_density_ref = densities[-1], 1.50515
    assert np.isclose(
        final_density, final_density_ref, atol=1e-5
    ), f"Final density is {final_density:.5f} instead of {final_density_ref:.5f}"
    ini_time, ini_time_ref = times[0], 0.0
    assert np.isclose(
        ini_time, ini_time_ref, atol=1e-5
    ), f"Initial time is {ini_time:.5f} instead of {ini_time_ref:.5f}"
    final_time, final_time_ref = times[-1], 100.0
    assert np.isclose(
        final_time, final_time_ref, atol=1e-5
    ), f"Final time is {final_time:.5f} instead of {final_time_ref:.5f}"


def test_compute_eq_density(analyzer):
    eq_density = analyzer.compute_eq_density(1100, eq_fraction=EQ_FRAC)
    assert isinstance(eq_density, float), "Equilibrium density is not a float"
    eq_density_ref = 1.52253
    assert np.isclose(
        eq_density, eq_density_ref, atol=1e-5
    ), f"Equilibrium density is {eq_density:.5f} instead of {eq_density_ref:.5f}"


def test_compute_thermal_expansion(analyzer):
    thermal_expansion = analyzer.compute_thermal_expansion(eq_fraction=EQ_FRAC)
    assert isinstance(
        thermal_expansion, dict
    ), "Thermal expansion results are not returned as a dictionary"
    assert (
        "thermal_expansion" in thermal_expansion
    ), "Thermal expansion results do not contain the key 'thermal_expansion'"
    thm_exp_coeff, thm_exp_coeff_ref = thermal_expansion["thermal_expansion"], 0.00027
    assert np.isclose(
        thm_exp_coeff, thm_exp_coeff_ref, atol=1e-5
    ), f"Thermal expansion coefficient is {thm_exp_coeff:.5f} instead of {thm_exp_coeff_ref:.5f}"


def test_compute_heat_capacity(analyzer):
    heat_capacity = analyzer.compute_heat_capacity(T=1200, eq_fraction=EQ_FRAC)
    assert isinstance(heat_capacity, float), "Heat capacity is not a float"
    heat_capacity_ref = 0.00023
    assert np.isclose(
        heat_capacity, heat_capacity_ref, atol=1e-5
    ), f"Heat capacity is {heat_capacity:.5f} instead of {heat_capacity_ref:.5f}"


def test_compute_diffusion_coefficient_arr_fit(analyzer):
    diff_coeffs = []
    T_list = [1100, 1150, 1200]
    for T in T_list:
        diffusion_coeff = analyzer.compute_diffusion_coefficient(T=T)
        diff_coeffs.append(diffusion_coeff)
        if T == 1100:
            assert isinstance(
                diffusion_coeff, float
            ), "Diffusion coefficient is not a float for T = {T} K"
            diffusion_coeff_ref = 0.00098
            assert np.isclose(
                diffusion_coeff, diffusion_coeff_ref, atol=1e-5
            ), f"Diffusion coefficient at {T} K is {diffusion_coeff:.5f} instead of {diffusion_coeff_ref:.5f}"

    arr_fit = analyzer.fit_arrhenius(temperatures=T_list, diffusion_coeffs=diff_coeffs)
    assert isinstance(
        arr_fit, dict
    ), "Arrhenius fit results are not returned as a dictionary"
    assert "Ea" in arr_fit, "Arrhenius fit results do not contain the key 'Ea'"
    Ea, Ea_ref = arr_fit["Ea"], 20926.76718
    assert np.isclose(
        Ea, Ea_ref, atol=1e-1
    ), f"Activation energy is {Ea:.1f} instead of {Ea_ref:.1f}"


def test_compute_rdf(analyzer):
    nbins = 5
    rdf_data = analyzer.compute_rdf(
        max_num_frames=3, rmax=5.0, nbins=nbins, pairs=[(11, 11)], T=1200
    )
    assert isinstance(
        rdf_data, dict
    ), "Radial distribution function results are not returned as a dictionary"
    assert (
        11,
        11,
    ) in rdf_data, (
        "Radial distribution function results do not contain the key '(11, 11)'"
    )
    (distances, avg_rdf), distances_ref, avg_rdf_ref = (
        rdf_data[(11, 11)],
        [0.5, 1.5, 2.5, 3.5, 4.5],
        [0.0, 0.0, 0.01664065, 1.17638912, 1.66206463],
    )
    assert isinstance(distances, np.ndarray), "Distances are not a numpy array"
    assert isinstance(avg_rdf, np.ndarray), "Average RDF is not a numpy array"
    assert len(distances) == len(
        avg_rdf
    ), f"Length of distances ({len(distances)}) and RDF ({len(avg_rdf)}) do not match"
    assert (
        len(distances) == nbins
    ), f"Length of distances ({len(distances)}) and set number of bins ({nbins}) do not match"
    assert np.allclose(
        distances, distances_ref, atol=1e-5
    ), f"Distances are {distances} instead of {distances_ref}"
    assert np.allclose(
        avg_rdf, avg_rdf_ref, atol=1e-5
    ), f"Average RDF is {avg_rdf} instead of {avg_rdf_ref}"


def test_rdf_auto_pairs(analyzer):
    rdf = analyzer.compute_rdf(T=1100, max_num_frames=2, nbins=2, rmax=2.0)
    assert isinstance(rdf, dict)
    rdf_keys, rdf_keys_ref = set(rdf.keys()), set([(11, 11), (11, 17), (17, 17)])
    assert (
        rdf_keys == rdf_keys_ref
    ), f"RDF auto-pairs {rdf_keys} do not match expected {rdf_keys_ref}"


def test_autocorr_fft_known_signal(analyzer):
    x = np.array([1.0, 2.0, 3.0])
    ac = analyzer._autocorr_fft(x, nmax=3)
    expected = np.array([14 / 3, 4.0, 3.0])
    assert np.allclose(
        ac, expected
    ), f"Autocorrelation function is {ac} instead of {expected}"


def test_compute_viscosity(analyzer):
    viscosity = analyzer.compute_viscosity(T=1200, tmax_fs=41)
    assert isinstance(viscosity, tuple), "Viscosity results are not returned as a tuple"
    assert len(viscosity) == 2, "Viscosity results do are not of expected length 2"
    eta, eta_ref = viscosity[0], 0.00015
    assert np.isclose(
        eta, eta_ref, atol=1e-5
    ), f"Viscosity is {eta:.5f} instead of {eta_ref:.5f}"
    (autocorrelation, times), autocorrelation_ref, times_ref = (
        viscosity[1],
        [6.78166120e-07, 4.15856007e-07, -5.36254178e-08],
        [0.0, 20.0, 40.0],
    )
    assert np.allclose(
        autocorrelation, autocorrelation_ref, atol=1e-5
    ), f"Autocorrelation function is {autocorrelation} instead of {autocorrelation_ref}"
    assert np.allclose(
        times, times_ref, atol=1e-5
    ), f"Autocorrelation times are {times} instead of {times_ref}"


# =========================================================
# Error handling tests
# =========================================================


def test_init_missing_traj_file():
    with pytest.raises(FileNotFoundError) as e:
        msc.MoltenSaltAnalyzer(
            traj_files_npt=["nonexistent.traj"],
            temperatures_npt=[1100],
        )
    assert f"Trajectory file nonexistent.traj not found" in str(e.value)


def test_invalid_temperature(analyzer):
    with pytest.raises(ValueError) as e:
        analyzer.compute_diffusion_coefficient(T=1000)
    assert "not found in any trajectories" in str(e.value)


def test_init_mismatched_lengths():
    with pytest.raises(ValueError) as e:
        msc.MoltenSaltAnalyzer(
            traj_files_npt=["a.traj", "b.traj"],
            temperatures_npt=[1100],
        )
    assert "Number of NPT trajectory files and temperatures_npt must match" in str(
        e.value
    )


def test_init_missing_temperatures():
    with pytest.raises(ValueError) as e:
        msc.MoltenSaltAnalyzer(traj_files_npt=["a.traj"])
    assert "Number of NPT trajectory files and temperatures_npt must match" in str(
        e.value
    )


def test_select_no_trajectory():
    analyzer = msc.MoltenSaltAnalyzer()
    with pytest.raises(ValueError) as e:
        analyzer._select_trajectory("npt", T=1100)
    assert "No trajectory files provided" in str(e.value)


def test_eq_density_invalid_fraction(analyzer):
    with pytest.raises(ValueError) as e:
        analyzer.compute_eq_density(T=1100, eq_fraction=1.5)
    assert "eq_fraction must be between 0 and 1." in str(e.value)


def test_invalid_thm_expansion(analyzer, monkeypatch):
    monkeypatch.setattr(analyzer, "trajs_npt", analyzer.trajs_nvt[:1])
    with pytest.raises(ValueError) as e:
        analyzer.compute_thermal_expansion(eq_fraction=0.1)
    assert "At least two NPT trajectory files are required" in str(e.value)
    monkeypatch.setattr(analyzer, "temperatures_npt", None)
    with pytest.raises(ValueError) as e:
        analyzer.compute_thermal_expansion(eq_fraction=0.1)
    assert "No NPT temperatures provided" in str(e.value)
    monkeypatch.setattr(analyzer, "trajs_npt", None)
    with pytest.raises(ValueError) as e:
        analyzer.compute_thermal_expansion(eq_fraction=0.1)
    assert "No NPT trajectory files provided" in str(e.value)


def test_rdf_no_pairs(analyzer):
    with pytest.raises(ValueError) as e:
        analyzer.compute_rdf(T=1100, max_num_frames=2, nbins=2, rmax=2.0, pairs=[])
    assert "No pairs specified" in str(e.value)


def test_viscosity_nonconstant_timestep():
    # Same as the 1200 K trajectory used above, but with the first timestep modified to 21 fs instead of 20 fs
    analyzer = msc.MoltenSaltAnalyzer(
        traj_files_nvt=[
            BASE
            / "test_analyzer_trajectories"
            / "nvt_NaCl_1200K_nonconstant_timestep.traj",
        ],
        temperatures_nvt=[1200],
    )
    with pytest.raises(ValueError) as e:
        analyzer.compute_viscosity(T=1200)
    assert "The timestep between the frames is not constant" in str(e.value)
