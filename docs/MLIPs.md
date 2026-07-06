# MLIP Backends

This module contains implementations of supported machine-learned interatomic potentials (MLIPs).

All models are registered via a common interface and can be selected by name when initializing the simulator. Please note that the models have conflicting dependencies and are not automatically installed with the package. To use a model, install the corresponding package with `pip install moltensaltcalc[model_name]` and make sure to use separate environments for each MLIP.

## Available models

- [7net](#7net)
- [AlphaNet](#alphanet)
- [CHGNet](#chgnet)
- [DeePMD (DPA)](#deepmd-dpa)
- [Eqnorm](#eqnorm)
- [Equflash (GGNN)](#equflash)
- [Equiformer V3](#equiformer-v3)
- [FairChem](#fairchem)
- [GRACE](#grace)
- [HIENet](#hienet)
- [MACE](#mace)
- [MatGL](#matgl)
- [MatRIS](#matris)
- [MatterSim](#mattersim)
- [NequIP](#nequip)
- [Nequix](#nequix)
- [Orbitals (orb_v3)](#orbitals-orb_v3)
- [TACE](#tace)
- [UPET](#upet)
- [VASP](#vasp)

The models can be listed with:

```python
import moltensaltcalc as msc
msc.available_models()
```

## Loading Models in the MoltenSaltSimulator

Models are automatically constructed when initializing the simulator by specifying the model name and corresponding parameters.

### Example

```python
from moltensaltcalc.simulator import MoltenSaltSimulator

sim = MoltenSaltSimulator(
    model="fairchem",
    model_params={
        "model_size": "small",
        "model_version": "1p1",
        "model_task": "omat",
    },
    dispersion="DFTD3",
    device="cuda:0",
)
```

### Notes

- The exact parameters depend on the selected model (see below)
- Models are loaded lazily during initialization
- Make sure the required backend is installed (e.g. `moltensaltcalc[fairchem]`)
- The model parameters are passed as a dictionary to the `model_params` argument. The keys and values depend on the selected model (see below).
- The dispersion model can be specified with the `dispersion` argument, which implements the [TorchDFTD3Calculator](https://github.com/pfnet-research/torch-dftd). Options are `DFTD2`, `DFTD3`, `DFTD4` and `None` (no dispersion). `DFTD4` is quite slow compared to `DFTD3`.
- The device can be specified with the `device` argument, which can be a string like `cpu`, `cuda:0`, `cuda:1`, etc. The default is `cuda` and if no `cuda` device is available, it will fall back to `cpu`. The device is selected by specifying the environment variable `CUDA_VISIBLE_DEVICES`, meaning that it will only ignore the other GPUs if `torch` was not loaded before the calculator is initialized.

---

## 7net

Pre-trained universal models from the [7net project](https://github.com/MDIL-SNU/SevenNet). Pre-trained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_name` | `str` | `7net-omni`, `7net-mf-ompa`, `7net-omat24`, `7net-l3i5`, `7net-0` | `7net-omni` | Name of the pre-trained model. |
| `model_task` | `str` | `mpa`, `omat24`, `matpes_pbe`, `matpes_r2scan`, `mp_r2scan`, `oc20`, `oc22`, `odac23`, `omol25_low`, `omol25_high`, `spice`, `qcml`, `pet_mad` | `omat24` | Task head used by the model. |

---

## AlphaNet

Pre-trained universal models from the [AlphaNet project](https://github.com/zmyybc/AlphaNet), as packaged in [AlphaNet python package](https://github.com/daniisler/AlphaNet/tree/feature-python-package).

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_path` | `str` | `AlphaNet-oma-v1`, `AlphaNet-MPtrj-v1` | `AlphaNet-oma-v1` | Path to a local file or a string specifier from the choice list which leads to a download from figshare. |

---

## CHGNet

Pre-trained universal models from the [CHGNet project](https://github.com/CederGroupHub/chgnet). Pre-trained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_name` | `str` | `0.3.0`, `0.2.0`, `r2scan` | `0.3.0` | Name of the pre-trained CHGNet model. |

---

## DeePMD (DPA)

Pre-trained universal models from the [DeePMD project](https://github.com/deepmodeling/deepmd-kit).

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_path` | `str` | `DPA-3.3-1M`, `DPA-2.4-7M`, `DPA-3.1-3M-FT`, `DPA3-Omol-Large`, `DPA-3.2-5M`, `DPA-3.1-3M` | `DPA-3.3-1M` | Path to a local file or a string specifier from the choice list which leads to a download from huggingface/figshare. |
| `model_task` | `str` | `Domains_Alloy`, `Domains_Anode`, `Domains_Cluster`, `Domains_FerroEle`, `Domains_SSE_PBE`, `Domains_SemiCond`, `H2O_H2O_PD`, `Metals_AlMgCu`, `Metals_AgAu_PBED3`, `Others_In2Se3`, `MPGen_OpenCSP`, `Alloy_tongqi`, `SSE_ABACUS`, `Hybrid_Perovskite`, `Electrolyte`, `ODAC23`, `Alex2D`, `Omat24`, `OC20M`, `OC22`, `Organic_Reactions`, `OMol25`, `MPTrj` | `Omat24` | Task head used by the model. |

---

## Eqnorm

Pre-trained universal models from the [Eqnorm project](https://github.com/yzchen08/eqnorm), as packaged in [Eqnorm python package](https://github.com/daniisler/Eqnorm/tree/feature-python-package).

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_name` | `str` | `eqnorm` | `eqnorm` | Name of the pre-trained Eqnorm model. |
| `model_task` | `str` | `eqnorm-omat`, `eqnorm-mptrj`, `eqnorm-max-mptrj` | `eqnorm-mptrj` | Task head used by the model. |

---

## Equflash (GGNN)

Pre-trained universal models from the [Equflash (GGNN) project](https://github.com/SamsungDS/GGNN), as packaged in [Equflash python package](https://github.com/daniisler/GGNN/tree/feature-python-package).

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_path` | `str` | `equflash`, `equflash-omat`, `equflash_v2`, `equflash_v2-omat` | `equflash_v2-omat` | Path to a local file or a string specifier from the choice list which leads to a download from figshare. |

---

## Equiformer V3

Pre-trained universal models from [EquiFormer V3](https://github.com/atomicarchitects/equiformer_v3), as packaged in [equiformer-v3 python package](https://github.com/daniisler/equiformer_v3/tree/feature-python-package).

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_path` | `str` | `omat24_direct`, `omat24_gradient`, `omat24-mptrj-salex_gradient`, `mptrj_gradient` | `omat24_direct` | Path to a local file or a string specifier from the choice list which leads to a download from HuggingFace. |
| `model_revision` | `str` | `main` | `main` | Revision of the model to download from HuggingFace. |
| `dont_clean_model` | `bool` | `False` | `False` | Whether to skip cleaning the model by stripping the `_orig_mod.module.` from the keys (necessary for the current model version from HuggingFace). |

---

## FairChem

Pre-trained universal models from the FairChem project. Pre-trained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_size` | `str` | `s`, `m` | `s` | Size of the FairChem model |
| `model_version` | `str` | `1p1`, `1p2` | `1p2` | Version of the pre-trained model |
| `model_task` | `str` | `omc`, `omol`, `odac`, `oc20`, `omat` | `omat` | Task the model is trained for |
| `InferenceSettings` | `fairchem.core.units.mlip_unit.api.inference.InferenceSettings` | `...` | `Turbo settings from FAIRCHEM with compile=False` | Settings for the inference of the FAIRCHEM model. |

### Notes

- Medium models are currently only available as `1p1`
- Make sure to have access to the [UMA model repository](https://huggingface.co/facebook/UMA) and have logged in with e.g. `huggingface-cli login` once
- When FAIRCHEM is initialized, it would reset the seeds of at least the python `random` and `numpy.random` modules (see [issue #1896](https://github.com/facebookresearch/fairchem/issues/1896)). This is mitigated in moltensaltcalc by resetting to the original state after the model was loaded (see `models/fairchem.py` for details).
- Compile in the `turbo_settings` is disabled by default, but can be enabled by setting `compile=True` in the `InferenceSettings` parameter. The turbo settings work only if the composition is constant.

---

## GRACE

Foundation models from the [GRACE framework](https://github.com/ICAMS/grace-tensorpotential), supporting multiple sizes and layer configurations. Pre-trained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_size` | `str` | `small`, `medium`, `large` | `small` | Size of the model. |
| `num_layers` | `int` | `1`, `2` | `1` | Number of message-passing layers. |
| `model_task` | `str` | `OAM`, `OMAT` | `OMAT` | Task the model is trained for. |

---

## HIENet

Pre-trained universal models from the [AIRS](https://github.com/divelab/AIRS) OpenMat [HIENet](https://github.com/divelab/AIRS/tree/main/OpenMat/HIENet), as packaged in [HIENet python package](https://github.com/daniisler/AIRS/tree/feature-python-package/OpenMat/HIENet).

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_path` | `str` | `HIENet-0` | `HIENet-0` | Path to a checkpoint file or a string specifier from the choice list which leads to choose the model contained in the package. |

---

## MACE

Foundation models from the [MACE framework](https://github.com/ACEsuit/mace). Pre-trained models can be automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_path` | `str` | `...` | `https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model?raw=true` | Path or URL to a `.model` file. |
| `model_task` | `str` | `omat_pbe`, `omol`, `spice_wB97M`, `rgd1_b3lyp`, `oc20_usemppbe`, `matpes_r2scan` | `default` | Task head used by the model. |

### Notes

- Pre-trained models must be downloaded automatically
- See: [github.com/ACEsuit/mace-foundations](https://github.com/ACEsuit/mace-foundations)
- Not all combinations (specifically `model_task`) are available from MACE

---

## MatGL

Pre-trained universal models from the [MatGL library](https://github.com/materialsvirtuallab/matgl).

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_name` | `str` | `CHGNet-PES-MatPES-PBE-2025.2.10`, `CHGNet-PES-MatPES-R2SCAN-2025.2.10`, `M3GNet-Eform-MP-2018.6.1`, `M3GNet-PES-MatPES-PBE-2025.2`, `M3GNet-PES-MatPES-R2SCAN-2025.2`, `MEGNet-BandGap-mfi-MP-2019.4.1`, `MEGNet-Eform-MP-2018.6.1`, `QET-PES-MatPES-PBE-2025.2`, `QET-PES-MatPES-R2SCAN-2025.2`, `QET-PES-MatQ0`, `SO3Net-PES-ANI-1x-Subset`, `TensorNet-PES-ANI-1x-Subset`, `TensorNet-PES-MatPES-PBE-2025.2`, `TensorNet-PES-MatPES-PBE-2025.2-m`, `TensorNet-PES-MatPES-R2SCAN-2025.2`, `TensorNet-PES-MatPES-R2SCAN-2025.2-m` | `QET-PES-MatQ0` | The name of the model to use. |

---

## MatRIS

Pre-trained universal models from the [MatRIS project](https://github.com/HPC-AI-Team/MatRIS) as packaged in [MatRIS python package](https://github.com/daniisler/MatRIS/tree/model-download-fixes).

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_name` | `str` | `matris_10m_oam`, `matris_10m_mp` | `matris_10m_oam` | The name of the model to use which will be downloaded from figshare. |

---

## MatterSim

Pre-trained universal models from the [MatterSim project](https://github.com/microsoft/mattersim). Pre-trained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_path` | `str` | `...` | `None` | Path to a pytorch model file, e.g. 'MatterSim-v1.0.0-5M.pth' that can be downloaded from https://github.com/microsoft/mattersim. If `None` is provided, the model is automatically downloaded. |

---

## NequIP

Pre-trained universal models from the [NequIP project](https://github.com/mir-group/nequip). Pre-trained models required manual precompilation. A description how to compile the model can be found at [nequip.net/models](https://www.nequip.net/models).

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_path` | `str` | `...` | `None` | Path to a precompiled NequIP model file (e.g. `nequip_models/mir-group__NequIP-OAM-S__0.1.nequip.pth`). The filename must end with `.nequip.pth` (torchvision) or `.nequip.pt2` (aotinductor). |

---

## Nequix

Pre-trained universal models from the [Nequix project](https://github.com/atomicarchitects/nequix). Pre-trained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_task` | `str` | `mp`, `omat`, `oam` | `omat` | Task the model is trained for. |
| `model_path` | `str` | `...` | `None` | Provide the path to a Nequix model file. Overrides the `model_task` parameter. |
| `model_backend` | `str` | `torch`, `jax` | `jax` | Backend to use for the Nequix model. |

---

## Orbitals (orb_v3)

Pre-trained universal models from the [Orbitals](https://github.com/orbital-materials/orb-models) project.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_task` | `str` | `mpa`, `omat` | `omat` | Task the model is trained for. |
| `max_neighbors` | `str` | `20`, `inf` | `20` | Maximum number of neighbors to consider for the model. |
| `model_type` | `str` | `direct`, `conservative` | `conservative` | How inference of the forces is achieved. |

---

## TACE

Models from the [TACE](https://github.com/xvzemin/tace) project with pre-trained models from [TACE-foundations](https://github.com/xvzemin/tace-foundations).

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_name` | `str` | `TACE-v1-OMat24-M`, `TACE-v1-OAM-M`, `TACE-v1-LES-REICO-5-PdAgCHO.pt` | `TACE-v1-OMat24-M` | Name of pre-trained model or path to the checkpoint, deployed model or the model itself. |

---

## UPET

Pre-trained universal models from the [UPET project](https://github.com/lab-cosmo/upet). Pre-trained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_task` | `str` | `omat`, `oam`, `mad`, `omatpes`, `omad`, `spice` | `omat` | Task the model is trained for. |
| `model_size` | `str` | `xs`, `s`, `m`, `l`, `xl` | `s` | Size of the UPET model. |
| `model_version` | `str` | `latest`, `v0.1.0`, `v0.2.0` | `latest` | Version of the pre-trained UPET model |
| `checkpoint_path` | `str` | `...` | `None` | Path to a pre-trained UPET model checkpoint file, optional. |

---

## VASP

Handler for the [VASP](https://www.vasp.at/) DFT code, needs the executable to be in the `PATH` environment variable.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `command` | `str` | `mpirun vasp_std`, `vasp_std`, `vasp_gam`, `vasp_ncl`, `Disable` | `mpirun vasp_std` | VASP command to use. `Disable` will lead to use from the `ASE_VASP_COMMAND` environment variable. |
| `xc` | `str` | `PBE`, `LDA`, `revPBE`, `RPBE` | `PBE` | Exchange-correlation functional. |
| `encut` | `int` | `...` | `400` | Plane-wave cutoff energy in eV. |
| `ivdw` | `int` | `...` | `12` | Van der Waals dispersion correction (e.g., 12 for Grimme D3 with `bj` damping). |
| `kpts` | `list` | `...` | `[1, 1, 1]` | K-point grid as a list of 3 integers. |
| `nelm` | `int` | `...` | `60` | Maximum electronic SC cycles per ionic step. |
| `pp_version` | `int | str` | `...` | `""` | The pseudopotential version (suffix) to use, such as `''`, `52`, `54`, `64`. |
| `prec` | `str` | `Accurate`, `Normal`, `Low` | `Accurate` | Accuracy to use for the calculations. |
| `ismear` | `int` | `...` | `0` | Gaussian smearing (ideal for liquids/molten salts). |
| `sigma` | `float` | `...` | `0.05` | Smearing width. |
| `lreal` | `str | bool` | `Auto`, `On`, True, False` | `Auto` | Evaluate projection operators in real space for speed. |
| `algo` | `str` | `VeryFast`, `Fast`, `Normal` | `VeryFast` | Electronic minimization algorithm: RMM-DIIS electronic optimization for fast MD steps, hybrid approach or blocked-Davidson. |
