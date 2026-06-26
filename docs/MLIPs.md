# MLIP Backends

This module contains implementations of supported machine-learned interatomic potentials (MLIPs).

All models are registered via a common interface and can be selected by name when initializing the simulator. Please note that the models have conflicting dependencies and are not automatically installed with the package. To use a model, install the corresponding package with `pip install moltensaltcalc[model_name]` and make sure to use separate environments for each MLIP.

## Available models

- [7net](#7net)
- [CHGNet](#chgnet)
- [FairChem](#fairchem)
- [GRACE](#grace)
- [MACE](#mace)
- [MatterSim](#mattersim)
- [NequIP](#nequip)
- [Nequix](#nequix)
- [UPET](#upet)

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
)
```

### Notes

- The exact parameters depend on the selected model (see below)
- Models are loaded lazily during initialization
- Make sure the required backend is installed (e.g. `moltensaltcalc[fairchem]`)

---

## 7net

Pre-trained universal models from the [7net project](https://github.com/MDIL-SNU/SevenNet). Pretrained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_name` | `str` | `7net-omni`, `7net-mf-ompa`, `7net-omat24`, `7net-l3i5`, `7net-0` | `7net-omni` | Name of the pretrained model |
| `model_task` | `str` | `mpa`, `omat24`, `matpes_pbe`, `matpes_r2scan`, `mp_r2scan`, `oc20`, `oc22`, `odac23`, `omol25_low`, `omol25_high`, `spice`, `qcml`, `pet_mad` | `omat24` | Task head used by the model |

## CHGNet

Pre-trained universal models from the [CHGNet project](https://github.com/CederGroupHub/chgnet). Pretrained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_name` | `str` | `0.3.0`, `0.2.0`, `r2scan` | `0.3.0` | Name of the pretrained CHGNet model |

---

## FairChem

Pre-trained universal models from the FairChem project. Pretrained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_size` | `str` | `s`, `m` | `s` | Size of the FairChem model |
| `model_version` | `str` | `1p1`, `1p2` | `1p2` | Version of the pretrained model |
| `model_task` | `str` | `omc`, `omol`, `odac`, `oc20`, `omat` | `omat` | Task the model is trained for |
| `InferenceSettings` | `fairchem.core.units.mlip_unit.api.inference.InferenceSettings` | `...` | `Turbo settings from FAIRCHEM with compile=False` | Settings for the inference of the FAIRCHEM model |

### Notes

- Medium models are currently only available as `1p1`
- Make sure to have access to the [UMA model repository](https://huggingface.co/facebook/UMA) and have logged in with e.g. `huggingface-cli login` once
- When FAIRCHEM is initialized, it would reset the seeds of at least the python `random` and `numpy.random` modules (see [issue #1896](https://github.com/facebookresearch/fairchem/issues/1896)). This is mitigated in moltensaltcalc by resetting to the original state after the model was loaded (see `models/fairchem.py` for details).
- Compile in the `turbo_settings` is disabled by default, but can be enabled by setting `compile=True` in the `InferenceSettings` parameter. The turbo settings work only if the composition is constant.

---

## GRACE

Foundation models from the [GRACE framework](https://github.com/ICAMS/grace-tensorpotential), supporting multiple sizes and layer configurations. Pretrained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_size` | `str` | `small`, `medium`, `large` | `small` | Size of the model |
| `num_layers` | `int` | `1`, `2` | `1` | Number of message-passing layers |
| `model_task` | `str` | `OAM`, `OMAT` | `OMAT` | Task the model is trained for |

---

## MACE

Foundation models from the [MACE framework](https://github.com/ACEsuit/mace). Pretrained models can be automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_path` | `str` | `...` | `https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model?raw=true` | Path or URL to a `.model` file |
| `model_task` | `str` | `omat_pbe`, `omol`, `spice_wB97M`, `rgd1_b3lyp`, `oc20_usemppbe`, `matpes_r2scan` | `default` | Task head used by the model |

### Notes

- Pretrained models must be downloaded automatically
- See: [github.com/ACEsuit/mace-foundations](https://github.com/ACEsuit/mace-foundations)
- Not all combinations (specifically `model_task`) are available from MACE

---

## MatterSim

Pre-trained universal models from the [MatterSim project](https://github.com/microsoft/mattersim). Pretrained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_path` | `str` | `...` | `None` | Path to a pytorch model file, e.g. 'MatterSim-v1.0.0-5M.pth' that can be downloaded from https://github.com/microsoft/mattersim. If `None` is provided, the model is automatically downloaded. |

---

## NequIP

Pre-trained universal models from the [NequIP project](https://github.com/mir-group/nequip). Pretrained models required manual precompilation. A description how to compile the model can be found at [nequip.net/models](https://www.nequip.net/models).

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_path` | `str` | `...` | `None` | Path to a precompiled NequIP model file (e.g. `nequip_models/mir-group__NequIP-OAM-S__0.1.nequip.pth`). The filename must end with `.nequip.pth` (torchvision) or `.nequip.pt2` (aotinductor). |

---

## Nequix

Pre-trained universal models from the [Nequix project](https://github.com/atomicarchitects/nequix). Pretrained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_task` | `str` | `mp`, `omat`, `oam` | `omat` | Task the model is trained for |
| `model_path` | `str` | `...` | `None` | Provide the path to a Nequix model file. Overrides the `model_task` parameter. |
| `model_backend` | `str` | `torch`, `jax` | `jax` | Backend to use for the Nequix model |

---

## UPET

Pre-trained universal models from the [UPET project](https://github.com/lab-cosmo/upet). Pretrained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_task` | `str` | `omat`, `oam`, `mad`, `omatpes`, `omad`, `spice` | `omat` | Task the model is trained for |
| `model_size` | `str` | `xs`, `s`, `m`, `l`, `xl` | `s` | Size of the UPET model. |
| `model_version` | `str` | `latest`, `v0.1.0`, `v0.2.0` | `latest` | Version of the pretrained UPET model |
| `checkpoint_path` | `str` | `...` | `None` | Path to a pretrained UPET model checkpoint file, optional |

---

## EquiFormer V3

Pre-trained universal models from [EquiFormer V3](https://github.com/atomicarchitects/equiformer_v3), as packaged in [equiformer-v3 python package](https://github.com/daniisler/equiformer_v3/tree/feature-python-package).

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_path` | `str` | `omat24_direct`, `omat24_gradient`, `omat24-mptrj-salex_gradient`, `mptrj_gradient` | `omat24_direct` | Path to a local file path (e.g. 'models/omat24_direct.pt') or a string specifier (`omat24_direct`). |
| `model_revision` | `str` | `main` | `main` | Revision of the model to download from HuggingFace. |
| `dont_clean_model` | `bool` | `False` | `False` | Whether to skip cleaning the model by stripping the `_orig_mod.module.` from the keys (necessary for the current model version). |

---

## EquFlash / GGNN

Pre-trained universal models from the [GGNN](https://github.com/SamsungDS/GGNN) project, as packaged in [GGNN python package](https://github.com/daniisler/GGNN/tree/feature-python-package).

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_path` | `str` | `equflash`, `equflash_v2` | `None` | Path to a local file path (e.g. 'models/GGNN.pt') or a string specifier. |

---

## TACE

Models from the [TACE](https://github.com/xvzemin/tace) project with pre-trained models from [TACE-foundations](https://github.com/xvzemin/tace-foundations).

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_name` | `str` | `TACE-v1-OMat24-M`, `TACE-v1-OAM-M`, `TACE-v1-LES-REICO-5-PdAgCHO.pt` | `TACE-v1-OMat24-M` | Name of pre-trained model or path to the checkpoint, deployed model or the model itself. |

---

## Orbital Models

Pre-trained universal models from the [ORB Models](https://github.com/orbital-materials/orb-models) project.

### Parameters

| Parameter | Type | Choices | Default | Description |
|-|-|-|-|-|
| `model_task` | `str` | `omat`, `mpa` | `omat` | Task the model is trained for. |
| `max_neighbors` | `str` | `20`, `inf` | `20` | Maximum number of neighbors to consider for the model. |
| `model_type` | `str` | `direct`, `conservative` | `conservative` | How inference of the forces is achieved. |

---
