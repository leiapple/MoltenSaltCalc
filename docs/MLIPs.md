# MLIP Backends

This module contains implementations of supported machine-learned interatomic potentials (MLIPs).

All models are registered via a common interface and can be selected by name when initializing the simulator. Please note that the models have conflicting dependencies and are not automatically installed with the package. To use a model, install the corresponding package with `pip install moltensaltcalc[model_name]` and make sure to use separate environments for each MLIP.

## Available models

- GRACE
- FairChem
- MACE

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

## FairChem

Pre-trained universal models from the FairChem project. Pretrained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Description |
|-|-|-|-|
| `model_size` | `str` | `small`, `medium` | Size of the FairChem model |
| `model_version` | `str` | `1p1`, `1p2` | Version of the pretrained model |
| `model_task` | `str` | `omc`, `omol`, `odac`, `oc20`, `omat` | Task the model is trained for |

### Notes

- Medium models currently use version `1p1` internally
- Make sure to have access to the [UMA model repository](https://huggingface.co/facebook/UMA) and have logged in with e.g. `huggingface-cli login` once
- When FAIRCHEM is initialized, it would reset the seeds of at least the python `random` and `numpy.random` modules (see [issue #1896](https://github.com/facebookresearch/fairchem/issues/1896)). This is mitigated in moltensaltcalc by resetting to the original state after the model was loaded (see `models/fairchem.py` for details).

---

## GRACE

Foundation models from the [GRACE framework](https://github.com/ICAMS/grace-tensorpotential), supporting multiple sizes and layer configurations. Pretrained models are automatically downloaded and require no manual setup.

### Parameters

| Parameter | Type | Choices | Description |
|-|-|-|-|
| `model_size` | `str` | `small`, `medium`, `large` | Size of the model |
| `num_layers` | `int` | `1`, `2` | Number of message-passing layers |
| `model_task` | `str` | `OAM`, `OMAT` | Task the model is trained for |

### Notes

- Not all parameter combinations are valid
- Internally mapped to specific pretrained checkpoints

---

## MACE

MACE models require a user-provided model file.

### Parameters

| Parameter | Type | Choices | Description |
|-|-|-|-|
| `model_path` | `str` | — | Path to a `.model` file |
| `model_task` | `str` | `omat_pbe`, `omol`, `spice_wB97M`, `rgd1_b3lyp`, `oc20_usemppbe`, `matpes_r2scan` | Task head used by the model |

### Notes

- Pretrained models must be downloaded automatically
- See: [github.com/ACEsuit/mace-foundations](https://github.com/ACEsuit/mace-foundations)

---

## Implementation

::: moltensaltcalc.models.fairchem.build_fairchem

::: moltensaltcalc.models.grace.build_grace

::: moltensaltcalc.models.mace.build_mace
