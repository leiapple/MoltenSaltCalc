"""Implementation of the EquFlashV2 (fairchem) MLIP."""

from moltensaltcalc.registry import register_model

AVAILABLE_MODELS_DICT = {
    "equflash": "https://api.figshare.com/v2/file/download/65435004",
    "equflash_v2": "https://api.figshare.com/v2/file/download/65435007",
}


def download_model(model_url: str):
    """Downloads the model from figshare, inspired from fairchem."""
    import shutil
    from pathlib import Path

    import requests
    from tqdm.auto import tqdm

    cache_dir = Path.home() / ".cache" / "equflash"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / Path(model_url).name
    if not local_path.exists():
        local_path_tmp = local_path.with_suffix(local_path.suffix + ".tmp")
        session = requests.Session()
        response = session.get(model_url, stream=True, allow_redirects=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with (
            open(local_path_tmp, "wb") as f,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=local_path.name,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=1024**2):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        shutil.move(local_path_tmp, local_path)
    return local_path


@register_model(
    "equflash",
    metadata={
        "model_path": {
            "type": "str",
            "choices": list(AVAILABLE_MODELS_DICT.keys()) + ["..."],
            "description": f"A string specifier ({AVAILABLE_MODELS_DICT.keys()}), which leads to model download from figshare or a path to a local checkpoint (.pt) file.",
            "default": "equflash_v2",
        },
    },
)
def _build(params, device):
    """Import and build the EquFlashV2 MLIP."""
    import numpy as np
    from GGNN.common.calculator import UCalculator

    # Fairchem resets the rng seeds when loading the model, so we need to keep it
    rng_seed_before = int(np.random.get_state()[1][0])  # type: ignore
    model_path = params.get("model_path", "equflash_v2")
    if model_path in AVAILABLE_MODELS_DICT:
        # Download the model from figshare
        model_url = AVAILABLE_MODELS_DICT[model_path]
        model_path = download_model(model_url)

    calc = UCalculator(checkpoint_path=model_path, seed=rng_seed_before, cpu=device.startswith("cpu"))

    return calc
