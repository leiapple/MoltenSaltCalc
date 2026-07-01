"""Download models from provided urls to local cache."""

import shutil
import tempfile
import warnings
from pathlib import Path

import requests
from tqdm.auto import tqdm


def figshare_download(figshare_id: str) -> Path:
    """Downloads the model from figshare, inspired from fairchem to make sure cache is only written once the download has succeeded.

    Args:
        figshare_id (str): The figshare file id.

    Returns:
        Path: The path to the downloaded model.
    """

    cache_dir = Path.home() / ".cache" / "figshare"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / figshare_id
    if not local_path.exists():
        local_path_tmp = local_path.with_suffix(local_path.suffix + ".tmp")
        session = requests.Session()
        response = session.get(
            f"https://api.figshare.com/v2/file/download/{figshare_id}", stream=True, allow_redirects=True
        )
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


def ensure_model_extension(model_path: Path | str, extension: str = ".pt") -> Path:
    """Temporarily copy the model with its extension because the some calculators resolve symlinks and depend on the model checkpoint extension for backend selection.

    Args:
        model_path (Path | str): The path to the model.
        extension (str, optional): The extension to use. Defaults to ".pt".

    Returns:
        Path: The path to the model (tmpfile) with the extension.
    """
    model_path = Path(model_path)
    model_path_old = model_path.resolve()
    if "." not in model_path_old.name:
        model_path = Path(tempfile.mkdtemp()) / model_path.name
        # If still no extension is present, assume extension
        if "." not in model_path.name:
            warnings.warn(f"No extension found for model_path {model_path.name}. Assuming '{extension}'.", stacklevel=2)
            model_path = model_path.with_suffix(extension)
        shutil.copy(model_path_old, model_path)
    return model_path
