"""Implementation of the DeePMD (DPA) MLIP."""

from moltensaltcalc.download_model import ensure_model_extension, figshare_download
from moltensaltcalc.registry import register_model

MODEL_HF_ID = "deepmodelingcommunity"
AVAILABLE_MODELS_HF = {
    "DPA-3.3-1M": "DPA-3.3-1M.pt",
    "DPA-2.4-7M": "DPA-2.4-7M-patched-mt.pt",
    "DPA-3.1-3M-FT": "dpa-3.1-3m-ft.pth",
    "DPA3-Omol-Large": "DPA3-Omol-Large.pt",
    "DPA-3.2-5M": "DPA-3.2-5M.pt",
    "DPA-3.1-3M": "DPA-3.1-3M.pt",
}

AVAILABLE_MODELS_FIGSHARE = {"DPA-4.0.1-Pro-MPtrj": "65469204"}


@register_model(
    "deepmd",
    metadata={
        "model_path": {
            "type": "str",
            "choices": list(AVAILABLE_MODELS_HF.keys()) + list(AVAILABLE_MODELS_FIGSHARE.keys()),
            "description": f"A string specifier ({AVAILABLE_MODELS_HF.keys()}), which leads to model download from huggingface/figshare or a path to a local checkpoint (.pt) file.",
            "default": "DPA-3.3-1M",
        },
        "model_task": {
            "type": "str",
            "choices": [
                "Domains_Alloy",
                "Domains_Anode",
                "Domains_Cluster",
                "Domains_FerroEle",
                "Domains_SSE_PBE",
                "Domains_SemiCond",
                "H2O_H2O_PD",
                "Metals_AlMgCu",
                "Metals_AgAu_PBED3",
                "Others_In2Se3",
                "MPGen_OpenCSP",
                "Alloy_tongqi",
                "SSE_ABACUS",
                "Hybrid_Perovskite",
                "Electrolyte",
                "ODAC23",
                "Alex2D",
                "Omat24",
                "OC20M",
                "OC22",
                "Organic_Reactions",
                "OMol25",
                "MPTrj",
            ],
            "description": "The task to run the model for.",
            "default": "Omat24",
        },
        "model_revision": {
            "type": "str",
            "description": "Revision of the model to download from HuggingFace.",
            "default": "main",
        },
    },
)
def _build(params, device):
    """Import and build the DeePMD (DPA) MLIP."""
    from deepmd.pt.utils.ase_calc import DPCalculator
    from huggingface_hub import hf_hub_download

    model_path = params.get("model_path", "DPA-3.3-1M")
    model_task = params.get("model_task", "Omat24")
    rev = params.get("model_revision", "main")

    if model_path in AVAILABLE_MODELS_HF or model_path in AVAILABLE_MODELS_FIGSHARE:
        if model_path in AVAILABLE_MODELS_HF:
            model_path = hf_hub_download(
                repo_id=f"{MODEL_HF_ID}/{model_path}", filename=AVAILABLE_MODELS_HF[model_path], revision=rev
            )
        else:
            model_path = figshare_download(AVAILABLE_MODELS_FIGSHARE[model_path])

    model_path = ensure_model_extension(model_path)

    calc = DPCalculator(model_path, head=model_task, device=device)

    return calc
