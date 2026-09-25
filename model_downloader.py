"""
Auto-download model weights from Hugging Face if not present locally.
"""

import os
import hashlib
import logging
import urllib.request
import shutil

logger = logging.getLogger(__name__)

HF_REPO = "pxlpshr/ComfyUI-SD-CN-Animation"

# Network timeout (seconds) for the urllib fallback. Downloads run at ComfyUI
# startup, so an unreachable host must not hang the server.
DOWNLOAD_TIMEOUT = 30

MODELS = {
    "FloweR_0.1.2.pth": {
        "hf_path": "models/FloweR/FloweR_0.1.2.pth",
        "url": f"https://huggingface.co/{HF_REPO}/resolve/main/models/FloweR/FloweR_0.1.2.pth",
        "size_mb": 8,
        "sha256": "5813bb213ec87971ec4cfc238af172de7a104ebb94c3d3630bbf7491cb2b0997",
    },
    "raft-things.pth": {
        "hf_path": "models/RAFT/raft-things.pth",
        "url": f"https://huggingface.co/{HF_REPO}/resolve/main/models/RAFT/raft-things.pth",
        "size_mb": 20,
        "sha256": "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1",
    },
}


def register_model_folder(folder_name, model_dir):
    """
    Register a model folder with ComfyUI, restricted to model file extensions so
    stray files (.DS_Store, partial downloads) don't show up in loader dropdowns.
    """
    import folder_paths

    os.makedirs(model_dir, exist_ok=True)
    folder_paths.add_model_folder_path(folder_name, model_dir)
    paths, _ = folder_paths.folder_names_and_paths[folder_name]
    folder_paths.folder_names_and_paths[folder_name] = (paths, folder_paths.supported_pt_extensions)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(model_dir, filename):
    """
    Check if model exists in model_dir. If not, download from Hugging Face.
    Returns the full path to the model file.
    """
    model_path = os.path.join(model_dir, filename)

    if os.path.isfile(model_path):
        return model_path

    if filename not in MODELS:
        return None

    info = MODELS[filename]
    url = info["url"]
    size_mb = info["size_mb"]

    logger.info(f"Downloading {filename} (~{size_mb}MB) from Hugging Face...")
    os.makedirs(model_dir, exist_ok=True)

    # Always write to a temp file and move it into place only after the hash
    # checks out, so an interrupted download never leaves a truncated model
    # that later looks valid.
    tmp_path = model_path + ".download"
    try:
        downloaded_via = None

        # Try huggingface_hub first (supports resume, progress)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            hf_hub_download = None

        if hf_hub_download is not None:
            cache_dir = os.path.join(model_dir, ".hf_cache")
            try:
                downloaded = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=info.get("hf_path", filename),
                    cache_dir=cache_dir,
                    local_dir=None,
                )
                shutil.copyfile(downloaded, tmp_path)
                downloaded_via = "huggingface_hub"
            finally:
                if os.path.isdir(cache_dir):
                    shutil.rmtree(cache_dir, ignore_errors=True)

        if downloaded_via is None:
            logger.info(f"Falling back to urllib download for {filename}...")
            req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-SD-CN-Animation"})
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as response, open(tmp_path, 'wb') as out_file:
                total = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        if downloaded % (10 * 1024 * 1024) < chunk_size:  # Log every ~10MB
                            logger.info(f"  {filename}: {pct:.0f}%")
            downloaded_via = "urllib"

        expected = info.get("sha256")
        if expected:
            actual = _sha256(tmp_path)
            if actual != expected:
                raise RuntimeError(f"checksum mismatch (expected {expected}, got {actual})")

        os.replace(tmp_path, model_path)
        logger.info(f"Downloaded {filename} via {downloaded_via}")
        return model_path

    except Exception as e:
        # Cleanup partial download
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        logger.error(f"Failed to download {filename}: {e}")
        raise RuntimeError(
            f"Could not download {filename}. Please download manually from:\n"
            f"  {url}\n"
            f"and place it in:\n"
            f"  {model_dir}"
        ) from e
