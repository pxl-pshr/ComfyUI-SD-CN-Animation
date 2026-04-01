"""
Auto-download model weights from Hugging Face if not present locally.
"""

import os
import logging
import urllib.request
import shutil

logger = logging.getLogger(__name__)

HF_REPO = "pxlpshr/ComfyUI-SD-CN-Animation"

MODELS = {
    "FloweR_0.1.2.pth": {
        "hf_path": "models/FloweR/FloweR_0.1.2.pth",
        "url": f"https://huggingface.co/{HF_REPO}/resolve/main/models/FloweR/FloweR_0.1.2.pth",
        "size_mb": 8,
    },
    "raft-things.pth": {
        "hf_path": "models/RAFT/raft-things.pth",
        "url": f"https://huggingface.co/{HF_REPO}/resolve/main/models/RAFT/raft-things.pth",
        "size_mb": 20,
    },
}


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

    tmp_path = model_path + ".download"
    try:
        # Try huggingface_hub first (supports resume, progress)
        try:
            from huggingface_hub import hf_hub_download
            hf_path = info.get("hf_path", filename)
            downloaded = hf_hub_download(
                repo_id=HF_REPO,
                filename=hf_path,
                cache_dir=os.path.join(model_dir, ".hf_cache"),
                local_dir=None,
            )
            # Copy from cache to the expected flat path
            shutil.copy2(downloaded, model_path)
            # Clean up the cache directory
            cache_dir = os.path.join(model_dir, ".hf_cache")
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)
            logger.info(f"Downloaded {filename} via huggingface_hub")
            return model_path
        except ImportError:
            pass

        # Fallback to urllib
        logger.info(f"Falling back to urllib download for {filename}...")
        req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-SD-CN-Animation"})
        with urllib.request.urlopen(req) as response, open(tmp_path, 'wb') as out_file:
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

        os.rename(tmp_path, model_path)
        logger.info(f"Downloaded {filename} via urllib")
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
        )
