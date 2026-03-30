"""
Shared sampling utilities for SD-CN-Animation ComfyUI nodes.
"""

import torch
import numpy as np
import logging
from PIL import Image
from io import BytesIO

import comfy.sample
import comfy.model_management as mm
import latent_preview

logger = logging.getLogger(__name__)


def histogram_match_tensor(source, reference):
    """
    Match histogram of source image to reference image.
    Both inputs are (1, H, W, 3) float32 tensors in [0, 1].
    Returns matched tensor in same format.
    """
    try:
        from skimage.exposure import match_histograms
    except ImportError:
        logger.warning("scikit-image not installed, skipping histogram matching")
        return source

    src_np = (source[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    ref_np = (reference[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    matched = match_histograms(src_np, ref_np, channel_axis=-1)
    matched = np.clip(matched, 0, 255).astype(np.float32) / 255.0
    return torch.from_numpy(matched).unsqueeze(0).to(source.device)


def apply_controlnet_to_cond(conditioning, control_net, hint_image, strength):
    """
    Apply ControlNet hint to conditioning, matching ComfyUI's ControlNetApply pattern.
    hint_image: (1, H, W, 3) float32 [0,1]
    """
    if strength == 0 or control_net is None:
        return conditioning

    control_hint = hint_image.movedim(-1, 1)  # (1, 3, H, W)
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        c_net = control_net.copy().set_cond_hint(control_hint, strength)
        if 'control' in t[1]:
            c_net.set_previous_controlnet(t[1]['control'])
        n[1]['control'] = c_net
        n[1]['control_apply_to_uncond'] = True
        c.append(n)
    return c


def do_sample(model, vae, positive, negative, image_tensor, seed, steps, cfg,
              sampler_name, scheduler, denoise, noise_mask=None, disable_pbar=True):
    """
    Perform img2img-style sampling: encode image -> add noise -> sample -> decode.

    image_tensor: (1, H, W, 3) float32 [0,1] - the init image
    noise_mask: (1, H, W) float32 [0,1] or None - inpaint mask (1=denoise, 0=keep)
    Returns: (1, H, W, 3) float32 [0,1] decoded image
    """
    # Encode to latent
    latent = vae.encode(image_tensor[:, :, :, :3])  # (1, 4, H//8, W//8)
    latent = comfy.sample.fix_empty_latent_channels(model, latent)

    # Prepare noise
    noise = comfy.sample.prepare_noise(latent, seed)

    # Prepare noise mask for inpainting (latent resolution)
    latent_mask = None
    if noise_mask is not None:
        # Reshape mask to (B, 1, H, W) — ComfyUI handles resizing internally
        mask_4d = noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1]))
        latent_mask = mask_4d

    # Sample
    callback = latent_preview.prepare_callback(model, steps)
    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, latent,
        denoise=denoise,
        noise_mask=latent_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed
    )

    # Decode
    decoded = vae.decode(samples)  # (1, H, W, 3)
    return decoded


def frame_to_preview(frame_tensor, max_size=512):
    """
    Convert a (1, H, W, 3) float32 [0,1] tensor to a ComfyUI preview tuple.
    Returns ("JPEG", PIL.Image, max_size) for use with ProgressBar.update_absolute().
    """
    img_np = (frame_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    return ("JPEG", pil_img, max_size)
