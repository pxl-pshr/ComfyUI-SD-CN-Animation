"""
SD-CN-Animation Vid2Vid node for ComfyUI.
Stylizes input video frames using RAFT optical flow for temporal coherence
and two-pass SD sampling (process + refine) with histogram matching.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import logging

import comfy.samplers
import comfy.utils
import comfy.model_management as mm
import folder_paths

from .flow_utils import raft_estimate_flow, raft_load_model, raft_clear_memory, compute_diff_map
from .sampling_utils import histogram_match_tensor, apply_controlnet_to_cond, do_sample, frame_to_preview
from .model_downloader import ensure_model

logger = logging.getLogger(__name__)


# Register RAFT model folder
raft_model_dir = os.path.join(folder_paths.models_dir, "RAFT")
os.makedirs(raft_model_dir, exist_ok=True)
folder_paths.add_model_folder_path("raft", raft_model_dir)

# Auto-download RAFT model if not present
try:
    ensure_model(raft_model_dir, "raft-things.pth")
except Exception as e:
    print(f"[SD-CN-Animation] RAFT model auto-download failed: {e}")
    print("[SD-CN-Animation] Please download raft-things.pth manually from https://huggingface.co/pxlpshr/ComfyUI-SD-CN-Animation")


class LoadRAFTModel:
    """Load the RAFT optical flow estimation model."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("raft"),),
            }
        }

    RETURN_TYPES = ("RAFT_MODEL",)
    RETURN_NAMES = ("raft_model",)
    FUNCTION = "load_model"
    CATEGORY = "SD-CN-Animation"

    def load_model(self, model_name):
        model_path = folder_paths.get_full_path("raft", model_name)
        return ({"model_path": model_path},)


class SDCNVid2Vid:
    """
    Stylize video frames using RAFT optical flow for temporal coherence.

    Pipeline per frame:
    1. RAFT estimates bidirectional optical flow between consecutive frames
    2. Compute occlusion mask and warp previous styled frame
    3. Blend warped styled frame with current input frame
    4. Inpaint pass: img2img with processing_strength
    5. Histogram match + blend into occlusion areas
    6. Refine pass: img2img with fix_frame_strength
    7. Final histogram match
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "raft_model": ("RAFT_MODEL",),
                "frames": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 15, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "processing_strength": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoising strength for main stylization pass"
                }),
                "fix_frame_strength": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoising strength for refinement pass (0 to skip)"
                }),
                "blend_alpha": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blend between warped styled frame (0) and current input frame (1)"
                }),
                "occlusion_mask_blur": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Gaussian blur applied to occlusion mask"
                }),
                "occlusion_mask_flow_multiplier": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Weight for flow-based occlusion detection"
                }),
                "occlusion_mask_difo_multiplier": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Weight for original frame difference occlusion"
                }),
                "occlusion_mask_difs_multiplier": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Weight for styled frame difference occlusion"
                }),
                "occlusion_mask_trailing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Blend current occlusion with previous to reduce ghosting"
                }),
            },
            "optional": {
                "control_net": ("CONTROL_NET",),
                "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "SD-CN-Animation"
    DESCRIPTION = (
        "Stylizes input video frames using RAFT optical flow for temporal coherence "
        "with two-pass SD sampling."
    )

    def generate(self, model, vae, positive, negative, raft_model, frames, seed, steps, cfg,
                 sampler_name, scheduler, processing_strength, fix_frame_strength,
                 blend_alpha, occlusion_mask_blur, occlusion_mask_flow_multiplier,
                 occlusion_mask_difo_multiplier, occlusion_mask_difs_multiplier,
                 occlusion_mask_trailing,
                 control_net=None, cn_strength=1.0):

        device = mm.get_torch_device()
        B, H, W, C = frames.shape
        output_frames = []

        if B < 2:
            raise ValueError("Vid2Vid requires at least 2 input frames.")

        # Occlusion args dict (matches original code's expected keys)
        occ_args = {
            'occlusion_mask_blur': occlusion_mask_blur,
            'occlusion_mask_flow_multiplier': occlusion_mask_flow_multiplier,
            'occlusion_mask_difo_multiplier': occlusion_mask_difo_multiplier,
            'occlusion_mask_difs_multiplier': occlusion_mask_difs_multiplier,
        }

        # --- Load RAFT model ---
        model_path = raft_model["model_path"]
        raft_load_model(model_path, device=device)
        logger.info("RAFT model loaded")

        # --- Process first frame ---
        curr_frame_np = (frames[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        curr_frame_tensor = frames[0:1]  # (1, H, W, 3)

        # Apply ControlNet if provided
        pos_cond = positive
        neg_cond = negative
        if control_net is not None:
            pos_cond = apply_controlnet_to_cond(positive, control_net, curr_frame_tensor, cn_strength)
            neg_cond = apply_controlnet_to_cond(negative, control_net, curr_frame_tensor, cn_strength)

        # img2img first frame with full processing_strength
        styled_first = do_sample(
            model, vae, pos_cond, neg_cond,
            curr_frame_tensor, seed, steps, cfg,
            sampler_name, scheduler,
            denoise=processing_strength,
            disable_pbar=False
        )
        styled_first = torch.clamp(styled_first, 0, 1)

        # Histogram match against input frame
        styled_first = histogram_match_tensor(styled_first, curr_frame_tensor)

        output_frames.append(styled_first)
        prev_frame_np = curr_frame_np.copy()
        prev_styled_np = (styled_first[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        prev_alpha_mask = None

        pbar = comfy.utils.ProgressBar(B - 1)

        # --- Process remaining frames ---
        for i in range(1, B):
            frame_seed = seed + i

            curr_frame_np = (frames[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            curr_frame_tensor = frames[i:i+1]  # (1, H, W, 3)

            # RAFT optical flow between consecutive frames
            next_flow, prev_flow, _ = raft_estimate_flow(
                prev_frame_np, curr_frame_np, device=device, model_path=model_path
            )

            # Compute occlusion mask and warp previous styled frame
            alpha_mask, warped_styled = compute_diff_map(
                next_flow, prev_flow, prev_frame_np, curr_frame_np,
                prev_styled_np, occ_args
            )

            # Trailing occlusion mask
            if occlusion_mask_trailing and prev_alpha_mask is not None:
                alpha_mask = alpha_mask + prev_alpha_mask * 0.5
            prev_alpha_mask = alpha_mask.copy()

            alpha_mask = np.clip(alpha_mask, 0, 1)
            occlusion_mask_uint8 = np.clip(alpha_mask * 255, 0, 255).astype(np.uint8)

            # Blend warped styled frame with current input
            warped_styled = curr_frame_np.astype(float) * alpha_mask + warped_styled.astype(float) * (1 - alpha_mask)

            # Blend between warped styled and current input based on blend_alpha
            init_img = warped_styled * (1 - blend_alpha) + curr_frame_np.astype(float) * blend_alpha
            init_img = np.clip(init_img, 0, 255).astype(np.uint8)
            init_tensor = torch.from_numpy(init_img).float().unsqueeze(0) / 255.0  # (1, H, W, 3)

            # Occlusion mask as tensor for inpainting
            occl_tensor = torch.from_numpy(occlusion_mask_uint8[..., 0]).float().unsqueeze(0) / 255.0  # (1, H, W)

            # Apply ControlNet with current input frame as hint
            pos_cond = positive
            neg_cond = negative
            if control_net is not None:
                pos_cond = apply_controlnet_to_cond(positive, control_net, curr_frame_tensor, cn_strength)
                neg_cond = apply_controlnet_to_cond(negative, control_net, curr_frame_tensor, cn_strength)

            # --- Step 1: Process/inpaint ---
            processed = do_sample(
                model, vae, pos_cond, neg_cond,
                init_tensor, frame_seed, steps, cfg,
                sampler_name, scheduler,
                denoise=processing_strength,
                noise_mask=occl_tensor,
                disable_pbar=True
            )
            processed = torch.clamp(processed, 0, 1)

            # Histogram match against current input frame
            processed = histogram_match_tensor(processed, curr_frame_tensor)

            # Blend processed result into occlusion areas
            proc_np = (processed[0].cpu().numpy() * 255).clip(0, 255).astype(np.float32)
            proc_np = proc_np * alpha_mask + warped_styled * (1 - alpha_mask)
            proc_np = np.clip(proc_np, 0, 255).astype(np.uint8)
            prev_styled_np = proc_np.copy()

            # --- Step 2: Refine ---
            if fix_frame_strength > 0:
                refine_tensor = torch.from_numpy(proc_np).float().unsqueeze(0) / 255.0

                refined = do_sample(
                    model, vae, pos_cond, neg_cond,
                    refine_tensor, frame_seed + B, steps, cfg,
                    sampler_name, scheduler,
                    denoise=fix_frame_strength,
                    noise_mask=None,
                    disable_pbar=True
                )
                refined = torch.clamp(refined, 0, 1)
                refined = histogram_match_tensor(refined, curr_frame_tensor)
                output_frames.append(refined)
                prev_styled_np = (refined[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            else:
                output_frames.append(torch.from_numpy(proc_np).float().unsqueeze(0) / 255.0)

            prev_frame_np = curr_frame_np.copy()

            preview = frame_to_preview(output_frames[-1])
            pbar.update_absolute(i, B - 1, preview)
            logger.info(f"Frame {i + 1}/{B} complete")

        # Cleanup RAFT
        raft_clear_memory()
        mm.soft_empty_cache()

        all_frames = torch.cat(output_frames, dim=0)
        return (all_frames,)


NODE_CLASS_MAPPINGS = {
    "LoadRAFTModel": LoadRAFTModel,
    "SDCNVid2Vid": SDCNVid2Vid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadRAFTModel": "Load RAFT Model",
    "SDCNVid2Vid": "SD-CN Animation Vid2Vid",
}
