"""
SD-CN-Animation Txt2Vid node for ComfyUI.
Iteratively generates video frames using FloweR optical flow prediction
and two-pass SD sampling (inpaint + refine) with histogram matching.
"""

import math
import torch
import torch.nn.functional as F
import logging

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management as mm
import latent_preview

from ..flower_model import FloweR
from ..flow_utils import frames_norm, frames_renorm, occl_renorm
from ..sampling_utils import histogram_match_tensor, apply_controlnet_to_cond, do_sample, frame_to_preview, get_cond_for_frame

logger = logging.getLogger(__name__)


class SDCNTxt2Vid:
    """
    Generate a video sequence from text using FloweR optical flow prediction.

    Pipeline per frame:
    1. FloweR predicts next frame estimate + occlusion mask from last 4 frames
    2. Inpaint pass: img2img the prediction using occlusion as mask (processing_strength)
    3. Refine pass: img2img the result with low denoise (fix_frame_strength)
    4. Histogram match against first frame for color consistency
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "flower_model": ("FLOWER_MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 15, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "width": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 30, "min": 2, "max": 9999}),
                "processing_strength": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoising strength for inpaint pass (higher = more creative, less coherent)"
                }),
                "fix_frame_strength": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoising strength for refinement pass (keep low for stability)"
                }),
                "loop_frames": ("INT", {
                    "default": 0, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "Number of frames at end to blend back toward first frame for seamless loop. 0 = disabled."
                }),
            },
            "optional": {
                "init_image": ("IMAGE",),
                "control_net": ("CONTROL_NET",),
                "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "motion_ctrl": ("MOTION_CTRL",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "SD-CN-Animation"
    DESCRIPTION = (
        "Generates a video frame sequence using FloweR optical flow prediction "
        "with two-pass SD sampling for temporal coherence."
    )

    def generate(self, model, vae, positive, negative, flower_model, seed, steps, cfg,
                 sampler_name, scheduler, width, height, num_frames,
                 processing_strength, fix_frame_strength, loop_frames=0,
                 init_image=None, control_net=None, cn_strength=1.0, motion_ctrl=None):

        device = mm.get_torch_device()
        output_frames = []

        # Log conditioning info
        cond_batch = positive[0][0].shape[0] if positive and len(positive) > 0 else 0
        logger.info(f"Positive conditioning batch size: {cond_batch} "
                     f"({'scheduled' if cond_batch > 1 else 'single prompt'}), "
                     f"dict keys: {list(positive[0][1].keys()) if positive else []}")

        # Clamp loop_frames to valid range
        if loop_frames > 0:
            loop_frames = min(loop_frames, num_frames - 1)

        # --- Step 1: Generate or use first frame ---
        pos_cond = get_cond_for_frame(positive, 0)
        neg_cond = get_cond_for_frame(negative, 0)

        if init_image is not None:
            # Resize init image to target dimensions
            first_frame = init_image[0:1]  # (1, H, W, 3)
            if first_frame.shape[1] != height or first_frame.shape[2] != width:
                first_frame = first_frame.permute(0, 3, 1, 2)
                first_frame = F.interpolate(first_frame, size=(height, width), mode="bilinear", align_corners=False)
                first_frame = first_frame.permute(0, 2, 3, 1)

            # Run through img2img so model patches (IPAdapter etc.) can influence it
            logger.info("Generating first frame from init image via img2img...")
            first_frame = do_sample(
                model, vae, pos_cond, neg_cond,
                first_frame, seed, steps, cfg,
                sampler_name, scheduler,
                denoise=processing_strength,
                disable_pbar=False
            )
            first_frame = torch.clamp(first_frame, 0, 1)
        else:
            # txt2img: generate from empty latent with full denoise
            logger.info("Generating first frame via txt2img...")
            empty_latent = torch.zeros([1, 4, height // 8, width // 8],
                                       device="cpu")
            empty_latent = comfy.sample.fix_empty_latent_channels(model, empty_latent)
            noise = comfy.sample.prepare_noise(empty_latent, seed)

            callback = latent_preview.prepare_callback(model, steps)
            samples = comfy.sample.sample(
                model, noise, steps, cfg, sampler_name, scheduler,
                pos_cond, neg_cond, empty_latent,
                denoise=1.0,
                callback=callback,
                disable_pbar=False,
                seed=seed
            )
            first_frame = vae.decode(samples)  # (1, H, W, 3)
            first_frame = torch.clamp(first_frame, 0, 1)

        output_frames.append(first_frame)
        init_frame_ref = first_frame.clone()  # Reference for histogram matching

        # --- Step 2: Setup FloweR ---
        flower_h = height // 128 * 128
        flower_w = width // 128 * 128
        if flower_h == 0 or flower_w == 0:
            raise ValueError(f"Image dimensions must be at least 128px. Got {width}x{height}.")

        flower_net = FloweR(input_size=(flower_h, flower_w))
        flower_net.load_state_dict(flower_model["state_dict"])
        flower_net.to(device).eval()

        # 4-frame buffer for FloweR (stored at FloweR resolution, [0,1] range)
        clip_frames = torch.zeros(4, flower_h, flower_w, 3, device="cpu")

        prev_frame = first_frame  # (1, H, W, 3) on CPU
        pbar = comfy.utils.ProgressBar(num_frames - 1)

        # --- Step 3: Iterative frame generation ---
        for i in range(num_frames - 1):
            frame_seed = seed + i + 1  # Vary seed per frame (original uses -1 = random)

            # Update clip buffer with previous frame (resized to FloweR size)
            clip_frames = torch.roll(clip_frames, -1, dims=0)
            prev_resized = prev_frame.permute(0, 3, 1, 2)  # (1, 3, H, W)
            if prev_resized.shape[2] != flower_h or prev_resized.shape[3] != flower_w:
                prev_resized = F.interpolate(
                    prev_resized, size=(flower_h, flower_w),
                    mode="bilinear", align_corners=False
                )
            clip_frames[-1] = prev_resized[0].permute(1, 2, 0).cpu()  # (fH, fW, 3)

            # FloweR prediction
            # Normalize: FloweR expects [-1, 1] from pixel values [0, 255]
            clip_normed = frames_norm(clip_frames * 255.0)
            with torch.no_grad():
                pred_data = flower_net(clip_normed.unsqueeze(0).to(device))[0]  # (fH, fW, 6)

            # Extract predictions
            pred_occl = occl_renorm(pred_data[..., 2:3])  # [0, 255]

            # Occlusion multiplier (default 10, overridden by motion_ctrl)
            occl_mult = 10.0
            if motion_ctrl is not None:
                occl_mult = motion_ctrl["occlusion_multiplier"]
            pred_occl = torch.clamp(pred_occl * occl_mult, 0, 255)

            if motion_ctrl is not None and (
                motion_ctrl["flow_scale"] != 1.0 or
                motion_ctrl["pan_x"] != 0.0 or
                motion_ctrl["pan_y"] != 0.0 or
                motion_ctrl["zoom"] != 1.0 or
                motion_ctrl["rotate"] != 0.0
            ):
                # --- Motion control: extract raw flow, modify, re-warp ---
                # Raw flow from FloweR output (channels 0-1 are pred_flow/255)
                raw_flow = pred_data[..., 0:2] * 255.0  # (fH, fW, 2) pixel displacements
                raw_flow = raw_flow * motion_ctrl["flow_scale"]
                raw_flow[..., 0] += motion_ctrl["pan_x"]
                raw_flow[..., 1] += motion_ctrl["pan_y"]

                # Raw neural prediction (before FloweR's internal compositing)
                pred_raw = pred_data[..., 3:6]  # [-1, 1] normalized
                pred_raw = torch.clamp(pred_raw, -1, 1)

                # Re-warp previous frame with modified flow (replicating flower_model.py logic)
                fh, fw = flower_h, flower_w
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(0, fh), torch.arange(0, fw), indexing='ij'
                )
                flow_grid = torch.stack((grid_x, grid_y), dim=0).float().to(device)
                flow_grid = flow_grid.unsqueeze(0) + raw_flow.permute(2, 0, 1).unsqueeze(0)
                flow_grid[:, 0, :, :] = 2 * flow_grid[:, 0, :, :] / (fw - 1) - 1
                flow_grid[:, 1, :, :] = 2 * flow_grid[:, 1, :, :] / (fh - 1) - 1
                flow_grid = flow_grid.permute(0, 2, 3, 1)

                # Warp previous frame (from buffer, in [0,1])
                prev_buf = clip_frames[-1].to(device)  # (fH, fW, 3)
                prev_buf_bchw = prev_buf.permute(2, 0, 1).unsqueeze(0)  # (1, 3, fH, fW)
                # Convert to [-1,1] to match FloweR's range
                prev_buf_norm = prev_buf_bchw * 2.0 - 1.0
                warped = F.grid_sample(
                    prev_buf_norm, flow_grid, mode="nearest",
                    padding_mode="reflection", align_corners=False
                )

                # Composite: same formula as FloweR but with user's occlusion multiplier
                alpha = torch.clamp(occl_renorm(pred_data[..., 2:3]).permute(2, 0, 1).unsqueeze(0) / 255.0 * occl_mult, 0, 1) * 0.04
                warped = torch.clamp(warped, -1, 1)
                pred_raw_bchw = pred_raw.permute(2, 0, 1).unsqueeze(0)
                composite = pred_raw_bchw * alpha + warped * (1 - alpha)

                # Apply zoom and rotate as affine transform
                if motion_ctrl["zoom"] != 1.0 or motion_ctrl["rotate"] != 0.0:
                    z = motion_ctrl["zoom"]
                    angle_rad = math.radians(motion_ctrl["rotate"])
                    cos_a, sin_a = math.cos(angle_rad) / z, math.sin(angle_rad) / z
                    theta = torch.tensor([
                        [cos_a, -sin_a, 0.0],
                        [sin_a, cos_a, 0.0]
                    ], dtype=torch.float32, device=device).unsqueeze(0)
                    affine_grid = F.affine_grid(theta, composite.shape, align_corners=False)
                    composite = F.grid_sample(
                        composite, affine_grid, mode="bilinear",
                        padding_mode="reflection", align_corners=False
                    )

                # Convert back to [0,255] HWC for the existing resize/convert path
                pred_next = ((composite[0] + 1.0) * 127.5).permute(1, 2, 0)  # (fH, fW, 3)
                pred_next = torch.clamp(pred_next, 0, 255)
            else:
                # Standard path: use FloweR's composited prediction
                pred_next = frames_renorm(pred_data[..., 3:6])  # [0, 255]
                pred_next = torch.clamp(pred_next, 0, 255)

            # Resize to output resolution
            if flower_h != height or flower_w != width:
                pred_next = pred_next.unsqueeze(0).permute(0, 3, 1, 2)
                pred_next = F.interpolate(pred_next, size=(height, width), mode="bilinear", align_corners=False)
                pred_next = pred_next.permute(0, 2, 3, 1)[0]

                pred_occl = pred_occl.unsqueeze(0).permute(0, 3, 1, 2)
                pred_occl = F.interpolate(pred_occl, size=(height, width), mode="bilinear", align_corners=False)
                pred_occl = pred_occl.permute(0, 2, 3, 1)[0]

            # Convert to [0, 1] for ComfyUI
            pred_next_img = (pred_next / 255.0).unsqueeze(0).cpu()  # (1, H, W, 3)
            pred_next_img = torch.clamp(pred_next_img, 0, 1)
            pred_occl_mask = (pred_occl[..., 0] / 255.0).unsqueeze(0).cpu()  # (1, H, W)
            pred_occl_mask = torch.clamp(pred_occl_mask, 0, 1)

            # --- Loop blending: steer toward first frame in final frames ---
            if loop_frames > 0:
                frames_remaining = (num_frames - 1) - i  # how many frames left after this one
                if frames_remaining < loop_frames:
                    # blend_t goes from ~0 (start of loop zone) to 1.0 (last frame)
                    blend_t = 1.0 - (frames_remaining / loop_frames)
                    # Ease-in curve: stays low for most of the zone, ramps up at end
                    blend_t = blend_t ** 3
                    pred_next_img = pred_next_img * (1.0 - blend_t) + init_frame_ref * blend_t
                    pred_next_img = torch.clamp(pred_next_img, 0, 1)
                    # Also soften the occlusion mask — less inpainting as we converge
                    pred_occl_mask = pred_occl_mask * (1.0 - blend_t)

            # Get conditioning for this frame (supports prompt scheduling)
            frame_num = i + 1
            pos_cond = get_cond_for_frame(positive, frame_num)
            neg_cond = get_cond_for_frame(negative, frame_num)
            if control_net is not None:
                pos_cond = apply_controlnet_to_cond(pos_cond, control_net, pred_next_img, cn_strength)
                neg_cond = apply_controlnet_to_cond(neg_cond, control_net, pred_next_img, cn_strength)

            # --- Inpaint pass (processing_strength) ---
            inpainted = do_sample(
                model, vae, pos_cond, neg_cond,
                pred_next_img, frame_seed, steps, cfg,
                sampler_name, scheduler,
                denoise=processing_strength,
                noise_mask=pred_occl_mask,
                disable_pbar=True
            )
            inpainted = torch.clamp(inpainted, 0, 1)

            # Histogram match against first frame
            inpainted = histogram_match_tensor(inpainted, init_frame_ref)

            # --- Refine pass (fix_frame_strength) ---
            refined = do_sample(
                model, vae, pos_cond, neg_cond,
                inpainted, frame_seed + num_frames, steps, cfg,
                sampler_name, scheduler,
                denoise=fix_frame_strength,
                noise_mask=None,
                disable_pbar=True
            )
            refined = torch.clamp(refined, 0, 1)

            # Histogram match again
            refined = histogram_match_tensor(refined, init_frame_ref)

            output_frames.append(refined)
            prev_frame = refined

            preview = frame_to_preview(refined, frame_num=i + 2, total_frames=num_frames)
            pbar.update_absolute(i + 1, num_frames - 1, preview)
            logger.info(f"Frame {i + 2}/{num_frames} complete")

        # Cleanup FloweR
        flower_net.to("cpu")
        del flower_net
        mm.soft_empty_cache()

        # Stack all frames into batch: (N, H, W, 3)
        all_frames = torch.cat(output_frames, dim=0)
        return (all_frames,)


NODE_CLASS_MAPPINGS = {
    "SDCNTxt2Vid": SDCNTxt2Vid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDCNTxt2Vid": "SD-CN Animation Txt2Vid",
}
