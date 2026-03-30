"""
FloweR model loader and prediction nodes for ComfyUI.
"""

import os
import cv2
import torch
import numpy as np
import folder_paths
import comfy.model_management as mm
import comfy.utils

from .flower_model import FloweR
from .flow_utils import frames_norm, frames_renorm, occl_renorm
from .model_downloader import ensure_model

# Register FloweR model folder
flower_model_dir = os.path.join(folder_paths.models_dir, "FloweR")
os.makedirs(flower_model_dir, exist_ok=True)
folder_paths.add_model_folder_path("flower", flower_model_dir)

# Auto-download FloweR model if not present
try:
    ensure_model(flower_model_dir, "FloweR_0.1.2.pth")
except Exception as e:
    print(f"[SD-CN-Animation] FloweR model auto-download failed: {e}")
    print("[SD-CN-Animation] Please download FloweR_0.1.2.pth manually from https://huggingface.co/pxlpshr/ComfyUI-SD-CN-Animation")


class LoadFloweRModel:
    """Load the FloweR optical flow reconstruction model."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("flower"),),
            }
        }

    RETURN_TYPES = ("FLOWER_MODEL",)
    RETURN_NAMES = ("flower_model",)
    FUNCTION = "load_model"
    CATEGORY = "SD-CN-Animation"

    def load_model(self, model_name):
        model_path = folder_paths.get_full_path("flower", model_name)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        return ({"state_dict": state_dict},)


class FloweRPredict:
    """
    Run FloweR prediction on a batch of frames.
    Takes 4+ frames and predicts the next frame, occlusion mask, and optical flow.
    Useful as a standalone utility for debugging and visualization.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "flower_model": ("FLOWER_MODEL",),
                "frames": ("IMAGE",),  # (B, H, W, 3) - needs B >= 4
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("predicted_frame", "occlusion_mask", "flow_visualization")
    FUNCTION = "predict"
    CATEGORY = "SD-CN-Animation"

    def predict(self, flower_model, frames):
        device = mm.get_torch_device()
        B, H, W, C = frames.shape

        if B < 4:
            raise ValueError(
                f"FloweR requires at least 4 input frames, got {B}. "
                f"Pass a batch of 4+ frames."
            )

        # FloweR needs dimensions as multiples of 128
        flower_h = H // 128 * 128
        flower_w = W // 128 * 128

        if flower_h == 0 or flower_w == 0:
            raise ValueError(
                f"Image dimensions must be at least 128px. Got {W}x{H}."
            )

        # Construct model with correct input_size and load weights
        model = FloweR(input_size=(flower_h, flower_w))
        model.load_state_dict(flower_model["state_dict"])
        model.to(device).eval()

        # Take last 4 frames, resize to FloweR size
        last_4 = frames[-4:]  # (4, H, W, 3)

        # Resize if needed
        if H != flower_h or W != flower_w:
            last_4_bchw = last_4.permute(0, 3, 1, 2)  # (4, 3, H, W)
            last_4_bchw = torch.nn.functional.interpolate(
                last_4_bchw, size=(flower_h, flower_w), mode="bilinear", align_corners=False
            )
            last_4_resized = last_4_bchw.permute(0, 2, 3, 1)  # (4, fH, fW, 3)
        else:
            last_4_resized = last_4

        # Normalize: FloweR expects [-1, 1] from pixel values [0, 255]
        # ComfyUI images are [0, 1] float, so multiply by 255 first
        clip_normed = frames_norm(last_4_resized * 255.0)

        with torch.no_grad():
            pred_data = model(clip_normed.unsqueeze(0).to(device))[0]  # (fH, fW, 6)

        # Extract predictions
        pred_occl = occl_renorm(pred_data[..., 2:3])  # [0, 255]
        pred_next = frames_renorm(pred_data[..., 3:6])  # [0, 255]

        pred_occl = torch.clamp(pred_occl * 10, 0, 255)
        pred_next = torch.clamp(pred_next, 0, 255)

        # Resize back to original resolution if needed
        if H != flower_h or W != flower_w:
            pred_next = pred_next.unsqueeze(0).permute(0, 3, 1, 2)
            pred_next = torch.nn.functional.interpolate(
                pred_next, size=(H, W), mode="bilinear", align_corners=False
            )
            pred_next = pred_next.permute(0, 2, 3, 1)[0]

            pred_occl = pred_occl.unsqueeze(0).permute(0, 3, 1, 2)
            pred_occl = torch.nn.functional.interpolate(
                pred_occl, size=(H, W), mode="bilinear", align_corners=False
            )
            pred_occl = pred_occl.permute(0, 2, 3, 1)[0]

        # Convert to ComfyUI format [0, 1]
        predicted_frame = (pred_next / 255.0).unsqueeze(0).cpu()  # (1, H, W, 3)
        occlusion_mask = (pred_occl[..., 0] / 255.0).unsqueeze(0).cpu()  # (1, H, W)

        # Flow visualization (HSV encoding)
        pred_flow = pred_data[..., :2].cpu().numpy()
        flow_vis = self._flow_to_hsv(pred_flow, flower_h, flower_w)
        if H != flower_h or W != flower_w:
            flow_vis_t = torch.from_numpy(flow_vis).unsqueeze(0).permute(0, 3, 1, 2)
            flow_vis_t = torch.nn.functional.interpolate(
                flow_vis_t, size=(H, W), mode="bilinear", align_corners=False
            )
            flow_vis = flow_vis_t.permute(0, 2, 3, 1)[0].numpy()
        flow_vis_tensor = torch.from_numpy(flow_vis).unsqueeze(0).float()  # (1, H, W, 3)

        # Cleanup
        model.to("cpu")
        del model
        mm.soft_empty_cache()

        return (predicted_frame, occlusion_mask, flow_vis_tensor)

    def _flow_to_hsv(self, flow, h, w):
        """Convert optical flow to HSV color visualization."""
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        angle = np.arctan2(flow[..., 1], flow[..., 0])

        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[..., 1] = 255
        max_mag = magnitude.max() if magnitude.max() > 0 else 1
        hsv[..., 2] = np.clip(magnitude / max_mag * 255, 0, 255).astype(np.uint8)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb.astype(np.float32) / 255.0


NODE_CLASS_MAPPINGS = {
    "LoadFloweRModel": LoadFloweRModel,
    "FloweRPredict": FloweRPredict,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFloweRModel": "Load FloweR Model",
    "FloweRPredict": "FloweR Predict",
}
