"""
Flow normalization and optical flow utilities.
Ported from SD-CN-Animation/scripts/core/flow_utils.py
"""

import numpy as np
import cv2
import torch
import argparse
import gc

from .raft.raft import RAFT
from .raft.utils.utils import InputPadder


# FloweR normalization helpers

def frames_norm(frame):
    """Normalize frame pixels from [0, 255] to [-1, 1]"""
    return frame / 127.5 - 1


def frames_renorm(frame):
    """Denormalize frame pixels from [-1, 1] to [0, 255]"""
    return (frame + 1) * 127.5


def flow_renorm(flow):
    """Denormalize optical flow from normalized to [-255, 255]"""
    return flow * 255


def occl_renorm(occl):
    """Denormalize occlusion mask from normalized to [0, 255]"""
    return (occl + 1) * 127.5


# RAFT optical flow estimation

_raft_model = None


def raft_clear_memory():
    global _raft_model
    if _raft_model is not None:
        del _raft_model
        gc.collect()
        torch.cuda.empty_cache()
        _raft_model = None


def raft_load_model(model_path, device='cuda'):
    """Load RAFT model from weights file."""
    global _raft_model

    if _raft_model is not None:
        return _raft_model

    use_mixed_precision = torch.cuda.is_available()
    args = argparse.Namespace(**{
        'model': model_path,
        'mixed_precision': use_mixed_precision,
        'small': False,
        'alternate_corr': False,
        'path': ""
    })

    try:
        model = RAFT(args)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)

        # Handle DataParallel state dicts (keys prefixed with "module.")
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        _raft_model = model
    except Exception as e:
        _raft_model = None
        raise RuntimeError(f"Failed to load RAFT model from {model_path}: {e}")

    return _raft_model


def raft_estimate_flow(frame1, frame2, device='cuda', model_path=None):
    """
    Estimate bidirectional optical flow between two frames using RAFT.

    Args:
        frame1, frame2: numpy arrays (H, W, 3) uint8 RGB
        device: torch device
        model_path: path to raft-things.pth

    Returns:
        next_flow, prev_flow: (H, W, 2) numpy arrays
        occlusion_mask: (H, W, 3) numpy array
    """
    global _raft_model

    org_size = frame1.shape[1], frame1.shape[0]
    size = frame1.shape[1] // 16 * 16, frame1.shape[0] // 16 * 16
    frame1 = cv2.resize(frame1, size)
    frame2 = cv2.resize(frame2, size)

    if _raft_model is None and model_path is not None:
        raft_load_model(model_path, device)

    if _raft_model is None:
        raise RuntimeError("RAFT model not loaded. Provide model_path or call raft_load_model first.")

    with torch.no_grad():
        frame1_torch = torch.from_numpy(frame1).permute(2, 0, 1).float()[None].to(device)
        frame2_torch = torch.from_numpy(frame2).permute(2, 0, 1).float()[None].to(device)

        padder = InputPadder(frame1_torch.shape)
        image1, image2 = padder.pad(frame1_torch, frame2_torch)

        _, next_flow = _raft_model(image1, image2, iters=20, test_mode=True)
        _, prev_flow = _raft_model(image2, image1, iters=20, test_mode=True)

        next_flow = next_flow[0].permute(1, 2, 0).cpu().numpy()
        prev_flow = prev_flow[0].permute(1, 2, 0).cpu().numpy()

        fb_flow = next_flow + prev_flow
        fb_norm = np.linalg.norm(fb_flow, axis=2)
        occlusion_mask = fb_norm[..., None].repeat(3, axis=-1)

    next_flow = cv2.resize(next_flow, org_size)
    prev_flow = cv2.resize(prev_flow, org_size)

    return next_flow, prev_flow, occlusion_mask


def compute_diff_map(next_flow, prev_flow, prev_frame, cur_frame, prev_frame_styled, args_dict):
    """
    Compute occlusion/difference mask and warp the previous styled frame.

    Args:
        next_flow, prev_flow: (H, W, 2) optical flow arrays
        prev_frame: previous input frame (H, W, 3) uint8
        cur_frame: current input frame (H, W, 3) uint8
        prev_frame_styled: previous stylized frame (H, W, 3) uint8
        args_dict: dict with occlusion_mask_blur, occlusion_mask_flow_multiplier,
                   occlusion_mask_difo_multiplier, occlusion_mask_difs_multiplier

    Returns:
        alpha_mask: (H, W, 3) float [0, 1]
        warped_frame_styled: (H, W, 3) float
    """
    h, w = cur_frame.shape[:2]
    fl_h, fl_w = next_flow.shape[:2]

    # normalize flow: divide [dx, dy] by [width, height]
    next_flow = next_flow / np.array([fl_w, fl_h])
    prev_flow = prev_flow / np.array([fl_w, fl_h])

    # compute occlusion mask from forward-backward consistency
    fb_flow = next_flow + prev_flow
    fb_norm = np.linalg.norm(fb_flow, axis=2)

    zero_flow_mask = np.clip(1 - np.linalg.norm(prev_flow, axis=-1)[..., None] * 20, 0, 1)
    diff_mask_flow = fb_norm[..., None] * zero_flow_mask

    # resize flow and denormalize: multiply [dx, dy] by [width, height]
    next_flow = cv2.resize(next_flow, (w, h))
    next_flow = (next_flow * np.array([w, h])).astype(np.float32)
    prev_flow = cv2.resize(prev_flow, (w, h))
    prev_flow = (prev_flow * np.array([w, h])).astype(np.float32)

    # Generate sampling grids
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
    flow_grid = torch.stack((grid_x, grid_y), dim=0).float()
    flow_grid += torch.from_numpy(prev_flow).permute(2, 0, 1)
    flow_grid = flow_grid.unsqueeze(0)
    flow_grid[:, 0, :, :] = 2 * flow_grid[:, 0, :, :] / (w - 1) - 1
    flow_grid[:, 1, :, :] = 2 * flow_grid[:, 1, :, :] / (h - 1) - 1
    flow_grid = flow_grid.permute(0, 2, 3, 1)

    prev_frame_torch = torch.from_numpy(prev_frame).float().unsqueeze(0).permute(0, 3, 1, 2)
    prev_frame_styled_torch = torch.from_numpy(prev_frame_styled).float().unsqueeze(0).permute(0, 3, 1, 2)

    warped_frame = torch.nn.functional.grid_sample(
        prev_frame_torch, flow_grid, mode="nearest",
        padding_mode="reflection", align_corners=True
    ).permute(0, 2, 3, 1)[0].numpy()

    warped_frame_styled = torch.nn.functional.grid_sample(
        prev_frame_styled_torch, flow_grid, mode="nearest",
        padding_mode="reflection", align_corners=True
    ).permute(0, 2, 3, 1)[0].numpy()

    diff_mask_org = np.abs(warped_frame.astype(np.float32) - cur_frame.astype(np.float32)) / 255
    diff_mask_org = diff_mask_org.max(axis=-1, keepdims=True)

    diff_mask_stl = np.abs(warped_frame_styled.astype(np.float32) - cur_frame.astype(np.float32)) / 255
    diff_mask_stl = diff_mask_stl.max(axis=-1, keepdims=True)

    alpha_mask = np.maximum.reduce([
        diff_mask_flow * args_dict['occlusion_mask_flow_multiplier'] * 10,
        diff_mask_org * args_dict['occlusion_mask_difo_multiplier'],
        diff_mask_stl * args_dict['occlusion_mask_difs_multiplier']
    ])
    alpha_mask = alpha_mask.repeat(3, axis=-1)

    if args_dict['occlusion_mask_blur'] > 0:
        blur_filter_size = min(w, h) // 15 | 1
        alpha_mask = cv2.GaussianBlur(
            alpha_mask, (blur_filter_size, blur_filter_size),
            args_dict['occlusion_mask_blur'], cv2.BORDER_REFLECT
        )

    alpha_mask = np.clip(alpha_mask, 0, 1)

    return alpha_mask, warped_frame_styled
