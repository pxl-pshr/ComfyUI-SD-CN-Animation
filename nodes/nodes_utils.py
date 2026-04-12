"""
Utility nodes for SD-CN-Animation ComfyUI pack.
"""

import torch
import numpy as np
import cv2


class HistogramMatch:
    """
    Match the color histogram of source images to a reference image in LAB space.
    LAB matching preserves hue relationships unlike RGB matching.
    Useful for maintaining color consistency across video frames.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": ("IMAGE",),
                "reference": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("matched",)
    FUNCTION = "match"
    CATEGORY = "SD-CN-Animation/utils"

    def match(self, source, reference):
        from skimage.exposure import match_histograms

        # Reference is the first frame (use frame 0)
        ref_np = (reference[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB)

        results = []
        for i in range(source.shape[0]):
            src_np = (source[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            src_lab = cv2.cvtColor(src_np, cv2.COLOR_RGB2LAB)
            matched_lab = match_histograms(src_lab, ref_lab, channel_axis=-1)
            matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)
            matched = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
            matched = matched.astype(np.float32) / 255.0
            results.append(torch.from_numpy(matched))

        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "HistogramMatch": HistogramMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HistogramMatch": "Histogram Match",
}
