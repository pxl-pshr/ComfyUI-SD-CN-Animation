"""
Utility nodes for SD-CN-Animation ComfyUI pack.
"""

import torch
import numpy as np


class HistogramMatch:
    """
    Match the color histogram of source images to a reference image.
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
        try:
            from skimage.exposure import match_histograms
        except ImportError:
            raise ImportError(
                "scikit-image is required for histogram matching. "
                "Install it with: pip install scikit-image"
            )

        # Reference is the first frame (use frame 0)
        ref_np = (reference[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        results = []
        for i in range(source.shape[0]):
            src_np = (source[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            matched = match_histograms(src_np, ref_np, channel_axis=-1)
            matched = np.clip(matched, 0, 255).astype(np.float32) / 255.0
            results.append(torch.from_numpy(matched))

        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "HistogramMatch": HistogramMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HistogramMatch": "Histogram Match",
}
