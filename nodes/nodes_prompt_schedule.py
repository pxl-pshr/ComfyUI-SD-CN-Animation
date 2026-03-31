"""
Prompt schedule node for SD-CN-Animation.
Parses a Deforum-style keyframe schedule and outputs batched CONDITIONING
compatible with the Txt2Vid/Vid2Vid nodes' prompt scheduling support.
"""

import json
import re
import torch
import logging

logger = logging.getLogger(__name__)


def parse_schedule(text):
    """
    Parse a prompt schedule string into a sorted list of (frame, prompt) tuples.

    Accepts Deforum-style format:
        "0": "a cat on a beach",
        "30": "a dog on a mountain"

    Also accepts simplified format:
        0: a cat on a beach
        30: a dog on a mountain
    """
    keyframes = []

    # Try JSON parse first (wrapped in braces if needed)
    stripped = text.strip()
    if not stripped.startswith("{"):
        stripped = "{" + stripped + "}"
    # Ensure trailing commas don't break JSON
    stripped = re.sub(r',\s*}', '}', stripped)
    try:
        data = json.loads(stripped)
        for k, v in data.items():
            keyframes.append((int(k), str(v)))
        keyframes.sort(key=lambda x: x[0])
        return keyframes
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: line-by-line parsing
    for line in text.strip().splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        match = re.match(r'"?(\d+)"?\s*:\s*"?(.*?)"?\s*$', line)
        if match:
            keyframes.append((int(match.group(1)), match.group(2)))

    keyframes.sort(key=lambda x: x[0])
    return keyframes


def encode_prompt(clip, prompt_text):
    """Encode a text prompt using CLIP, returning (cond_tensor, pooled_or_none)."""
    tokens = clip.tokenize(prompt_text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = output.pop("cond")
    pooled = output.get("pooled_output", None)
    return cond, pooled


class SDCNPromptSchedule:
    """
    Create a prompt schedule for SD-CN-Animation nodes.

    Encodes prompts at each keyframe and linearly interpolates conditioning
    between keyframes. Outputs batched CONDITIONING that the Txt2Vid/Vid2Vid
    nodes consume automatically per frame.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": '"0": "a serene landscape"\n"30": "a vibrant cityscape"',
                    "tooltip": (
                        'Keyframe schedule: "frame_number": "prompt"\n'
                        "Prompts interpolate smoothly between keyframes. "
                        "Frame count is controlled by the Txt2Vid/Vid2Vid node."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "create_schedule"
    CATEGORY = "SD-CN-Animation"
    DESCRIPTION = (
        "Parses a prompt schedule and outputs batched conditioning. "
        "Wire into Txt2Vid/Vid2Vid positive/negative for per-frame prompt changes."
    )

    def create_schedule(self, clip, text):
        keyframes = parse_schedule(text)

        if not keyframes:
            raise ValueError(
                "Could not parse any keyframes from the schedule. "
                'Use format: "0": "prompt here"'
            )

        # Total frames covers up to the last keyframe + 1
        num_frames = keyframes[-1][0] + 1

        # Encode all unique prompts and cache
        unique_prompts = list(dict.fromkeys(p for _, p in keyframes))
        encoded = {}
        for prompt_text in unique_prompts:
            cond, pooled = encode_prompt(clip, prompt_text)
            encoded[prompt_text] = (cond, pooled)

        # Build per-frame conditioning by interpolating between keyframes
        cond_list = []
        pooled_list = []
        has_pooled = encoded[unique_prompts[0]][1] is not None

        for frame_idx in range(num_frames):
            # Find bracketing keyframes
            prev_kf = keyframes[0]
            next_kf = keyframes[-1]
            for j in range(len(keyframes) - 1):
                if keyframes[j][0] <= frame_idx <= keyframes[j + 1][0]:
                    prev_kf = keyframes[j]
                    next_kf = keyframes[j + 1]
                    break
            else:
                if frame_idx <= keyframes[0][0]:
                    prev_kf = next_kf = keyframes[0]
                elif frame_idx >= keyframes[-1][0]:
                    prev_kf = next_kf = keyframes[-1]

            prev_cond, prev_pooled = encoded[prev_kf[1]]
            next_cond, next_pooled = encoded[next_kf[1]]

            # Compute interpolation weight
            if prev_kf[0] == next_kf[0]:
                weight = 0.0
            else:
                weight = (frame_idx - prev_kf[0]) / (next_kf[0] - prev_kf[0])

            # Pad tensors to same seq_len if needed (different prompts = different token counts)
            if prev_cond.shape[1] != next_cond.shape[1]:
                max_len = max(prev_cond.shape[1], next_cond.shape[1])
                if prev_cond.shape[1] < max_len:
                    prev_cond = torch.nn.functional.pad(
                        prev_cond, (0, 0, 0, max_len - prev_cond.shape[1])
                    )
                if next_cond.shape[1] < max_len:
                    next_cond = torch.nn.functional.pad(
                        next_cond, (0, 0, 0, max_len - next_cond.shape[1])
                    )

            # Lerp
            blended_cond = prev_cond * (1.0 - weight) + next_cond * weight
            cond_list.append(blended_cond)

            if has_pooled and prev_pooled is not None and next_pooled is not None:
                blended_pooled = prev_pooled * (1.0 - weight) + next_pooled * weight
                pooled_list.append(blended_pooled)

        # Stack into batched tensors
        final_cond = torch.cat(cond_list, dim=0)  # (num_frames, seq_len, hidden_dim)

        cond_dict = {}
        if pooled_list:
            final_pooled = torch.cat(pooled_list, dim=0)  # (num_frames, pooled_dim)
            cond_dict["pooled_output"] = final_pooled

        return ([[final_cond, cond_dict]],)


NODE_CLASS_MAPPINGS = {
    "SDCNPromptSchedule": SDCNPromptSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDCNPromptSchedule": "SD-CN Prompt Schedule",
}
