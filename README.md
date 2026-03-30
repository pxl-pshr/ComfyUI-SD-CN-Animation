# ComfyUI-SD-CN-Animation

Temporally coherent video generation and stylization for ComfyUI using optical flow. Port of [SD-CN-Animation](https://github.com/volotat/SD-CN-Animation) to ComfyUI custom nodes.

Generate videos from text prompts (**Txt2Vid**) or stylize existing videos (**Vid2Vid**) with any Stable Diffusion checkpoint, ControlNet, and IPAdapter — while maintaining smooth frame-to-frame consistency via optical flow prediction.

![Example output](examples/example.gif)

## How It Works

Both pipelines use a two-pass approach per frame to balance creativity with coherence:

**Txt2Vid** uses the **FloweR** model to predict optical flow, occlusion, and the next frame from the previous 4 frames. Each frame is refined through two SD sampling passes — an inpaint pass guided by the occlusion mask, and a refinement pass for cleanup — with histogram matching to prevent color drift.

**Vid2Vid** uses **RAFT** optical flow estimation between consecutive input video frames to warp the previous stylized frame forward. Occlusion detection identifies areas where the warp breaks down, and those regions get inpainted by SD. A second refinement pass and histogram matching complete the frame.

## Nodes

| Node | Description |
|------|-------------|
| **Load FloweR Model** | Loads FloweR optical flow prediction model |
| **SD-CN Animation Txt2Vid** | Generate video from text prompt using FloweR |
| **Load RAFT Model** | Loads RAFT optical flow estimation model |
| **SD-CN Animation Vid2Vid** | Stylize video frames using RAFT optical flow |
| **FloweR Predict** | Standalone FloweR prediction (debug/visualization) |
| **Histogram Match** | Standalone color histogram matching utility |

## Installation

Install via [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) or clone manually into `ComfyUI/custom_nodes/`.

Model weights (~1.2GB FloweR + ~20MB RAFT) are **downloaded automatically** from [Hugging Face](https://huggingface.co/pxlpshr/ComfyUI-SD-CN-Animation) on first launch.

## Workflows

### Txt2Vid (Text to Video)

```
CheckpointLoader ──→ model ──────────────────→ SD-CN Animation Txt2Vid ──→ VHS_VideoCombine
                 ──→ clip ──→ CLIPTextEncode ──→ positive                   (set fps: 12)
                 ──→ vae  ──────────────────→
LoadFloweRModel ──→ flower_model ───────────→
```

**Key parameters:**
- `num_frames` — total frames to generate (30-120+)
- `processing_strength` — how much SD changes each frame (0.5-0.85). Lower = smoother morphing, higher = more creative variation
- `fix_frame_strength` — refinement pass strength (0.1-0.2). Keep low for stability, set to 0 to skip entirely
- `loop_frames` — set to 10-20 to blend the end back into the first frame for a seamless loop. 0 = disabled

**For smooth morphing animation:** Use `processing_strength: 0.55`, `fix_frame_strength: 0.12`, `num_frames: 90`, and set your video combine node to 12-14 fps.

### Vid2Vid (Video Stylization)

```
VHS_LoadVideo ──→ frames ───────────────────→ SD-CN Animation Vid2Vid ──→ VHS_VideoCombine
CheckpointLoader ──→ model ─────────────────→
                 ──→ clip ──→ CLIPTextEncode → positive
                 ──→ vae  ──────────────────→
LoadRAFTModel ──→ raft_model ───────────────→
```

**Key parameters:**
- `processing_strength` — denoising for main stylization (0.75-0.85)
- `fix_frame_strength` — refinement pass (0.1-0.2, or 0 to skip)
- `blend_alpha` — blend between warped styled frame (0.0) and current input (1.0). Higher values keep more of the original video structure
- `occlusion_mask_blur` — smooths the mask that decides what gets regenerated (2-4)
- `occlusion_mask_trailing` — enable to reduce ghosting artifacts

### ControlNet

Wire a **Load ControlNet Model** node into the `control_net` input:

- **Vid2Vid:** The current input video frame is automatically used as the ControlNet hint each frame. Great with depth, canny, or lineart ControlNets to preserve structure.
- **Txt2Vid:** The FloweR-predicted frame is used as the hint.

### Multiple ControlNets / IPAdapter

Apply additional ControlNets or IPAdapter to the **conditioning before** passing it into the animation node:

```
CLIPTextEncode ──→ IPAdapterApply ──→ ControlNetApply ──→ positive
                                                            ↓
                                                     SDCNVid2Vid ← control_net (per-frame depth/canny)
```

ControlNets applied to conditioning beforehand use static hints. The node's `control_net` input is the one that gets updated per-frame.

## Tips

- **Speed vs quality:** Reducing `steps` to 10-12 speeds things up with minimal quality loss. Skipping the refine pass (`fix_frame_strength: 0`) halves per-frame time.
- **Temporal coherence:** Lower `processing_strength` = smoother motion but less stylization.
- **Frame interpolation:** Run output through a RIFE node before combining into video for extra smoothness.
- **Playback speed:** The node outputs raw frames — set FPS on your video combine node (12 fps is a good default).

## Credits

- **SD-CN-Animation** — original A1111 extension and FloweR model by [volotat](https://github.com/volotat/SD-CN-Animation) (MIT License)
- **RAFT** — Recurrent All-Pairs Field Transforms for Optical Flow by Zachary Teed and Jia Deng, Princeton Vision Lab ([paper](https://arxiv.org/abs/2003.12039), [code](https://github.com/princeton-vl/RAFT), BSD 3-Clause License)

## License

MIT. RAFT model code under BSD 3-Clause (see `raft/` directory).
