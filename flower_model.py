"""
FloweR - Optical Flow Reconstruction model
Original: https://github.com/volotat/SD-CN-Animation
Ported to ComfyUI custom node pack.

U-Net encoder-decoder that takes 4 consecutive frames and predicts:
  - Optical flow (2 channels)
  - Occlusion mask (1 channel)
  - Next frame estimate (3 channels)
"""

import torch
import torch.nn as nn


class FloweR(nn.Module):
    def __init__(self, input_size=(384, 384), window_size=4):
        super(FloweR, self).__init__()

        self.input_size = input_size
        self.window_size = window_size

        # 2 channels for optical flow
        # 1 channel for occlusion mask
        # 3 channels for next frame prediction
        self.out_channels = 6

        ### DOWNSCALE ###
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3 * self.window_size, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H x W x 128

        self.conv_block_2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H/2 x W/2 x 128

        self.conv_block_3 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H/4 x W/4 x 128

        self.conv_block_4 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H/8 x W/8 x 128

        self.conv_block_5 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H/16 x W/16 x 128

        self.conv_block_6 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H/32 x W/32 x 128

        self.conv_block_7 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H/64 x W/64 x 128

        self.conv_block_8 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H/128 x W/128 x 128

        ### UPSCALE ###
        self.conv_block_9 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        self.conv_block_10 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        self.conv_block_11 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        self.conv_block_12 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        self.conv_block_13 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        self.conv_block_14 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        self.conv_block_15 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        self.conv_block_16 = nn.Conv2d(128, self.out_channels, kernel_size=3, stride=1, padding='same')

    def forward(self, input_frames):
        if input_frames.size(1) != self.window_size:
            raise Exception(
                f'Shape of the input is not compatible. '
                f'There should be exactly {self.window_size} frames in an input video.'
            )

        h, w = self.input_size
        # batch, frames, height, width, colors
        input_frames_permuted = input_frames.permute((0, 1, 4, 2, 3))
        # batch, frames, colors, height, width

        in_x = input_frames_permuted.reshape(-1, self.window_size * 3, self.input_size[0], self.input_size[1])

        ### DOWNSCALE ###
        block_1_out = self.conv_block_1(in_x)
        block_2_out = self.conv_block_2(block_1_out)
        block_3_out = self.conv_block_3(block_2_out)
        block_4_out = self.conv_block_4(block_3_out)
        block_5_out = self.conv_block_5(block_4_out)
        block_6_out = self.conv_block_6(block_5_out)
        block_7_out = self.conv_block_7(block_6_out)
        block_8_out = self.conv_block_8(block_7_out)

        ### UPSCALE with skip connections ###
        block_9_out = block_7_out + self.conv_block_9(block_8_out)
        block_10_out = block_6_out + self.conv_block_10(block_9_out)
        block_11_out = block_5_out + self.conv_block_11(block_10_out)
        block_12_out = block_4_out + self.conv_block_12(block_11_out)
        block_13_out = block_3_out + self.conv_block_13(block_12_out)
        block_14_out = block_2_out + self.conv_block_14(block_13_out)
        block_15_out = block_1_out + self.conv_block_15(block_14_out)

        block_16_out = self.conv_block_16(block_15_out)
        out = block_16_out.reshape(-1, self.out_channels, self.input_size[0], self.input_size[1])

        device = out.get_device()

        pred_flow = out[:, :2, :, :] * 255  # (-255, 255)
        pred_occl = (out[:, 2:3, :, :] + 1) / 2  # [0, 1]
        pred_next = out[:, 3:6, :, :]

        # Generate sampling grids
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
        flow_grid = torch.stack((grid_x, grid_y), dim=0).float()
        flow_grid = flow_grid.unsqueeze(0).to(device=device)
        flow_grid = flow_grid + pred_flow

        flow_grid[:, 0, :, :] = 2 * flow_grid[:, 0, :, :] / (w - 1) - 1
        flow_grid[:, 1, :, :] = 2 * flow_grid[:, 1, :, :] / (h - 1) - 1
        flow_grid = flow_grid.permute(0, 2, 3, 1)

        previous_frame = input_frames_permuted[:, -1, :, :, :]
        sampling_mode = "bilinear" if self.training else "nearest"
        warped_frame = torch.nn.functional.grid_sample(
            previous_frame, flow_grid, mode=sampling_mode,
            padding_mode="reflection", align_corners=False
        )
        alpha_mask = torch.clip(pred_occl * 10, 0, 1) * 0.04
        pred_next = torch.clip(pred_next, -1, 1)
        warped_frame = torch.clip(warped_frame, -1, 1)
        next_frame = pred_next * alpha_mask + warped_frame * (1 - alpha_mask)

        res = torch.cat((pred_flow / 255, pred_occl * 2 - 1, next_frame), dim=1)

        # batch, channels, height, width -> batch, height, width, channels
        res = res.permute((0, 2, 3, 1))
        return res
