# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import ttnn
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from models.experimental.swin2sr.reference.swin2sr import Swin2SR as TorchSwin2SR
from models.experimental.swin2sr.tt.tt_swin2sr import TtSwin2SR
from models.experimental.swin2sr.tests.pcc.test_ttnn_swin2sr import create_swin2sr_preprocessor
from models.experimental.swin2sr.tt.utils import ensure_checkpoint_downloaded


def load_image(image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    h, w = img.shape[:2]
    img = img.astype(np.float32) / 255.0
    img = img[:, :, [2, 1, 0]]
    img = np.transpose(img, (2, 0, 1))
    return img, (h, w)


def prepare_ttnn_input(torch_tensor: torch.Tensor, device: ttnn.Device) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def save_output_image(tensor: ttnn.Tensor, output_path: str, original_size: tuple[int, int], upscale: int):
    output = ttnn.to_torch(tensor)
    img = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()

    if img.ndim == 3:
        target_h = original_size[0] * upscale
        target_w = original_size[1] * upscale
        if img.shape[1] > target_h:
            img = img[:, :target_h, :]
        if img.shape[2] > target_w:
            img = img[:, :, :target_w]
        img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))

    img = (img * 255.0).round().astype(np.uint8)
    cv2.imwrite(output_path, img)
    logger.info(f"Saved output image to: {output_path}")


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: ttnn.Device,
    img_size: int | tuple[int, int],
    embed_dim: int = 180,
    depths: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
    num_heads: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
    window_size: int = 8,
    mlp_ratio: float = 2.0,
    upscale: int = 2,
    resi_connection: str = "1conv",
) -> tuple[TorchSwin2SR, TtSwin2SR]:
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    params = checkpoint["params"] if "params" in checkpoint else checkpoint

    logger.info("Creating PyTorch model...")
    torch_model = TorchSwin2SR(
        img_size=img_size,
        patch_size=1,
        in_chans=3,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        upscale=upscale,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection=resi_connection,
    )

    torch_model.load_state_dict(params, strict=False)
    torch_model.eval()

    logger.info("Preprocessing parameters for TTNN...")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_swin2sr_preprocessor(device),
        device=device,
    )

    logger.info("Creating TTNN model...")
    tt_model = TtSwin2SR(
        device=device,
        parameters=parameters,
        img_size=img_size,
        patch_size=1,
        in_chans=3,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        upscale=upscale,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection=resi_connection,
    )

    return torch_model, tt_model


def process_image_tiled(
    tt_model: TtSwin2SR,
    device: ttnn.Device,
    img_array: np.ndarray,
    tile_size: int,
    tile_overlap: int,
    window_size: int,
    upscale: int,
) -> np.ndarray:
    C, H, W = img_array.shape
    assert tile_size % window_size == 0, f"tile_size ({tile_size}) must be a multiple of window_size ({window_size})"

    img_tensor = torch.from_numpy(img_array).float().unsqueeze(0)
    stride = tile_size - tile_overlap
    h_idx_list = list(range(0, H - tile_size, stride)) + [max(0, H - tile_size)]
    w_idx_list = list(range(0, W - tile_size, stride)) + [max(0, W - tile_size)]

    output_h = H * upscale
    output_w = W * upscale
    E = torch.zeros(1, C, output_h, output_w, dtype=torch.float32)
    W_mask = torch.zeros(1, C, output_h, output_w, dtype=torch.float32)

    logger.info(f"Processing {len(h_idx_list)}x{len(w_idx_list)} = {len(h_idx_list) * len(w_idx_list)} tiles...")

    for tile_idx, (h_idx, w_idx) in enumerate([(h, w) for h in h_idx_list for w in w_idx_list]):
        if (tile_idx + 1) % 10 == 0:
            logger.info(f"  Processed {tile_idx + 1}/{len(h_idx_list) * len(w_idx_list)} tiles...")

        in_patch = img_tensor[:, :, h_idx : h_idx + tile_size, w_idx : w_idx + tile_size]

        if in_patch.shape[2] < tile_size or in_patch.shape[3] < tile_size:
            pad_h = tile_size - in_patch.shape[2]
            pad_w = tile_size - in_patch.shape[3]
            in_patch = torch.nn.functional.pad(in_patch, (0, pad_w, 0, pad_h), mode="reflect")

        tt_input = prepare_ttnn_input(in_patch, device)
        tt_output = tt_model.forward(tt_input)
        out_patch = ttnn.to_torch(tt_output)
        out_patch = out_patch.float().cpu()

        patch_h = min(tile_size * upscale, output_h - h_idx * upscale)
        patch_w = min(tile_size * upscale, output_w - w_idx * upscale)
        out_patch = out_patch[:, :, :patch_h, :patch_w]

        out_h_start = h_idx * upscale
        out_w_start = w_idx * upscale
        out_h_end = out_h_start + patch_h
        out_w_end = out_w_start + patch_w

        E[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += out_patch
        W_mask[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += 1.0

    W_mask = torch.clamp(W_mask, min=1.0)
    output = E / W_mask

    return output.squeeze(0).numpy()


def run_demo_tiled(
    image_path: str,
    checkpoint_path: str,
    output_path: str = None,
    scale: int = 2,
    tile_size: int = 64,
    tile_overlap: int = 32,
    device_id: int = 0,
):
    logger.info("=" * 80)
    logger.info("Swin2SR TTNN Demo - Tiled Processing")
    logger.info("=" * 80)
    logger.info(f"Input image: {image_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Upscale factor: {scale}x")
    logger.info(f"Tile size: {tile_size}x{tile_size}")
    logger.info(f"Tile overlap: {tile_overlap} pixels")
    logger.info("=" * 80)

    window_size = 8
    if tile_size % window_size != 0:
        raise ValueError(f"tile_size ({tile_size}) must be a multiple of window_size ({window_size})")

    logger.info("\n[1/4] Opening TT device...")
    device = ttnn.open_device(device_id=device_id)

    try:
        logger.info("\n[2/4] Loading and preprocessing image...")
        img_array, (h, w) = load_image(image_path)
        logger.info(f"Image size: {h}x{w}")

        img_size = tile_size
        logger.info(f"Using img_size={img_size} for model initialization (matches tile_size and checkpoint)")

        logger.info("\n[3/4] Loading model from checkpoint...")
        torch_model, tt_model = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            img_size=img_size,
            embed_dim=180,
            depths=(6, 6, 6, 6, 6, 6),
            num_heads=(6, 6, 6, 6, 6, 6),
            window_size=window_size,
            mlp_ratio=2.0,
            upscale=scale,
            resi_connection="1conv",
        )

        logger.info("\n[4/4] Processing image in tiles...")
        output_array = process_image_tiled(
            tt_model=tt_model,
            device=device,
            img_array=img_array,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            window_size=window_size,
            upscale=scale,
        )

        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            ext = os.path.splitext(image_path)[1]
            output_path = f"{base_name}_ttnn_tiled_output{ext}"

        logger.info(f"\nSaving output...")
        output_tensor = torch.from_numpy(output_array).unsqueeze(0)
        save_output_image(
            ttnn.from_torch(output_tensor, device=device, dtype=ttnn.bfloat16), output_path, (h, w), scale
        )

        logger.info(f"\nOutput shape: {output_array.shape}")
        logger.info(
            f"Upscale factor achieved: {output_array.shape[1] / h:.2f}x (height), {output_array.shape[2] / w:.2f}x (width)"
        )

        logger.info("\n" + "=" * 80)
        logger.info("Demo completed successfully!")
        logger.info("=" * 80)

    finally:
        logger.info("\nClosing TT device...")
        ttnn.close_device(device)


def main():
    parser = argparse.ArgumentParser(description="Swin2SR TTNN Demo with Tiled Processing")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (if not provided, will auto-select based on scale)",
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save output image")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4, 8], help="Upscale factor")
    parser.add_argument("--tile-size", type=int, default=64, help="Tile size (must be multiple of 8, default: 64)")
    parser.add_argument("--tile-overlap", type=int, default=32, help="Overlap between tiles (default: 32)")
    parser.add_argument("--device-id", type=int, default=0, help="TTNN device ID")

    args = parser.parse_args()

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    checkpoint_dir = os.path.join(workspace_root, "models", "experimental", "swin2sr", "resources", "checkpoints")

    if not os.path.isabs(args.image):
        args.image = (
            os.path.join(workspace_root, args.image)
            if args.image.startswith("models/")
            else os.path.abspath(args.image)
        )

    if args.checkpoint is None:
        checkpoint_map = {
            2: "models/experimental/swin2sr/resources/checkpoints/Swin2SR_ClassicalSR_X2_64.pth",
            4: "models/experimental/swin2sr/resources/checkpoints/Swin2SR_ClassicalSR_X4_64.pth",
        }
        if args.scale in checkpoint_map:
            args.checkpoint = checkpoint_map[args.scale]
            logger.info(f"Auto-selected checkpoint for scale {args.scale}x: {args.checkpoint}")
        else:
            raise ValueError(f"No default checkpoint available for scale {args.scale}x. Please provide --checkpoint")

    # Download checkpoint only if needed (based on selected scale)
    if args.checkpoint.endswith("Swin2SR_ClassicalSR_X2_64.pth"):
        ensure_checkpoint_downloaded(
            "Swin2SR_ClassicalSR_X2_64.pth",
            "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X2_64.pth",
            checkpoint_dir,
        )
    elif args.checkpoint.endswith("Swin2SR_ClassicalSR_X4_64.pth"):
        ensure_checkpoint_downloaded(
            "Swin2SR_ClassicalSR_X4_64.pth",
            "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X4_64.pth",
            checkpoint_dir,
        )

    if not os.path.isabs(args.checkpoint):
        args.checkpoint = (
            os.path.join(workspace_root, args.checkpoint)
            if args.checkpoint.startswith("models/")
            else os.path.abspath(args.checkpoint)
        )

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.tile_size % 8 != 0:
        raise ValueError(f"tile_size ({args.tile_size}) must be a multiple of 8")

    run_demo_tiled(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        scale=args.scale,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        device_id=args.device_id,
    )


if __name__ == "__main__":
    main()
