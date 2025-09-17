# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import ttnn
import argparse
from loguru import logger
from PIL import Image
import torchvision.transforms as transforms

from models.experimental.SSR.reference.SSR.model.ssr import SSR, SSR_wo_conv
from models.experimental.SSR.tt.ssr import TTSSR, TTSSR_wo_conv

from models.experimental.SSR.reference.SSR.model.net_blocks import window_reverse
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import tt2torch_tensor
from models.utility_functions import (
    comp_pcc,
)

from models.experimental.SSR.tests.test_ssr import create_ssr_preprocessor


class Args:
    """Args class for SSR model"""

    def __init__(self):
        self.token_size = 4
        self.imgsz = 256
        self.patchsz = 2
        self.pretrain = False
        self.ckpt = None
        self.dim = 96


def load_image(image_path, target_size=(256, 256)):
    """Load and preprocess image for SSR model"""
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ]
    )

    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 256, 256)

    return image_tensor


def save_tensor_as_image(tensor, output_path):
    """Save tensor as image"""
    # Remove batch dimension and convert to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension

    # Convert BFloat16 to Float32 if needed
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)

    # Clamp values to [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to PIL Image
    transform = transforms.ToPILImage()
    image = transform(tensor)

    # Save image
    image.save(output_path)
    logger.info(f"Image saved to: {output_path}")


def run_ssr_inference(input_image_path, output_dir="models/experimental/SSR/demo/images/", with_conv=True):
    """Run SSR model inference on input image"""

    # Load input image
    logger.info(f"Loading image from: {input_image_path}")
    x = load_image(input_image_path)
    logger.info(f"Input image shape: {x.shape}")

    torch.manual_seed(0)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create args
    args = Args()
    num_cls = 1
    depth = [1]
    num_heads = [1]

    # Create reference PyTorch model
    if with_conv:
        ref_model = SSR(args, num_cls, depth=depth, num_heads=num_heads)
    else:
        ref_model = SSR_wo_conv(args, num_cls, depth=depth, num_heads=num_heads)
    ref_model.eval()

    # Get reference output
    logger.info("Running PyTorch reference model...")
    with torch.no_grad():
        ref_sr, ref_patch_fea3, ref_patch_fea2, ref_patch_fea1 = ref_model(x)

    # Save reference output
    ref_output_path = os.path.join(output_dir, "reference_output.png")
    logger.info("Saving PyTorch reference output...")
    save_tensor_as_image(ref_sr, ref_output_path)

    # Open TTNN device with larger L1 cache to handle memory requirements
    device = ttnn.open_device(device_id=0, l1_small_size=32768)

    try:
        # Preprocess model parameters
        logger.info("Preprocessing model parameters...")
        parameters = preprocess_model_parameters(
            initialize_model=lambda: ref_model,
            custom_preprocessor=create_ssr_preprocessor(device, args, num_cls, depth),
            device=device,
        )

        # Create TTNN model
        logger.info("Creating TTNN model...")
        if with_conv:
            tt_model = TTSSR(
                device=device,
                parameters=parameters,
                args=args,
                num_cls=num_cls,
                depth=depth,
                num_heads=num_heads,
            )
        else:
            tt_model = TTSSR_wo_conv(
                device=device,
                parameters=parameters,
                args=args,
                num_cls=num_cls,
                depth=depth,
                num_heads=num_heads,
            )

        # Convert input to TTNN tensor
        logger.info("Converting input to TTNN tensor...")
        tt_input = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

        # Run TTNN model
        logger.info("Running TTNN model inference...")
        tt_sr, tt_patch_fea3 = tt_model(tt_input)
        # Convert back to torch tensors
        tt_torch_sr = tt2torch_tensor(tt_sr)
        tt_torch_patch_fea3 = tt2torch_tensor(tt_patch_fea3)
        tt_torch_sr = tt_torch_sr.permute(0, 3, 1, 2)
        if not with_conv:
            _, _, H, W = x.shape
            tt_torch_sr = window_reverse(tt_torch_sr.permute(0, 2, 3, 1), window_size=H, H=H * 4, W=W * 4)
            tt_torch_sr = tt_torch_sr.permute(0, 3, 1, 2)

        # Save TTNN output image
        ttnn_output_path = os.path.join(output_dir, "ttnn_output.png")
        logger.info("Saving TTNN super-resolved image...")
        save_tensor_as_image(tt_torch_sr, ttnn_output_path)

        # Compare outputs (optional - for validation)
        sr_pass, sr_pcc_message = comp_pcc(ref_sr, tt_torch_sr, 0.90)
        logger.info(f"SR Output PCC: {sr_pcc_message}")

        if sr_pass:
            logger.info("TTSSR inference completed successfully!")
        else:
            logger.warning("TTSSR inference completed with quality concerns.")

        logger.info(f"Reference output saved to: {ref_output_path}")
        logger.info(f"TTNN output saved to: {ttnn_output_path}")

        return tt_sr, ref_sr

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSR Super-Resolution Inference")
    parser.add_argument(
        "--input",
        type=str,
        default="models/experimental/SSR/demo/images/ssr_test_image.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/experimental/SSR/demo/images/", help="Directory to save output images"
    )
    parser.add_argument("--with-conv", action="store_true", default=False, help="Use SSR model with conv layers")

    args = parser.parse_args()

    run_ssr_inference(args.input, args.output_dir, args.with_conv)
