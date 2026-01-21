# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtConvTranspose2D(LightweightModule):
    def __init__(
        self,
        conv_transpose,
        conv_transpose_pth,
        device=None,
        activation=None,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        is_dealloc_act=False,
        return_dims=False,
        reshape_output=False,
        memory_config=None,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
    ):
        super().__init__()
        self.conv_transpose = conv_transpose
        self.device = device
        self.in_channels = conv_transpose.in_channels
        self.out_channels = conv_transpose.out_channels
        self.kernel_size = conv_transpose.kernel_size
        self.stride = conv_transpose.stride
        self.padding = conv_transpose.padding
        self.output_padding = conv_transpose.output_padding

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=math_approx_mode,
        )

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=shard_layout,
            deallocate_activation=is_dealloc_act,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            activation=activation,
        )

        if conv_transpose_pth.bias is not None:
            self.bias = ttnn.from_device(conv_transpose_pth.bias)
        else:
            self.bias = None

        self.weight = ttnn.from_device(conv_transpose_pth.weight)
        self.activation_dtype = activation_dtype
        self.return_dims = return_dims
        self.reshape_output = reshape_output
        self.memory_config = memory_config
        self._weights_prepared = False

    def forward(self, x, shape=None):
        if shape is not None:
            batch_size = shape[0]
            input_height = shape[1]
            input_width = shape[2]
        else:
            batch_size = x.shape[0]
            input_height = x.shape[1]
            input_width = x.shape[2]

        [x, [_out_height, _out_width], [self.weight, self.bias]] = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            in_channels=self.conv_transpose.in_channels,
            out_channels=self.conv_transpose.out_channels,
            device=self.device,
            kernel_size=self.conv_transpose.kernel_size,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
            dilation=self.conv_transpose.dilation,
            groups=self.conv_transpose.groups,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.activation_dtype,
            memory_config=self.memory_config,
            mirror_kernel=True,
        )

        shape = (batch_size, _out_height, _out_width, x.shape[-1])
        if self.reshape_output:
            x = ttnn.reshape(x, shape, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.return_dims:
            return x, shape
        else:
            return x


def download_weights(weights_path=None, model_name="ctdet_coco_dlav0_1x"):
    import shutil
    import subprocess

    if weights_path and os.path.exists(weights_path):
        logger.info(f"Using provided weights file: {weights_path}")
        return weights_path

    centernet_dir = Path(__file__).parent.parent
    default_weights_path = centernet_dir / f"{model_name}.pth"

    if default_weights_path.exists():
        logger.info(f"Found weights at: {default_weights_path}")
        return str(default_weights_path)

    alternative_paths = [
        centernet_dir / "wt" / f"{model_name}.pth",
        centernet_dir / "weights" / f"{model_name}.pth",
    ]

    for alt_path in alternative_paths:
        if alt_path.exists():
            logger.info(f"Found weights at alternative location: {alt_path}")
            try:
                default_weights_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Copying weights to default location: {default_weights_path}")
                shutil.copy2(str(alt_path), str(default_weights_path))
                return str(default_weights_path)
            except Exception as e:
                logger.warning(f"Could not copy weights to default location: {e}")
                return str(alt_path)

    # Weights not found - attempt automatic download
    logger.warning(f"Weights file '{model_name}.pth' not found!")
    logger.info(f"Attempting automatic download...")

    # Google Drive file ID for ctdet_coco_dlav0_1x.pth
    file_id = "1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT"

    # Create directory if it doesn't exist
    default_weights_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to install gdown if not available
    import sys

    try:
        import gdown
    except ImportError:
        logger.info("Installing gdown for automatic download...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "gdown", "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("✓ gdown installed successfully")
            import gdown
        except Exception as e:
            logger.warning(f"Could not install gdown: {e}")
            gdown = None

    if gdown is not None:
        try:
            logger.info(f"Downloading weights from Google Drive...")
            logger.info(f"  File: {model_name}.pth")
            logger.info(f"  Size: ~211 MB (this may take a few minutes)")

            folder_url = "https://drive.google.com/drive/folders/1S3NnppRgXea_IG4WeyquJcnOB3I6G-LX"
            temp_download_dir = centernet_dir / "model_zoo"

            logger.info(f"  Downloading from CenterNet MODEL ZOO...")
            result = subprocess.run(
                [sys.executable, "-m", "gdown", "--folder", folder_url, "--remaining-ok", "-O", str(temp_download_dir)],
                capture_output=True,
                text=True,
                timeout=600,
            )

            # Check if the specific file was downloaded
            downloaded_file = temp_download_dir / f"{model_name}.pth"
            if downloaded_file.exists():
                file_size_mb = downloaded_file.stat().st_size / (1024 * 1024)
                if file_size_mb > 100:
                    logger.info(f"  Moving weights to: {default_weights_path}")
                    shutil.move(str(downloaded_file), str(default_weights_path))
                    logger.info(f"✓ Successfully downloaded weights ({file_size_mb:.1f} MB)")
                    return str(default_weights_path)
                else:
                    logger.warning(f"Downloaded file seems too small ({file_size_mb:.1f} MB)")
                    downloaded_file.unlink()
            else:
                logger.warning(f"File {model_name}.pth not found in downloaded folder")
                if result.stderr:
                    logger.warning(f"gdown output: {result.stderr[:300]}")

        except subprocess.TimeoutExpired:
            logger.warning(f"Download timed out after 10 minutes")
        except Exception as e:
            logger.warning(f"Automatic download failed: {e}")
            import traceback

            logger.debug(traceback.format_exc())

    logger.error(f"Could not download weights automatically.")
    logger.info(f"")
    logger.info(f"Please download weights manually:")
    logger.info(f"  Expected location: {default_weights_path}")
    logger.info(f"")
    logger.info(f"Option 1 - Using gdown (recommended):")
    logger.info(f"  pip install gdown")
    logger.info(f"  cd {centernet_dir}")
    logger.info(f"  gdown {file_id} -O {model_name}.pth")
    logger.info(f"")
    logger.info(f"Option 2 - Manual download from browser:")
    logger.info(f"  1. Visit: https://drive.google.com/file/d/{file_id}/view")
    logger.info(f"  2. Click 'Download' button")
    logger.info(f"  3. Save as: {default_weights_path}")
    logger.info(f"     (File size should be ~211 MB)")

    raise FileNotFoundError(
        f"Weights file not found: {default_weights_path}\n"
        f"Please download from: https://drive.google.com/file/d/{file_id}/view"
    )
