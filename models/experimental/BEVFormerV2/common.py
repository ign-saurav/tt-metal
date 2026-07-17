# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import sys
import site


def ensure_gdown_installed():
    """
    Ensure gdown is installed and available for import.
    Installs gdown if not present and adds user site-packages to sys.path.
    """
    try:
        import gdown
        return gdown
    except ImportError:
        # Install gdown to user site-packages
        os.system("pip install gdown")
        
        # Add user site-packages to Python path dynamically
        user_site_packages = site.getusersitepackages()
        if user_site_packages not in sys.path:
            sys.path.insert(0, user_site_packages)
        
        # Import gdown after installation
        import gdown
        return gdown


def download_bevformerv2_weights():
    """
    Download BEVFormerV2 weights from Google Drive.
    Creates checkpoint directory if it doesn't exist.
    """
    # Ensure gdown is available
    gdown = ensure_gdown_installed()
    
    file_id = "1hC49RBbDW_qZJNHAfAjsmIezTtPKRevc"
    weights_path = "models/experimental/BEVFormerV2/chkpt/bevformer_v2_weights.pth"

    if not os.path.exists(weights_path):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        print("Downloading weights from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, weights_path, quiet=False)

    return weights_path


def load_torch_model(torch_model, layer="", model_location_generator=None):
    weights_path = "models/experimental/BEVFormerV2/chkpt/bevformer_v2_weights.pth"
    if not os.path.exists(weights_path):
        weights_path = download_bevformerv2_weights()

    torch_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(torch_dict, dict) and "state_dict" in torch_dict:
        torch_dict = torch_dict["state_dict"]

    if layer == "":
        new_state_dict = torch_dict
    else:
        state_dict = {k: v for k, v in torch_dict.items() if (k.startswith(layer))}
        new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))

    torch_model.load_state_dict(new_state_dict, strict=False)
    torch_model.eval()

    return torch_model
