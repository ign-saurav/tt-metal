# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import sys

sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.10/site-packages"))
import gdown


def download_bevformerv2_weights():
    import urllib.request

    file_id = "1hC49RBbDW_qZJNHAfAjsmIezTtPKRevc"
    weights_path = "models/experimental/BEVFormerV2/chkpt/bevformer_v2_weights.pth"

    if not os.path.exists(weights_path):
        try:
            print("Downloading weights from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, weights_path, quiet=False)
        except ImportError:
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            urllib.request.urlretrieve(direct_url, weights_path)

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
