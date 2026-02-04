# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import re

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _to_numpy_float64(x):
    """Convert to numpy float64, detaching if torch.Tensor (e.g. requires_grad)."""
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def plot_abs_diff(torch_output, ttnn_output, plot_name):
    """
    Plot elementwise abs(torch - ttnn) per batch, same style as analyze_preds_npy.
    Saves to tests/pcc/pcc_plots/<plot_name>.png
    """
    if not HAS_MATPLOTLIB:
        return
    torch_f = _to_numpy_float64(torch_output)
    tt_f = _to_numpy_float64(ttnn_output)
    if torch_f.shape != tt_f.shape:
        return
    batch_size = torch_f.shape[0]
    fig, axes = plt.subplots(batch_size, 1, figsize=(10, 4 * batch_size), sharex=True)
    if batch_size == 1:
        axes = [axes]
    max_points = 50_000  # Downsample to avoid slow plotting with 500k+ points
    for b in range(batch_size):
        out_b = np.abs(torch_f[b] - tt_f[b]).flatten()
        n = out_b.size
        if n > max_points:
            step = n // max_points
            indices = np.arange(0, n, step)[:max_points]
            out_plot = out_b[indices]
            x_plot = indices
        else:
            out_plot = out_b
            x_plot = np.arange(n)
        axes[b].plot(x_plot, out_plot, color="steelblue", linewidth=0.3)
        axes[b].set_ylabel("|torch - ttnn|")
        axes[b].set_title(f"Batch {b}")
    axes[-1].set_xlabel("Element index")
    fig.suptitle("Elementwise abs(torch - ttnn)", y=1.02)
    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "pcc_plots")
    os.makedirs(out_dir, exist_ok=True)
    safe_name = re.sub(r"[^\w\-.]", "_", plot_name)
    if not safe_name.endswith(".png"):
        safe_name = safe_name + ".png"
    path = os.path.join(out_dir, safe_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved abs diff plot to {path}")
