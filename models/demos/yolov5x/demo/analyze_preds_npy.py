#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Analyze and compare TTNN vs PyTorch prediction numpy files.
Loads preds_torch_model.npy and preds_tt_model.npy, computes differences,
reports max/mean/std, and plots elementwise difference and Torch vs TTNN scatter.
"""

import argparse
import os
import sys

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_preds(npy_dir: str):
    """Load torch and tt prediction arrays from preds_npy directory."""
    torch_path = os.path.join(npy_dir, "preds_torch_model.npy")
    tt_path = os.path.join(npy_dir, "preds_tt_model.npy")
    if not os.path.isfile(torch_path):
        raise FileNotFoundError(f"Torch preds not found: {torch_path}")
    if not os.path.isfile(tt_path):
        raise FileNotFoundError(f"TT preds not found: {tt_path}")
    torch_preds = np.load(torch_path)
    tt_preds = np.load(tt_path)
    return torch_preds, tt_preds


def analyze(torch_preds: np.ndarray, tt_preds: np.ndarray):
    """Compute elementwise and absolute differences and summary statistics."""
    if torch_preds.shape != tt_preds.shape:
        raise ValueError(f"Shape mismatch: torch {torch_preds.shape} vs tt {tt_preds.shape}")

    torch_f = torch_preds.astype(np.float64)
    tt_f = tt_preds.astype(np.float64)
    elem_diff = torch_f - tt_f  # elementwise difference (Torch - TT)
    abs_diff = np.abs(elem_diff)

    stats = {
        "shape": torch_preds.shape,
        "num_elements": abs_diff.size,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "std_abs_diff": float(np.std(abs_diff)),
        "median_abs_diff": float(np.median(abs_diff)),
    }

    # Per-batch statistics (shape is [batch, 84, 8400])
    batch_size = abs_diff.shape[0]
    per_batch_max = np.max(abs_diff.reshape(batch_size, -1), axis=1)
    per_batch_mean = np.mean(abs_diff.reshape(batch_size, -1), axis=1)
    stats["per_batch_max"] = per_batch_max
    stats["per_batch_mean"] = per_batch_mean

    # Location of max difference
    flat_idx = np.argmax(abs_diff)
    max_idx = np.unravel_index(flat_idx, abs_diff.shape)
    stats["max_diff_at"] = max_idx
    stats["torch_val_at_max"] = float(torch_preds[max_idx])
    stats["tt_val_at_max"] = float(tt_preds[max_idx])

    # TTNN batch 0 vs batch 1: are they the same?
    if batch_size >= 2:
        tt_b01_diff = np.abs(tt_f[0] - tt_f[1])
        stats["tt_batch0_vs_batch1_max_diff"] = float(np.max(tt_b01_diff))
        stats["tt_batch0_vs_batch1_mean_diff"] = float(np.mean(tt_b01_diff))
        stats["tt_batches_same"] = bool(np.allclose(tt_f[0], tt_f[1]))
    else:
        stats["tt_batch0_vs_batch1_max_diff"] = None
        stats["tt_batch0_vs_batch1_mean_diff"] = None
        stats["tt_batches_same"] = None

    return elem_diff, abs_diff, stats


def print_report(stats: dict):
    """Print a text report of the analysis."""
    print("=" * 60)
    print("Prediction comparison (Torch vs TTNN)")
    print("=" * 60)
    print(f"Shape: {stats['shape']}")
    print(f"Total elements: {stats['num_elements']}")
    print()
    print("Absolute difference statistics:")
    print(f"  Max absolute difference:  {stats['max_abs_diff']:.6g}")
    print(f"  Mean absolute difference: {stats['mean_abs_diff']:.6g}")
    print(f"  Std absolute difference:  {stats['std_abs_diff']:.6g}")
    print(f"  Median absolute difference: {stats['median_abs_diff']:.6g}")
    print()
    print("Per-batch max absolute difference:")
    for i, v in enumerate(stats["per_batch_max"]):
        print(f"  Batch index {i}: {v:.6g}")
    print()
    print("Per-batch mean absolute difference:")
    for i, v in enumerate(stats["per_batch_mean"]):
        print(f"  Batch index {i}: {v:.6g}")
    print()
    print(f"Max diff at index {stats['max_diff_at']}:")
    print(f"  Torch value: {stats['torch_val_at_max']:.6g}")
    print(f"  TT value:    {stats['tt_val_at_max']:.6g}")
    print()
    if stats.get("tt_batches_same") is not None:
        print("TTNN batch 0 vs batch 1:")
        print(f"  Batches identical (allclose): {stats['tt_batches_same']}")
        print(f"  Max |batch0 - batch1|:       {stats['tt_batch0_vs_batch1_max_diff']:.6g}")
        print(f"  Mean |batch0 - batch1|:      {stats['tt_batch0_vs_batch1_mean_diff']:.6g}")
    print("=" * 60)


def plot_results(
    torch_preds: np.ndarray,
    tt_preds: np.ndarray,
    out_dir: str,
):
    """Treat each batch separately: per-batch elementwise diff and scatter (Torch x vs TTNN y)."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available; skipping plots.")
        return

    os.makedirs(out_dir, exist_ok=True)
    batch_size = torch_preds.shape[0]
    torch_f = torch_preds.astype(np.float64)
    tt_f = tt_preds.astype(np.float64)

    # 1) Elementwise abs diff: one subplot per batch, each plots that batch's array
    fig, axes = plt.subplots(batch_size, 1, figsize=(10, 4 * batch_size), sharex=True)
    if batch_size == 1:
        axes = [axes]
    for b in range(batch_size):
        out_b = np.abs(torch_f[b] - tt_f[b]).flatten()
        axes[b].plot(out_b, color="steelblue", linewidth=0.3)
        axes[b].set_ylabel("|preds_torch - preds_ttnn|")
        axes[b].set_title(f"Batch {b}")
    axes[-1].set_xlabel("Element index")
    fig.suptitle("Elementwise abs(preds_torch - preds_ttnn)", y=1.02)
    fig.tight_layout()
    diff_path = os.path.join(out_dir, "elementwise_difference.png")
    fig.savefig(diff_path, dpi=150)
    plt.close(fig)
    print(f"Saved elementwise difference plot to {diff_path}")

    # 2) Scatter: one subplot per batch (Torch x vs TTNN y). Subsample if too many points.
    max_points = 50_000
    fig, axes = plt.subplots(batch_size, 1, figsize=(7, 7 * batch_size))
    if batch_size == 1:
        axes = [axes]
    for b in range(batch_size):
        torch_flat = torch_f[b].flatten()
        tt_flat = tt_f[b].flatten()
        n = torch_flat.size
        if n > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=max_points, replace=False)
            x_plot = torch_flat[idx]
            y_plot = tt_flat[idx]
        else:
            x_plot = torch_flat
            y_plot = tt_flat
        axes[b].scatter(x_plot, y_plot, alpha=0.3, s=1, c="steelblue")
        lims = [
            min(x_plot.min(), y_plot.min()),
            max(x_plot.max(), y_plot.max()),
        ]
        axes[b].plot(lims, lims, "r--", linewidth=1, label="y = x (perfect match)")
        axes[b].set_xlabel("Torch output")
        axes[b].set_ylabel("TTNN output")
        axes[b].set_title(f"Batch {b}")
        axes[b].set_aspect("equal")
        axes[b].legend()
    fig.suptitle("Torch vs TTNN output (scatter)", y=1.02)
    fig.tight_layout()
    scatter_path = os.path.join(out_dir, "torch_vs_ttnn_scatter.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot to {scatter_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Torch vs TTNN prediction .npy files")
    parser.add_argument(
        "npy_dir",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "runs", "preds_npy"),
        help="Directory containing preds_torch_model.npy and preds_tt_model.npy",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for plot outputs (default: npy_dir/analysis_plots)",
    )
    args = parser.parse_args()

    npy_dir = os.path.abspath(args.npy_dir)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(npy_dir, "analysis_plots")

    try:
        torch_preds, tt_preds = load_preds(npy_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    elem_diff, abs_diff, stats = analyze(torch_preds, tt_preds)
    print_report(stats)

    if not args.no_plot:
        plot_results(torch_preds, tt_preds, out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
