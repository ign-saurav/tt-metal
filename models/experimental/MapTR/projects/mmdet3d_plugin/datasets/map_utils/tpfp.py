import numpy as np

from .tpfp_chamfer import custom_polyline_score


def custom_tpfp_gen(gen_lines, gt_lines, threshold=0.5, metric="chamfer"):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """
    if metric == "chamfer":
        if threshold > 0:
            threshold = -threshold
    # else:
    #     raise NotImplementedError

    # import pdb;pdb.set_trace()
    num_gens = gen_lines.shape[0]
    num_gts = gt_lines.shape[0]

    # tp and fp
    tp = np.zeros((num_gens), dtype=np.float32)
    fp = np.zeros((num_gens), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if num_gts == 0:
        fp[...] = 1
        return tp, fp

    if num_gens == 0:
        return tp, fp

    gen_scores = gen_lines[:, -1]  # n
    # distance matrix: n x m

    # Debug: Log GT and prediction vectors before matching
    import os

    debug_enabled = os.environ.get("MAPTR_DEBUG_EVAL", "0") == "1"
    if debug_enabled and num_gens > 0 and num_gts > 0:
        pred_vectors = gen_lines[:, :-1].reshape(num_gens, -1, 2)
        gt_vectors = gt_lines.reshape(num_gts, -1, 2)

        print(f"\n=== Evaluation Debug: Before Chamfer Distance ===")
        for i in range(min(3, len(gt_vectors))):
            gt_v = gt_vectors[i]
            print(f"\nGT vector {i}:")
            print(f"  Points shape: {gt_v.shape}")
            print(f"  First 3 points: {gt_v[:3]}")
            print(f"  X range: [{gt_v[:, 0].min():.2f}, {gt_v[:, 0].max():.2f}]")
            print(f"  Y range: [{gt_v[:, 1].min():.2f}, {gt_v[:, 1].max():.2f}]")

        for i in range(min(3, len(pred_vectors))):
            pred_v = pred_vectors[i]
            print(f"\nPred vector {i}:")
            print(f"  Points shape: {pred_v.shape}")
            print(f"  First 3 points: {pred_v[:3]}")
            print(f"  X range: [{pred_v[:, 0].min():.2f}, {pred_v[:, 0].max():.2f}]")
            print(f"  Y range: [{pred_v[:, 1].min():.2f}, {pred_v[:, 1].max():.2f}]")

            # Compute distance to nearest GT
            from scipy.spatial import distance as scipy_dist

            min_dist = float("inf")
            for gt_v in gt_vectors:
                # Resample to same length if needed
                if gt_v.shape[0] != pred_v.shape[0]:
                    from shapely.geometry import LineString

                    gt_line = LineString(gt_v)
                    distances = np.linspace(0, gt_line.length, pred_v.shape[0])
                    gt_v_resampled = np.array([list(gt_line.interpolate(d).coords)[0] for d in distances])
                else:
                    gt_v_resampled = gt_v

                dist_mat = scipy_dist.cdist(pred_v, gt_v_resampled, "euclidean")
                chamfer_dist = (dist_mat.min(-1).mean() + dist_mat.min(-2).mean()) / 2
                min_dist = min(min_dist, chamfer_dist)

            print(f"  Min chamfer dist to any GT: {min_dist:.2f}m")
            print(f"  Matches at 1.5m threshold: {min_dist < 1.5}")

        # Create overlay visualization
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 10))
            # Plot GT in solid lines
            for i, gt_v in enumerate(gt_vectors):
                plt.plot(gt_v[:, 0], gt_v[:, 1], "g-", linewidth=2, label=f"GT {i}" if i < 5 else None)
            # Plot predictions in dashed lines
            for i, pred_v in enumerate(pred_vectors[:10]):
                plt.plot(
                    pred_v[:, 0], pred_v[:, 1], "r--", linewidth=1, alpha=0.5, label=f"Pred {i}" if i < 5 else None
                )
            plt.xlim([-15, 15])
            plt.ylim([-30, 30])
            plt.grid(True)
            plt.legend()
            plt.title("GT (solid green) vs Predictions (dashed red)")
            plt.savefig("gt_vs_pred_overlay.png", dpi=150)
            plt.close()
            print("\nSaved overlay visualization to gt_vs_pred_overlay.png")
        except Exception as e:
            print(f"Could not create overlay visualization: {e}")

    matrix = custom_polyline_score(
        gen_lines[:, :-1].reshape(num_gens, -1, 2), gt_lines.reshape(num_gts, -1, 2), linewidth=2.0, metric=metric
    )
    # for each det, the max iou with all gts
    matrix_max = matrix.max(axis=1)
    # for each det, which gt overlaps most with it
    matrix_argmax = matrix.argmax(axis=1)

    # Debug: Log matching statistics
    if num_gens > 0 and num_gts > 0:
        best_match_score = matrix_max.max()
        worst_match_score = matrix_max.min()
        print(
            f"DEBUG tpfp: num_gens={num_gens}, num_gts={num_gts}, threshold={threshold}, "
            f"best_match={best_match_score:.4f}, worst_match={worst_match_score:.4f}, "
            f"matrix shape={matrix.shape}, matrix range=[{matrix.min():.4f}, {matrix.max():.4f}]"
        )
        if num_gens <= 5 and num_gts <= 5:
            print(f"DEBUG tpfp: Full matrix:\n{matrix}")
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-gen_scores)

    gt_covered = np.zeros(num_gts, dtype=bool)

    # tp = 0 and fp = 0 means ignore this detected bbox,
    for i in sort_inds:
        if matrix_max[i] >= threshold:
            matched_gt = matrix_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    return tp, fp
