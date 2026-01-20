# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# Based on PointPillars implementation from https://github.com/zhulf0804/PointPillars
# Original implementation by zhulf0804 under MIT license

import torch


def boxes_overlap_bev(boxes_a, boxes_b):
    """
    Calculate boxes overlap in bird's eye view (PyTorch version)

    Args:
        boxes_a: (M, 5) [x1, y1, x2, y2, ry]
        boxes_b: (N, 5) [x1, y1, x2, y2, ry]

    Returns:
        overlap: (M, N) overlap area between boxes
    """
    M, N = boxes_a.shape[0], boxes_b.shape[0]
    overlap = torch.zeros((M, N), dtype=boxes_a.dtype, device=boxes_a.device)

    for i in range(M):
        for j in range(N):
            overlap[i, j] = box_overlap_single(boxes_a[i], boxes_b[j])

    return overlap


def box_overlap_single(box_a, box_b):
    """Calculate overlap between two rotated boxes in BEV"""
    # Get corners of both boxes
    corners_a = get_box_corners_2d(box_a)
    corners_b = get_box_corners_2d(box_b)

    # Use Sutherland-Hodgman algorithm for polygon intersection
    intersection_points = sutherland_hodgman(corners_a, corners_b)

    if len(intersection_points) < 3:
        return 0.0

    # Calculate area of intersection polygon
    return polygon_area(intersection_points)


def get_box_corners_2d(box):
    """Get 4 corners of a rotated box in BEV
    box: [x1, y1, x2, y2, ry]
    """
    x1, y1, x2, y2, ry = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1

    # Corners in local frame
    corners = torch.tensor(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], device=box.device, dtype=box.dtype
    )

    # Rotation matrix
    cos_ry = torch.cos(ry)
    sin_ry = torch.sin(ry)
    rot_mat = torch.tensor([[cos_ry, -sin_ry], [sin_ry, cos_ry]], device=box.device, dtype=box.dtype)

    # Rotate and translate
    corners = corners @ rot_mat.T
    corners[:, 0] += cx
    corners[:, 1] += cy

    return corners


def sutherland_hodgman(subject, clip):
    """Sutherland-Hodgman polygon clipping algorithm for 2D polygons"""

    def inside(p, edge_p1, edge_p2):
        # Check if point p is on the left side of the edge
        return (edge_p2[0] - edge_p1[0]) * (p[1] - edge_p1[1]) >= (edge_p2[1] - edge_p1[1]) * (p[0] - edge_p1[0])

    def line_intersection(p1, p2, edge_p1, edge_p2):
        """Find intersection point of two line segments in 2D"""
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = edge_p1[0], edge_p1[1]
        x4, y4 = edge_p2[0], edge_p2[1]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if torch.abs(denom) < 1e-10:
            return p1  # Lines are parallel

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return torch.stack([x, y])

    output = subject.clone()

    for i in range(len(clip)):
        if len(output) == 0:
            break

        edge_p1 = clip[i]
        edge_p2 = clip[(i + 1) % len(clip)]

        input_list = output
        output = []

        for j in range(len(input_list)):
            current = input_list[j]
            previous = input_list[j - 1]

            if inside(current, edge_p1, edge_p2):
                if not inside(previous, edge_p1, edge_p2):
                    output.append(line_intersection(previous, current, edge_p1, edge_p2))
                output.append(current)
            elif inside(previous, edge_p1, edge_p2):
                output.append(line_intersection(previous, current, edge_p1, edge_p2))

        if len(output) > 0:
            output = torch.stack(output)

    return output if isinstance(output, torch.Tensor) else torch.tensor([], device=subject.device)


def polygon_area(vertices):
    """Calculate area of polygon using shoelace formula"""
    if len(vertices) < 3:
        return 0.0

    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * torch.abs(torch.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])


def boxes_iou_bev(boxes_a, boxes_b):
    """
    Calculate boxes IoU in bird's eye view (PyTorch version)

    Args:
        boxes_a: (M, 5) [x1, y1, x2, y2, ry]
        boxes_b: (N, 5) [x1, y1, x2, y2, ry]

    Returns:
        iou: (M, N) IoU between boxes
    """
    M, N = boxes_a.shape[0], boxes_b.shape[0]
    iou = torch.zeros((M, N), dtype=boxes_a.dtype, device=boxes_a.device)

    for i in range(M):
        for j in range(N):
            box_a = boxes_a[i]
            box_b = boxes_b[j]

            # Calculate areas
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

            # Calculate overlap
            overlap = box_overlap_single(box_a, box_b)

            # Calculate IoU
            union = area_a + area_b - overlap
            iou[i, j] = overlap / torch.clamp(union, min=1e-8)

    return iou


def nms_cuda(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """
    NMS for rotated boxes (PyTorch version)

    Args:
        boxes: (N, 5) [x1, y1, x2, y2, ry]
        scores: (N,) confidence scores
        thresh: IoU threshold
        pre_maxsize: max boxes before NMS
        post_max_size: max boxes after NMS

    Returns:
        keep: indices of kept boxes
    """
    # Sort by scores
    order = torch.argsort(scores, descending=True)

    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order]
    scores = scores[order]

    keep = []

    while len(boxes) > 0:
        # Keep box with highest score
        keep.append(order[0].item())

        if len(boxes) == 1:
            break

        # Calculate IoU with remaining boxes
        ious = boxes_iou_bev(boxes[0:1], boxes[1:])

        # Keep boxes with IoU below threshold
        mask = ious[0] <= thresh
        boxes = boxes[1:][mask]
        order = order[1:][mask]

    keep = torch.tensor(keep, dtype=torch.long, device=scores.device)

    if post_max_size is not None:
        keep = keep[:post_max_size]

    return keep


def nms_normal_pytorch(boxes, scores, thresh):
    """
    Standard NMS for axis-aligned boxes (PyTorch version)

    Args:
        boxes: (N, 5) [x1, y1, x2, y2, ry] (ry ignored for axis-aligned)
        scores: (N,) confidence scores
        thresh: IoU threshold

    Returns:
        keep: indices of kept boxes
    """
    order = torch.argsort(scores, descending=True)

    boxes = boxes[order]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    keep = []

    while len(boxes) > 0:
        keep.append(order[0].item())

        if len(boxes) == 1:
            break

        # Calculate IoU with remaining boxes
        xx1 = torch.maximum(x1[0], x1[1:])
        yy1 = torch.maximum(y1[0], y1[1:])
        xx2 = torch.minimum(x2[0], x2[1:])
        yy2 = torch.minimum(y2[0], y2[1:])

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)

        overlap = w * h
        iou = overlap / (areas[0] + areas[1:] - overlap)

        # Keep boxes with IoU below threshold
        mask = iou <= thresh
        boxes = boxes[1:][mask]
        x1 = x1[1:][mask]
        y1 = y1[1:][mask]
        x2 = x2[1:][mask]
        y2 = y2[1:][mask]
        areas = areas[1:][mask]
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=scores.device)
