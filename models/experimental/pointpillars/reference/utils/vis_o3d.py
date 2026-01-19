# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# Based on PointPillars implementation from https://github.com/zhulf0804/PointPillars
# Original implementation by zhulf0804 under MIT license


import cv2

COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
COLORS_IMG = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]]

LINES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [2, 6], [7, 3], [1, 5], [4, 0]]


def vis_img_3d(img, image_points, labels, rt=True):
    """
    img: (h, w, 3)
    image_points: (n, 8, 2)
    labels: (n, )
    """

    for i in range(len(image_points)):
        label = labels[i]
        bbox_points = image_points[i]  # (8, 2)
        if label >= 0 and label < 3:
            color = COLORS_IMG[label]
        else:
            color = COLORS_IMG[-1]
        for line_id in LINES:
            x1, y1 = bbox_points[line_id[0]]
            x2, y2 = bbox_points[line_id[1]]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(img, (x1, y1), (x2, y2), color, 1)
    if rt:
        return img
    cv2.imshow("bbox", img)
    cv2.waitKey(0)
