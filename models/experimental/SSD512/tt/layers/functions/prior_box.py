# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn


# Generates prior boxes (anchor boxes) for SSD512 at multiple feature map scales
class TtPriorBox:
    def __init__(self, cfg, device=None):
        self.image_size = cfg["min_dim"]
        self.num_priors = len(cfg["aspect_ratios"])
        self.variance = cfg["variance"] or [0.1]
        self.feature_maps = cfg["feature_maps"]
        self.min_sizes = cfg["min_sizes"]
        self.max_sizes = cfg["max_sizes"]
        self.steps = cfg["steps"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.clip = cfg["clip"]
        self.device = device

        for v in self.variance:
            if v <= 0:
                raise ValueError("Variances must be greater than 0")

    # Generates prior boxes
    def __call__(self):
        all_boxes = []

        # Generate priors for each feature map scale
        for k, f in enumerate(self.feature_maps):
            f_k = self.image_size / self.steps[k]

            # Create coordinate grids for this feature map
            y_coords = ttnn.arange(0, f, device=self.device, dtype=ttnn.float32)
            x_coords = ttnn.arange(0, f, device=self.device, dtype=ttnn.float32)

            yy = ttnn.repeat(ttnn.reshape(y_coords, (f, 1)), (1, f))
            xx = ttnn.repeat(ttnn.reshape(x_coords, (1, f)), (f, 1))

            yy_flat = ttnn.reshape(yy, (-1,))
            xx_flat = ttnn.reshape(xx, (-1,))

            # Compute center coordinates normalized to [0, 1]
            cy = (yy_flat + 0.5) / f_k
            cx = (xx_flat + 0.5) / f_k

            # Compute scale factors for this feature map
            s_k = self.min_sizes[k] / self.image_size
            s_k_prime = ttnn.sqrt(
                ttnn.full((1,), s_k * (self.max_sizes[k] / self.image_size), device=self.device, dtype=ttnn.float32)
            )

            num_positions = f * f

            s_k_tensor = ttnn.full((num_positions,), s_k, device=self.device, dtype=ttnn.float32)
            box1 = ttnn.stack([cx, cy, s_k_tensor, s_k_tensor], dim=1)

            s_k_prime_expanded = ttnn.repeat(s_k_prime, (num_positions,))
            box2 = ttnn.stack([cx, cy, s_k_prime_expanded, s_k_prime_expanded], dim=1)

            boxes_for_map = [box1, box2]

            for ar in self.aspect_ratios[k]:
                ar_sqrt = ttnn.sqrt(ttnn.full((1,), ar, device=self.device, dtype=ttnn.float32))[0]

                w1 = s_k * ar_sqrt
                h1 = s_k / ar_sqrt
                w1_tensor = ttnn.full((num_positions,), w1, device=self.device, dtype=ttnn.float32)
                h1_tensor = ttnn.full((num_positions,), h1, device=self.device, dtype=ttnn.float32)
                box_ar1 = ttnn.stack([cx, cy, w1_tensor, h1_tensor], dim=1)

                w2 = s_k / ar_sqrt
                h2 = s_k * ar_sqrt
                w2_tensor = ttnn.full((num_positions,), w2, device=self.device, dtype=ttnn.float32)
                h2_tensor = ttnn.full((num_positions,), h2, device=self.device, dtype=ttnn.float32)
                box_ar2 = ttnn.stack([cx, cy, w2_tensor, h2_tensor], dim=1)

                boxes_for_map.extend([box_ar1, box_ar2])

            feature_boxes = ttnn.concat(boxes_for_map, dim=0)
            all_boxes.append(feature_boxes)

        output = ttnn.concat(all_boxes, dim=0)

        if self.clip:
            output = ttnn.clip(output, 0.0, 1.0)

        return output
