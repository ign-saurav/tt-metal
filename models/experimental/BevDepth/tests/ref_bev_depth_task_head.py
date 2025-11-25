import torch
import torch.nn as nn


# ------------------------------------------------------------------------
# ConvModule
# ------------------------------------------------------------------------
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ------------------------------------------------------------------------
# SeparateHead
# ------------------------------------------------------------------------
class SeparateHead(nn.Module):
    def __init__(self, in_channels, heatmap_out):
        super().__init__()

        def branch(out_c):
            return nn.Sequential(
                ConvModule(in_channels, in_channels),
                nn.Conv2d(in_channels, out_c, kernel_size=3, padding=1),
            )

        self.reg = branch(2)
        self.height = branch(1)
        self.dim = branch(3)
        self.rot = branch(2)
        self.vel = branch(2)
        self.heatmap = branch(heatmap_out)

    def forward(self, x):
        return {
            "reg": self.reg(x),
            "height": self.height(x),
            "dim": self.dim(x),
            "rot": self.rot(x),
            "vel": self.vel(x),
            "heatmap": self.heatmap(x),
        }


# ------------------------------------------------------------------------
# Main Model with load_checkpoint()
# ------------------------------------------------------------------------
class BEVDepthHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv = ConvModule(256, 64)

        heatmap_channels = [1, 2, 2, 1, 2, 2]

        self.task_heads = nn.ModuleList([SeparateHead(64, heatmap_out=heatmap_channels[i]) for i in range(6)])

    # ------------------------------------------------------------
    #                LOAD WEIGHTS FUNCTION
    # ------------------------------------------------------------
    def load_checkpoint(self, weight_path):
        print(f"Loading weights from: {weight_path}")

        ckpt = torch.load(weight_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        prefix = "model.head."
        new_state = {}
        skipped = {}

        for k, v in state.items():
            if not k.startswith(prefix):
                continue

            new_k = k[len(prefix) :]

            # key not in current model → skip
            if new_k not in self.state_dict():
                skipped[new_k] = ("not in model", v.shape)
                continue

            # shape mismatch → skip
            if self.state_dict()[new_k].shape != v.shape:
                skipped[new_k] = (self.state_dict()[new_k].shape, v.shape)
                continue

            new_state[new_k] = v

        # Load only valid keys
        missing, unexpected = self.load_state_dict(new_state, strict=False)

        print("\n=== Weight Loading Report ===")
        print("Loaded keys:", len(new_state))
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        if skipped:
            print("\n=== Skipped keys due to mismatch ===")
            for k, (model_shape, ckpt_shape) in skipped.items():
                print(f"{k} | model: {model_shape} vs ckpt: {ckpt_shape}")

        print("\nDone.\n")

    # ------------------------------------------------------------
    def forward(self, x):
        f = self.shared_conv(x)
        return [head(f) for head in self.task_heads]
