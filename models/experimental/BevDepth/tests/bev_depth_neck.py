import torch
import torch.nn as nn


class SECONDFPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.deblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(160, 64, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(160, 64, kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(320, 64, kernel_size=4, stride=4, bias=False),
                    nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(640, 64, kernel_size=8, stride=8, bias=False),
                    nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        self.init_weights()

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    # --------------------------------------------------------
    # load_checkpoint() inside SECONDFPN
    # --------------------------------------------------------
    def load_checkpoint(self, ckpt_path: str, strict=True, map_location="cpu"):
        print(f"Loading checkpoint: {ckpt_path}")

        state = torch.load(ckpt_path, map_location=map_location)

        # if checkpoint contains {"state_dict": {...}}
        if "state_dict" in state:
            state = state["state_dict"]

        # -----------------------------------------------
        # Strip prefixes like "model.head.neck."
        # so keys match "deblocks.X.Y.weight"
        # -----------------------------------------------
        cleaned = {}
        for k, v in state.items():
            if "neck." in k:
                new_k = k.split("neck.", 1)[1]  # keep everything after neck.
                cleaned[new_k] = v

        missing, unexpected = self.load_state_dict(cleaned, strict=strict)

        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        print("✔ Checkpoint loaded successfully.")

    # --------------------------------------------------------
    def forward(self, x_list):
        outs = []
        for i, x in enumerate(x_list):
            outs.append(self.deblocks[i](x))
        return torch.cat(outs, dim=1)
