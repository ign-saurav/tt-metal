import torch
import torch.nn as nn


class SECONDFPN(nn.Module):
    def __init__(self):
        super().__init__()

        # 4 deblocks exactly matching your structure
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

    def forward(self, x0, x1, x2, x3):
        # Each produces (B, 64, 128, 128)
        y0 = self.deblocks[0](x0)
        y1 = self.deblocks[1](x1)
        y2 = self.deblocks[2](x2)
        y3 = self.deblocks[3](x3)

        # Final concatenation: (B, 256, 128, 128)
        y = torch.cat([y0, y1, y2, y3], dim=1)
        return y


class BEVDepthHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.neck = SECONDFPN()

    def forward(self, x0, x1, x2, x3):
        return self.neck(x0, x1, x2, x3)

    def load_weights(self, path, strict=False):
        """
        Load PyTorch weights from a .pth file.

        Args:
            path (str): Path to the .pth checkpoint file.
            strict (bool): Whether to enforce strict matching.

        Returns:
            missing_keys, unexpected_keys
        """
        print(f"Loading checkpoint from {path} ...")
        ckpt = torch.load(path, map_location="cpu")

        # Case 1: checkpoint contains only state_dict
        if isinstance(ckpt, dict) and "state_dict" not in ckpt:
            state_dict = ckpt
            print("Detected pure state_dict format.")

        # Case 2: checkpoint contains state_dict
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            print("Detected checkpoint with state_dict entry.")

        else:
            raise ValueError("Unrecognized checkpoint format.")

        # Optional cleanup: remove prefixes like 'module.' or 'model.'
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k

            # remove 'module.' from DataParallel models
            if new_k.startswith("module."):
                new_k = new_k[len("module.") :]

            # adapt if weights are saved from model.head.neck.xxx
            if new_k.startswith("model."):
                new_k = new_k[len("model.") :]

            new_state_dict[new_k] = v

        missing, unexpected = self.load_state_dict(new_state_dict, strict=strict)

        print("Load complete.")
        if missing:
            print("\nMissing keys:")
            for k in missing:
                print("  -", k)
        if unexpected:
            print("\nUnexpected keys:")
            for k in unexpected:
                print("  -", k)

        return missing, unexpected
