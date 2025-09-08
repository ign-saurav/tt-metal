import ttnn
from models.common.lightweightmodule import LightweightModule


class TTPatchUnEmbed(LightweightModule):
    """Image to Patch Unembedding in TTNN"""

    def __init__(self, mesh_device, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()

        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        batch_size = x.shape[0]

        # Transpose from (B, N, C) to (B, C, N) equivalent
        x = ttnn.permute(x, (0, 2, 1))  # (batch_size, embed_dim, num_patches)

        # Reshape to spatial dimensions
        x = ttnn.reshape(x, (batch_size, self.embed_dim, x_size[0], x_size[1]), memory_config=ttnn.L1_MEMORY_CONFIG)

        return x
