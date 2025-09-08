import ttnn
from models.common.lightweightmodule import LightweightModule


class TTPatchUnEmbed(LightweightModule):
    """Image to Patch Unembedding in TTNN"""

    def __init__(self, mesh_device, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()

        self.mesh_device = mesh_device

        # Convert to tuples like in the original
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)

        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        batch_size = x.shape[0]

        # Transpose from (B, N, C) to (B, C, N) equivalent
        x = ttnn.permute(x, (0, 2, 1))  # (batch_size, embed_dim, num_patches)

        # Reshape to spatial dimensions
        x = ttnn.reshape(x, (batch_size, self.embed_dim, x_size[0], x_size[1]))

        return x
