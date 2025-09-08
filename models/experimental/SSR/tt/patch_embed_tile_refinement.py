import ttnn
from models.common.lightweightmodule import LightweightModule


class TTPatchEmbed(LightweightModule):
    """TTNN Image to Patch Embedding (simplified version)

    Args:
        img_size (int): Image size. Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer: Normalization layer. Default: None
        device: TTNN device
        parameters: Preprocessed parameters dictionary
        memory_config: TTNN memory configuration
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        device=None,
        parameters=None,
        memory_config=None,
    ):
        super().__init__()

        # Convert to tuples (assuming square images/patches)
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.device = device
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

        # Store normalization parameters if provided
        if norm_layer is not None and parameters is not None:
            self.norm_weight = parameters.get("norm", {}).get("weight")
            self.norm_bias = parameters.get("norm", {}).get("bias")
        else:
            self.norm_weight = None
            self.norm_bias = None

    def forward(self, x):
        """
        Forward pass through patch embedding

        Args:
            x: Input tensor of shape [batch, channels, height, width]

        Returns:
            Output tensor of shape [batch, num_patches, embed_dim]
        """

        if x.is_sharded():
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # Apply normalization if available
        if self.norm_weight is not None:
            x = ttnn.permute(x, [0, 2, 3, 1])
            x = ttnn.layer_norm(x, weight=self.norm_weight, bias=self.norm_bias, memory_config=self.memory_config)

        return x
