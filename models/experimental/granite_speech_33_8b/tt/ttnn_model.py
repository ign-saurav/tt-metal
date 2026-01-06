import ttnn
import torch
from models.experimental.granite_speech_33_8b.tt.ttnn_encoder_block import GraniteSpeechCTCEncoderTTNN
from models.experimental.granite_speech_33_8b.tt.ttnn_projector_block import GraniteSpeechEncoderProjectorTTNN

class GraniteEncoderAndProjector:
    """TTNN implementation of Encoder+Projector."""

    def __init__(self, device, config):
        self.device = device
        self.encoder = GraniteSpeechCTCEncoderTTNN(device, config, include_conformer_layernorm=False)
        self.projector = GraniteSpeechEncoderProjectorTTNN(device, config)

        self._setup_compute_config()

    def _setup_compute_config(self):
        """Setup compute kernel configuration for high accuracy."""
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,  # High fidelity for 0.99 PCC
            math_approx_mode=False,
            fp32_dest_acc_en=True,  # Enable FP32 accumulation for accuracy
            packer_l1_acc=False,
        )

    def prepare_weights(
        self, model
    ):
        """Load and convert PyTorch weights to TTNN format."""
        self.encoder.prepare_weights(model.encoder.state_dict())
        self.projector.prepare_weights(model.projector)

    def forward(self, input_features):
        """Get the audio features to merged into the multimodal embeddings."""
        encoder_embeds = self.encoder.forward(input_features)
        projected_embeds = self.projector.forward(encoder_embeds)

        return projected_embeds