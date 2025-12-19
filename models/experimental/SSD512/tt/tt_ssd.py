from models.experimental.SSD512.tt.utils import create_config_layers, post_conv_reshape
from models.experimental.SSD512.tt.layers.tt_extras_backbone import TtExtrasBackbone
from models.experimental.SSD512.tt.layers.tt_vgg_backbone import TtVGGBackbone
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
)
from models.experimental.SSD512.tt.layers.tt_multibox_heads import TtMultiBoxHEAD
from models.experimental.SSD512.tt.layers.tt_l2norm import TtL2Norm
import ttnn

sources_shape = [
    (1, 512, 64, 64),
    (1, 1024, 32, 32),
    (1, 512, 16, 16),
    (1, 256, 8, 8),
    (1, 256, 4, 4),
    (1, 256, 2, 2),
    (1, 256, 1, 1),
]


class TtSSD:
    def __init__(self, torch_model, torch_input, device, batch_size: int):
        self.batch_size = batch_size
        self.device = device

        vgg_backbone_config_layers, vgg_torch_output = create_config_layers(
            torch_model=torch_model.base, torch_input=torch_input, return_out=True
        )

        self.tt_vgg_backbone = TtVGGBackbone(
            conv_config_layer=vgg_backbone_config_layers,
            batch_size=batch_size,
            device=device,
        )

        self.tt_l2norm = TtL2Norm(n_channels=512, scale=20.0, device=device)

        extra_config_layers, extra_torch_output = create_config_layers(
            torch_model.extras, torch_input=vgg_torch_output, return_out=True
        )
        self.tt_extras = TtExtrasBackbone(
            conv_config_layer=extra_config_layers,
            batch_size=batch_size,
            device=device,
        )

        torch_conf_model = torch_model.conf
        torch_loc_model = torch_model.loc

        self.loc_kernel_layers = []
        self.conf_kernel_layers = []
        for source_idx, source in enumerate(sources_shape):
            loc_config_layers = Conv2dConfiguration.from_torch(
                torch_loc_model[source_idx],
                input_height=source[-2],
                input_width=source[-1],
                batch_size=source[0],
            )

            conf_config_layers = Conv2dConfiguration.from_torch(
                torch_conf_model[source_idx],
                input_height=source[-2],
                input_width=source[-1],
                batch_size=source[0],
            )
            self.loc_kernel_layers.append(
                TtMultiBoxHEAD(
                    device=device,
                    conv_config_layer=loc_config_layers,
                )
            )
            self.conf_kernel_layers.append(
                TtMultiBoxHEAD(
                    device=device,
                    conv_config_layer=conf_config_layers,
                )
            )

    def __call__(self, device, input):
        tt_sources = []
        tt_loc_preds = []
        tt_conf_preds = []
        tt_vgg_out, vgg_sources = self.tt_vgg_backbone(device, input, return_source=True)

        input_tensor = post_conv_reshape(vgg_sources[0], out_height=64, out_width=64)
        l2norm_out = self.tt_l2norm(input_tensor)
        l2norm_out = ttnn.permute(l2norm_out, (0, 2, 3, 1))

        _, extra_sources = self.tt_extras(device, tt_vgg_out, return_source=True)

        tt_sources.append(l2norm_out)
        tt_sources.append(tt_vgg_out)
        tt_sources.extend(extra_sources)

        for source, loc_layer, conf_layer in zip(tt_sources, self.loc_kernel_layers, self.conf_kernel_layers):
            loc_pred = loc_layer(device, source)
            conf_pred = conf_layer(device, source)
            tt_loc_preds.append(loc_pred)
            tt_conf_preds.append(conf_pred)

        return tt_loc_preds, tt_conf_preds
