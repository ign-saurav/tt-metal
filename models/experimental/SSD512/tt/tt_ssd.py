from models.experimental.SSD512.tt.utils import create_config_layers, post_conv_reshape
from models.experimental.SSD512.tt.layers.tt_extras_backbone import TtExtrasBackbone
from models.experimental.SSD512.tt.layers.tt_vgg_backbone import TtVGGBackbone
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
)
from models.experimental.SSD512.tt.layers.tt_multibox_heads import TtMultiBoxHEAD
from models.experimental.SSD512.tt.layers.tt_l2norm import TtL2Norm
import ttnn


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

        # TTNN L2Norm
        self.tt_l2norm = TtL2Norm(n_channels=512, scale=20.0, device=device)

        extra_config_layers, extra_torch_output = create_config_layers(
            torch_model.extras, torch_input=vgg_torch_output, return_out=True
        )
        self.tt_extras = TtExtrasBackbone(
            conv_config_layer=extra_config_layers,
            batch_size=batch_size,
            device=device,
        )
        # print(tt_extras)

        torch_conf_model = torch_model.conf
        torch_loc_model = torch_model.loc

        ##########################################333
        sources = [
            (1, 512, 64, 64),
            (1, 1024, 32, 32),
            (1, 512, 16, 16),
            (1, 256, 8, 8),
            (1, 256, 4, 4),
            (1, 256, 2, 2),
            (1, 256, 1, 1),
        ]
        self.loc_kernel_layers = []
        self.conf_kernel_layers = []
        for source_idx, source in enumerate(sources):
            # if isinstance(torch_loc_model[source_idx], nn.Conv2d):
            loc_config_layers = Conv2dConfiguration.from_torch(
                torch_loc_model[source_idx],
                input_height=source[-2],
                input_width=source[-1],
                batch_size=source[0],
                # **model_config,
            )

            conf_config_layers = Conv2dConfiguration.from_torch(
                torch_conf_model[source_idx],
                input_height=source[-2],
                input_width=source[-1],
                batch_size=source[0],
                # **model_config,
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
        # loc_kernel_layers.appen
        ##################################################3

    def __call__(self, device, input):
        tt_sources = []
        tt_loc_preds = []
        tt_conf_preds = []
        tt_vgg_out, vgg_sources = self.tt_vgg_backbone(device, input, return_source=True)

        input_tensor = post_conv_reshape(vgg_sources[0], out_height=64, out_width=64)
        #########################
        l2norm_out = self.tt_l2norm(input_tensor)
        l2norm_out = ttnn.permute(l2norm_out, (0, 2, 3, 1))

        tt_extras_out, extra_sources = self.tt_extras(device, tt_vgg_out, return_source=True)
        print(tt_extras_out)

        tt_sources.append(l2norm_out)
        # tt_sources.append(vgg_sources[0])
        tt_sources.append(tt_vgg_out)
        tt_sources.extend(extra_sources)

        for source, loc_layer, conf_layer in zip(tt_sources, self.loc_kernel_layers, self.conf_kernel_layers):
            print(loc_layer)
            loc_pred = loc_layer(device, source)
            conf_pred = conf_layer(device, source)
            tt_loc_preds.append(loc_pred)
            tt_conf_preds.append(conf_pred)

        return tt_loc_preds, tt_conf_preds

        # for i, layer in enumerate(self.block):
        #     if i == 0:
        #         result = layer(device, input)
        #     else:
        #         result = layer(device, result)

        # return result


#  # @staticmethod
#     def forward(self, x):
#         """Applies network layers and ops on input image(s) x.

#         Args:
#             x: input image or batch of images. Shape: [batch,3,300,300].

#         Return:
#             Depending on phase:
#             test:
#                 Variable(tensor) of output class label predictions,
#                 confidence score, and corresponding location predictions for
#                 each object detected. Shape: [batch,topk,7]

#             train:
#                 list of concat outputs from:
#                     1: confidence layers, Shape: [batch*num_priors,num_classes]
#                     2: localization layers, Shape: [batch,num_priors*4]
#                     3: priorbox layers, Shape: [2,num_priors*4]
#         """
#         sources = list()
#         loc = list()
#         conf = list()

#         # apply vgg up to conv4_3 relu
#         for k in range(23):
#             x = self.base[k](x)

#         s = self.L2Norm(x)
#         sources.append(s)

#         # apply vgg up to fc7
#         for k in range(23, len(self.base)):
#             x = self.base[k](x)
#         sources.append(x)

#         # apply extra layers and cache source layer outputs
#         for k, v in enumerate(self.extras):
#             x = F.relu(v(x), inplace=True)
#             if k % 2 == 1:
#                 sources.append(x)

#         # apply multibox head to source layers
#         for x, l, c in zip(sources, self.loc, self.conf):
#             loc.append(l(x).permute(0, 2, 3, 1).contiguous())
#             conf.append(c(x).permute(0, 2, 3, 1).contiguous())

#         loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
#         conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
#         if self.phase == "test":
#             output = self.detect(
#                 loc.view(loc.size(0), -1, 4),  # loc preds
#                 self.softmax(conf.view(conf.size(0), -1, self.num_classes)),  # conf preds
#                 self.priors.type(type(x.data)),  # default boxes
#             )
#         else:
#             output = (loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes), self.priors)
#         return output
