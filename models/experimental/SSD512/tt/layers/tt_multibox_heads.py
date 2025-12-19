from models.experimental.SSD512.tt.utils import Conv2dNormActivation


class TtMultiBoxHEAD:
    def __init__(self, conv_config_layer, device, activation_layer=None):
        # self.batch_size = batch_size
        self.device = device

        # layers = []
        self.layer = Conv2dNormActivation(
            device=device,
            conv_config=conv_config_layer,
            activation_layer=activation_layer,
        )

    #    return layers

    def __call__(self, device, input):
        result = self.layer(device, input)

        return result
