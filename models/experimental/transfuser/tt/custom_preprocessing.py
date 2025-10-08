import ttnn
import torch

from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

from models.experimental.transfuser.reference.transfuser_backbone import TransfuserBackbone
from models.experimental.transfuser.reference.bottleneck import Bottleneck
from models.experimental.transfuser.reference.stage import Stage
from models.experimental.transfuser.reference.common import Conv2d


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None, device=None
):
    parameters = {}
    if isinstance(model, Conv2d):
        if model.norm is not None:
            weight, bias = fold_batch_norm2d_into_conv2d(model, model.norm)
        else:
            weight = model.weight.clone().detach().contiguous()
            bias = (
                model.bias.clone().detach().contiguous() if model.bias is not None else torch.zeros(model.out_channels)
            )
        parameters["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
        parameters["bias"] = ttnn.from_torch(torch.reshape(bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
    elif isinstance(model, TransfuserBackbone):
        # Image encoder conv1
        if hasattr(model, "image_encoder") and hasattr(model.image_encoder, "features"):
            weight, bias = fold_batch_norm2d_into_conv2d(
                model.image_encoder.features.conv1, model.image_encoder.features.bn1
            )
            parameters["image_encoder"] = {}
            parameters["image_encoder"]["features"] = {}
            parameters["image_encoder"]["features"]["conv1"] = {}
            parameters["image_encoder"]["features"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            # Lidar encoder conv1
            if hasattr(model, "lidar_encoder") and hasattr(model.lidar_encoder, "_model"):
                lidar_weight, lidar_bias = fold_batch_norm2d_into_conv2d(
                    model.lidar_encoder._model.conv1, model.lidar_encoder._model.bn1
                )
                parameters["lidar_encoder"] = {}
                parameters["lidar_encoder"]["_model"] = {}
                parameters["lidar_encoder"]["_model"]["conv1"] = {}
                parameters["lidar_encoder"]["_model"]["conv1"]["weight"] = ttnn.from_torch(
                    lidar_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                )
                lidar_bias = lidar_bias.reshape((1, 1, 1, -1))
                parameters["lidar_encoder"]["_model"]["conv1"]["bias"] = ttnn.from_torch(
                    lidar_bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                )

        # layer1 preprocessing for image encoder
        if hasattr(model.image_encoder.features, "layer1"):
            parameters["image_encoder"]["features"]["layer1"] = {}

            # 1st bottleneck
            b1_block = model.image_encoder.features.layer1.b1
            parameters["image_encoder"]["features"]["layer1"]["b1"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv1.conv, b1_block.conv1.bn)
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv1"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv2.conv, b1_block.conv2.bn)
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv2"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["image_encoder"]["features"]["layer1"]["b1"]["se"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b1"]["se"]["fc1"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b1"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b1_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["image_encoder"]["features"]["layer1"]["b1"]["se"]["fc2"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b1"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b1_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv3.conv, b1_block.conv3.bn)
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv3"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # Downsample
            if hasattr(b1_block, "downsample") and b1_block.downsample is not None:
                weight, bias = fold_batch_norm2d_into_conv2d(b1_block.downsample.conv, b1_block.downsample.bn)
                parameters["image_encoder"]["features"]["layer1"]["b1"]["downsample"] = {}
                parameters["image_encoder"]["features"]["layer1"]["b1"]["downsample"]["weight"] = ttnn.from_torch(
                    weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                )
                bias = bias.reshape((1, 1, 1, -1))
                parameters["image_encoder"]["features"]["layer1"]["b1"]["downsample"]["bias"] = ttnn.from_torch(
                    bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                )

            # 2nd bottleneck (no downsample)
            b2_block = model.image_encoder.features.layer1.b2
            parameters["image_encoder"]["features"]["layer1"]["b2"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv1.conv, b2_block.conv1.bn)
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv1"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv2.conv, b2_block.conv2.bn)
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv2"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv3.conv, b2_block.conv3.bn)
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv3"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["image_encoder"]["features"]["layer1"]["b2"]["se"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b2"]["se"]["fc1"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b2"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b2_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["image_encoder"]["features"]["layer1"]["b2"]["se"]["fc2"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b2"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b2_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
        # layer1 preprocessing for lidar encoder
        if hasattr(model.lidar_encoder._model, "layer1"):
            parameters["lidar_encoder"]["_model"]["layer1"] = {}

            # 1st bottleneck
            b1_block = model.lidar_encoder._model.layer1.b1
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv1.conv, b1_block.conv1.bn)
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv1"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv2.conv, b1_block.conv2.bn)
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv2"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv3.conv, b1_block.conv3.bn)
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv3"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["se"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["se"]["fc1"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b1_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["se"]["fc2"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b1_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # Downsample
            if hasattr(b1_block, "downsample") and b1_block.downsample is not None:
                if not isinstance(b1_block.downsample, torch.nn.Identity):
                    weight, bias = fold_batch_norm2d_into_conv2d(b1_block.downsample.conv, b1_block.downsample.bn)
                    parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["downsample"] = {}
                    parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["downsample"]["weight"] = ttnn.from_torch(
                        weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                    )
                    bias = bias.reshape((1, 1, 1, -1))
                    parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["downsample"]["bias"] = ttnn.from_torch(
                        bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                    )

            # 2nd bottleneck for lidar
            b2_block = model.lidar_encoder._model.layer1.b2
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv1.conv, b2_block.conv1.bn)
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv1"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv2.conv, b2_block.conv2.bn)
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv2"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv3.conv, b2_block.conv3.bn)
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv3"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["se"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["se"]["fc1"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b2_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["se"]["fc2"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b2_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

        # layer2 preprocessing for image encoder
        if hasattr(model.image_encoder.features, "layer2"):
            parameters["image_encoder"]["features"]["layer2"] = {}

            # 1st bottleneck
            b1_block = model.image_encoder.features.layer2.b1
            parameters["image_encoder"]["features"]["layer2"]["b1"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv1.conv, b1_block.conv1.bn)
            parameters["image_encoder"]["features"]["layer2"]["b1"]["conv1"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b1"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b1"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv2.conv, b1_block.conv2.bn)
            parameters["image_encoder"]["features"]["layer2"]["b1"]["conv2"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b1"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b1"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["image_encoder"]["features"]["layer2"]["b1"]["se"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b1"]["se"]["fc1"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b1"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b1_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["image_encoder"]["features"]["layer2"]["b1"]["se"]["fc2"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b1"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b1_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv3.conv, b1_block.conv3.bn)
            parameters["image_encoder"]["features"]["layer2"]["b1"]["conv3"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b1"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b1"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # Downsample
            if hasattr(b1_block, "downsample") and b1_block.downsample is not None:
                weight, bias = fold_batch_norm2d_into_conv2d(b1_block.downsample.conv, b1_block.downsample.bn)
                parameters["image_encoder"]["features"]["layer2"]["b1"]["downsample"] = {}
                parameters["image_encoder"]["features"]["layer2"]["b1"]["downsample"]["weight"] = ttnn.from_torch(
                    weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                )
                bias = bias.reshape((1, 1, 1, -1))
                parameters["image_encoder"]["features"]["layer2"]["b1"]["downsample"]["bias"] = ttnn.from_torch(
                    bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                )

            # 2nd bottleneck (no downsample)
            b2_block = model.image_encoder.features.layer2.b2
            parameters["image_encoder"]["features"]["layer2"]["b2"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv1.conv, b2_block.conv1.bn)
            parameters["image_encoder"]["features"]["layer2"]["b2"]["conv1"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b2"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b2"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv2.conv, b2_block.conv2.bn)
            parameters["image_encoder"]["features"]["layer2"]["b2"]["conv2"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b2"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b2"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv3.conv, b2_block.conv3.bn)
            parameters["image_encoder"]["features"]["layer2"]["b2"]["conv3"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b2"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b2"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["image_encoder"]["features"]["layer2"]["b2"]["se"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b2"]["se"]["fc1"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b2"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b2_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["image_encoder"]["features"]["layer2"]["b2"]["se"]["fc2"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b2"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b2_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # 3rd bottleneck (no downsample)
            b3_block = model.image_encoder.features.layer2.b3
            parameters["image_encoder"]["features"]["layer2"]["b3"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b3_block.conv1.conv, b3_block.conv1.bn)
            parameters["image_encoder"]["features"]["layer2"]["b3"]["conv1"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b3"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b3"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b3_block.conv2.conv, b3_block.conv2.bn)
            parameters["image_encoder"]["features"]["layer2"]["b3"]["conv2"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b3"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b3"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b3_block.conv3.conv, b3_block.conv3.bn)
            parameters["image_encoder"]["features"]["layer2"]["b3"]["conv3"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b3"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b3"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["image_encoder"]["features"]["layer2"]["b3"]["se"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b3"]["se"]["fc1"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b3"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b3_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["image_encoder"]["features"]["layer2"]["b3"]["se"]["fc2"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b3"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b3_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # 4th bottleneck (no downsample)
            b4_block = model.image_encoder.features.layer2.b4
            parameters["image_encoder"]["features"]["layer2"]["b4"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b4_block.conv1.conv, b4_block.conv1.bn)
            parameters["image_encoder"]["features"]["layer2"]["b4"]["conv1"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b4"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b4"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b4_block.conv2.conv, b4_block.conv2.bn)
            parameters["image_encoder"]["features"]["layer2"]["b4"]["conv2"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b4"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b4"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b4_block.conv3.conv, b4_block.conv3.bn)
            parameters["image_encoder"]["features"]["layer2"]["b4"]["conv3"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b4"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b4"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["image_encoder"]["features"]["layer2"]["b4"]["se"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b4"]["se"]["fc1"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b4"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b4_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["image_encoder"]["features"]["layer2"]["b4"]["se"]["fc2"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b4"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b4_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # 5th bottleneck (no downsample)
            b5_block = model.image_encoder.features.layer2.b5
            parameters["image_encoder"]["features"]["layer2"]["b5"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b5_block.conv1.conv, b5_block.conv1.bn)
            parameters["image_encoder"]["features"]["layer2"]["b5"]["conv1"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b5"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b5"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b5_block.conv2.conv, b5_block.conv2.bn)
            parameters["image_encoder"]["features"]["layer2"]["b5"]["conv2"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b5"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b5"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b5_block.conv3.conv, b5_block.conv3.bn)
            parameters["image_encoder"]["features"]["layer2"]["b5"]["conv3"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b5"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer2"]["b5"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["image_encoder"]["features"]["layer2"]["b5"]["se"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b5"]["se"]["fc1"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b5"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b5_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["image_encoder"]["features"]["layer2"]["b5"]["se"]["fc2"] = {}
            parameters["image_encoder"]["features"]["layer2"]["b5"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b5_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

        # layer2 preprocessing for lidar encoder
        if hasattr(model.lidar_encoder._model, "layer2"):
            parameters["lidar_encoder"]["_model"]["layer2"] = {}

            # 1st bottleneck
            b1_block = model.lidar_encoder._model.layer2.b1
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv1.conv, b1_block.conv1.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["conv1"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv2.conv, b1_block.conv2.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["conv2"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv3.conv, b1_block.conv3.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["conv3"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["se"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["se"]["fc1"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b1_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["se"]["fc2"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b1_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # Downsample
            if hasattr(b1_block, "downsample") and b1_block.downsample is not None:
                if not isinstance(b1_block.downsample, torch.nn.Identity):
                    weight, bias = fold_batch_norm2d_into_conv2d(b1_block.downsample.conv, b1_block.downsample.bn)
                    parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["downsample"] = {}
                    parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["downsample"]["weight"] = ttnn.from_torch(
                        weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                    )
                    bias = bias.reshape((1, 1, 1, -1))
                    parameters["lidar_encoder"]["_model"]["layer2"]["b1"]["downsample"]["bias"] = ttnn.from_torch(
                        bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                    )

            # 2nd bottleneck for lidar
            b2_block = model.lidar_encoder._model.layer2.b2
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv1.conv, b2_block.conv1.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"]["conv1"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv2.conv, b2_block.conv2.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"]["conv2"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv3.conv, b2_block.conv3.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"]["conv3"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"]["se"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"]["se"]["fc1"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b2_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"]["se"]["fc2"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b2"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b2_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # 3rd bottleneck for lidar
            b3_block = model.lidar_encoder._model.layer2.b3
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b3_block.conv1.conv, b3_block.conv1.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"]["conv1"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b3_block.conv2.conv, b3_block.conv2.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"]["conv2"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b3_block.conv3.conv, b3_block.conv3.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"]["conv3"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"]["se"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"]["se"]["fc1"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b3_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"]["se"]["fc2"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b3"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b3_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # 4th bottleneck for lidar
            b4_block = model.lidar_encoder._model.layer2.b4
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b4_block.conv1.conv, b4_block.conv1.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"]["conv1"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b4_block.conv2.conv, b4_block.conv2.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"]["conv2"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b4_block.conv3.conv, b4_block.conv3.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"]["conv3"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"]["se"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"]["se"]["fc1"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b4_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"]["se"]["fc2"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b4"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b4_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # 5th bottleneck for lidar
            b5_block = model.lidar_encoder._model.layer2.b5
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b5_block.conv1.conv, b5_block.conv1.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"]["conv1"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b5_block.conv2.conv, b5_block.conv2.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"]["conv2"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b5_block.conv3.conv, b5_block.conv3.bn)
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"]["conv3"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"]["se"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"]["se"]["fc1"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b5_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"]["se"]["fc2"] = {}
            parameters["lidar_encoder"]["_model"]["layer2"]["b5"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b5_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
        # Add transformer1 preprocessing
        if hasattr(model, "transformer1"):
            parameters["transformer1"] = {}

            if hasattr(model.transformer1, "ln_f"):
                # )
                parameters["transformer1"]["ln_f_weight"] = ttnn.from_torch(
                    model.transformer1.ln_f.weight,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=mesh_mapper,
                )
                parameters["transformer1"]["ln_f_bias"] = ttnn.from_torch(
                    model.transformer1.ln_f.bias.reshape((1, -1)),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=mesh_mapper,
                )

            if hasattr(model.transformer1, "pos_emb"):
                parameters["transformer1"]["pos_emb"] = ttnn.from_torch(
                    model.transformer1.pos_emb,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=mesh_mapper,
                )

            # Velocity embedding parameters (if exists)
            if hasattr(model.transformer1, "vel_emb"):
                parameters["transformer1"]["vel_emb_weight"] = ttnn.from_torch(
                    model.transformer1.vel_emb.weight,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=mesh_mapper,
                )
                parameters["transformer1"]["vel_emb_bias"] = ttnn.from_torch(
                    model.transformer1.vel_emb.bias.reshape((1, -1)),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=mesh_mapper,
                )

            # Transformer blocks - iterate over actual blocks
            if hasattr(model.transformer1, "blocks"):
                for i in range(len(model.transformer1.blocks)):
                    block = model.transformer1.blocks[i]
                    parameters["transformer1"][f"blocks_{i}"] = {}

                    # Layer norm 1
                    if hasattr(block, "ln1"):
                        parameters["transformer1"][f"blocks_{i}"]["ln1_weight"] = ttnn.from_torch(
                            block.ln1.weight,
                            dtype=ttnn.bfloat16,
                            device=device,
                            layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=mesh_mapper,
                        )
                        parameters["transformer1"][f"blocks_{i}"]["ln1_bias"] = ttnn.from_torch(
                            block.ln1.bias.reshape((1, -1)),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            mesh_mapper=mesh_mapper,
                        )

                    # Layer norm 2
                    if hasattr(block, "ln2"):
                        parameters["transformer1"][f"blocks_{i}"]["ln2_weight"] = ttnn.from_torch(
                            block.ln2.weight,
                            dtype=ttnn.bfloat16,
                            device=device,
                            layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=mesh_mapper,
                        )
                        parameters["transformer1"][f"blocks_{i}"]["ln2_bias"] = ttnn.from_torch(
                            block.ln2.bias.reshape((1, -1)),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            mesh_mapper=mesh_mapper,
                        )

                    # Attention
                    if hasattr(block, "attn"):
                        attn = block.attn
                        parameters["transformer1"][f"blocks_{i}"]["attn"] = {}

                        if (
                            hasattr(attn, "key")
                            and hasattr(attn, "query")
                            and hasattr(attn, "value")
                            and hasattr(attn, "proj")
                        ):
                            # Query
                            parameters["transformer1"][f"blocks_{i}"]["attn"]["query"] = {}
                            parameters["transformer1"][f"blocks_{i}"]["attn"]["query"]["weight"] = ttnn.from_torch(
                                attn.query.weight,
                                dtype=ttnn.bfloat16,
                                device=device,
                                layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=mesh_mapper,
                            )
                            parameters["transformer1"][f"blocks_{i}"]["attn"]["query"]["bias"] = ttnn.from_torch(
                                attn.query.bias.reshape((1, -1)),
                                dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT,
                                device=device,
                                mesh_mapper=mesh_mapper,
                            )

                            # Key
                            parameters["transformer1"][f"blocks_{i}"]["attn"]["key"] = {}
                            parameters["transformer1"][f"blocks_{i}"]["attn"]["key"]["weight"] = ttnn.from_torch(
                                attn.key.weight,
                                dtype=ttnn.bfloat16,
                                device=device,
                                layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=mesh_mapper,
                            )
                            parameters["transformer1"][f"blocks_{i}"]["attn"]["key"]["bias"] = ttnn.from_torch(
                                attn.key.bias.reshape((1, -1)),
                                dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT,
                                device=device,
                                mesh_mapper=mesh_mapper,
                            )

                            # Value
                            parameters["transformer1"][f"blocks_{i}"]["attn"]["value"] = {}
                            parameters["transformer1"][f"blocks_{i}"]["attn"]["value"]["weight"] = ttnn.from_torch(
                                attn.value.weight,
                                dtype=ttnn.bfloat16,
                                device=device,
                                layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=mesh_mapper,
                            )
                            parameters["transformer1"][f"blocks_{i}"]["attn"]["value"]["bias"] = ttnn.from_torch(
                                attn.value.bias.reshape((1, -1)),
                                dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT,
                                device=device,
                                mesh_mapper=mesh_mapper,
                            )

                            # Projection
                            parameters["transformer1"][f"blocks_{i}"]["attn"]["proj"] = {}
                            parameters["transformer1"][f"blocks_{i}"]["attn"]["proj"]["weight"] = ttnn.from_torch(
                                attn.proj.weight,
                                dtype=ttnn.bfloat16,
                                device=device,
                                layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=mesh_mapper,
                            )
                            parameters["transformer1"][f"blocks_{i}"]["attn"]["proj"]["bias"] = ttnn.from_torch(
                                attn.proj.bias.reshape((1, -1)),
                                dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT,
                                device=device,
                                mesh_mapper=mesh_mapper,
                            )

                    # MLP
                    if hasattr(block, "mlp"):
                        parameters["transformer1"][f"blocks_{i}"]["mlp_0_weight"] = ttnn.from_torch(
                            block.mlp[0].weight,
                            dtype=ttnn.bfloat16,
                            device=device,
                            layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=mesh_mapper,
                        )
                        parameters["transformer1"][f"blocks_{i}"]["mlp_0_bias"] = ttnn.from_torch(
                            block.mlp[0].bias.reshape((1, -1)),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            mesh_mapper=mesh_mapper,
                        )

                        parameters["transformer1"][f"blocks_{i}"]["mlp_2_weight"] = ttnn.from_torch(
                            block.mlp[2].weight,
                            # block.mlp[2].weight.T.contiguous(),
                            dtype=ttnn.bfloat16,
                            device=device,
                            layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=mesh_mapper,
                        )
                        parameters["transformer1"][f"blocks_{i}"]["mlp_2_bias"] = ttnn.from_torch(
                            block.mlp[2].bias.reshape((1, -1)),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            mesh_mapper=mesh_mapper,
                        )

        if hasattr(model, "transformer2"):
            parameters["transformer2"] = {}

            if hasattr(model.transformer2, "ln_f"):
                # )
                parameters["transformer2"]["ln_f_weight"] = ttnn.from_torch(
                    model.transformer2.ln_f.weight,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=mesh_mapper,
                )
                parameters["transformer2"]["ln_f_bias"] = ttnn.from_torch(
                    model.transformer2.ln_f.bias.reshape((1, -1)),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=mesh_mapper,
                )

            if hasattr(model.transformer2, "pos_emb"):
                print(f"Reference transformer2.pos_emb shape: {model.transformer2.pos_emb.shape}")
                parameters["transformer2"]["pos_emb"] = ttnn.from_torch(
                    model.transformer2.pos_emb,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=mesh_mapper,
                )

            # Velocity embedding parameters (if exists)
            if hasattr(model.transformer2, "vel_emb"):
                parameters["transformer2"]["vel_emb_weight"] = ttnn.from_torch(
                    model.transformer2.vel_emb.weight,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=mesh_mapper,
                )
                parameters["transformer2"]["vel_emb_bias"] = ttnn.from_torch(
                    model.transformer2.vel_emb.bias.reshape((1, -1)),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=mesh_mapper,
                )

            # Transformer blocks - iterate over actual blocks
            if hasattr(model.transformer2, "blocks"):
                for i in range(len(model.transformer2.blocks)):
                    block = model.transformer2.blocks[i]
                    parameters["transformer2"][f"blocks_{i}"] = {}

                    # Layer norm 1
                    if hasattr(block, "ln1"):
                        parameters["transformer2"][f"blocks_{i}"]["ln1_weight"] = ttnn.from_torch(
                            block.ln1.weight,
                            dtype=ttnn.bfloat16,
                            device=device,
                            layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=mesh_mapper,
                        )
                        parameters["transformer2"][f"blocks_{i}"]["ln1_bias"] = ttnn.from_torch(
                            block.ln1.bias.reshape((1, -1)),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            mesh_mapper=mesh_mapper,
                        )

                    # Layer norm 2
                    if hasattr(block, "ln2"):
                        parameters["transformer2"][f"blocks_{i}"]["ln2_weight"] = ttnn.from_torch(
                            block.ln2.weight,
                            dtype=ttnn.bfloat16,
                            device=device,
                            layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=mesh_mapper,
                        )
                        parameters["transformer2"][f"blocks_{i}"]["ln2_bias"] = ttnn.from_torch(
                            block.ln2.bias.reshape((1, -1)),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            mesh_mapper=mesh_mapper,
                        )

                    # Attention
                    if hasattr(block, "attn"):
                        attn = block.attn
                        parameters["transformer2"][f"blocks_{i}"]["attn"] = {}

                        if (
                            hasattr(attn, "key")
                            and hasattr(attn, "query")
                            and hasattr(attn, "value")
                            and hasattr(attn, "proj")
                        ):
                            # Query
                            parameters["transformer2"][f"blocks_{i}"]["attn"]["query"] = {}
                            parameters["transformer2"][f"blocks_{i}"]["attn"]["query"]["weight"] = ttnn.from_torch(
                                attn.query.weight,
                                dtype=ttnn.bfloat16,
                                device=device,
                                layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=mesh_mapper,
                            )
                            parameters["transformer2"][f"blocks_{i}"]["attn"]["query"]["bias"] = ttnn.from_torch(
                                attn.query.bias.reshape((1, -1)),
                                dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT,
                                device=device,
                                mesh_mapper=mesh_mapper,
                            )

                            # Key
                            parameters["transformer2"][f"blocks_{i}"]["attn"]["key"] = {}
                            parameters["transformer2"][f"blocks_{i}"]["attn"]["key"]["weight"] = ttnn.from_torch(
                                attn.key.weight,
                                dtype=ttnn.bfloat16,
                                device=device,
                                layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=mesh_mapper,
                            )
                            parameters["transformer2"][f"blocks_{i}"]["attn"]["key"]["bias"] = ttnn.from_torch(
                                attn.key.bias.reshape((1, -1)),
                                dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT,
                                device=device,
                                mesh_mapper=mesh_mapper,
                            )

                            # Value
                            parameters["transformer2"][f"blocks_{i}"]["attn"]["value"] = {}
                            parameters["transformer2"][f"blocks_{i}"]["attn"]["value"]["weight"] = ttnn.from_torch(
                                attn.value.weight,
                                dtype=ttnn.bfloat16,
                                device=device,
                                layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=mesh_mapper,
                            )
                            parameters["transformer2"][f"blocks_{i}"]["attn"]["value"]["bias"] = ttnn.from_torch(
                                attn.value.bias.reshape((1, -1)),
                                dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT,
                                device=device,
                                mesh_mapper=mesh_mapper,
                            )

                            # Projection
                            parameters["transformer2"][f"blocks_{i}"]["attn"]["proj"] = {}
                            parameters["transformer2"][f"blocks_{i}"]["attn"]["proj"]["weight"] = ttnn.from_torch(
                                attn.proj.weight,
                                dtype=ttnn.bfloat16,
                                device=device,
                                layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=mesh_mapper,
                            )
                            parameters["transformer2"][f"blocks_{i}"]["attn"]["proj"]["bias"] = ttnn.from_torch(
                                attn.proj.bias.reshape((1, -1)),
                                dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT,
                                device=device,
                                mesh_mapper=mesh_mapper,
                            )

                    # MLP
                    if hasattr(block, "mlp"):
                        parameters["transformer2"][f"blocks_{i}"]["mlp_0_weight"] = ttnn.from_torch(
                            block.mlp[0].weight.T.contiguous(),
                            dtype=ttnn.bfloat16,
                            device=device,
                            layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=mesh_mapper,
                        )
                        print(parameters["transformer2"][f"blocks_{i}"]["mlp_0_weight"])
                        parameters["transformer2"][f"blocks_{i}"]["mlp_0_bias"] = ttnn.from_torch(
                            block.mlp[0].bias.reshape((1, -1)),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            mesh_mapper=mesh_mapper,
                        )

                        parameters["transformer2"][f"blocks_{i}"]["mlp_2_weight"] = ttnn.from_torch(
                            # block.mlp[2].weight,
                            block.mlp[2].weight.T.contiguous(),
                            dtype=ttnn.bfloat16,
                            device=device,
                            layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=mesh_mapper,
                        )
                        parameters["transformer2"][f"blocks_{i}"]["mlp_2_bias"] = ttnn.from_torch(
                            block.mlp[2].bias.reshape((1, -1)),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            mesh_mapper=mesh_mapper,
                        )
    elif isinstance(model, Bottleneck):
        # Handle standalone Bottleneck model
        # conv1 (1x1 convolution)
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1.conv, model.conv1.bn)
        parameters["conv1"] = {}
        parameters["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

        # conv2 (3x3 grouped convolution)
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv2.conv, model.conv2.bn)
        parameters["conv2"] = {}
        parameters["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

        # conv3 (1x1 convolution)
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv3.conv, model.conv3.bn)
        parameters["conv3"] = {}
        parameters["conv3"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv3"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

        # SE module
        parameters["se"] = {}
        parameters["se"]["fc1"] = {}
        parameters["se"]["fc1"]["weight"] = ttnn.from_torch(
            model.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters["se"]["fc2"] = {}
        parameters["se"]["fc2"]["weight"] = ttnn.from_torch(
            model.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )

        # Downsample
        if (
            hasattr(model, "downsample")
            and model.downsample is not None
            and model.downsample.__class__.__name__ != "Identity"
        ):
            weight, bias = fold_batch_norm2d_into_conv2d(model.downsample[0], model.downsample[1])
            parameters["downsample"] = {}
            parameters["downsample"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
            bias = bias.reshape((1, 1, 1, -1))
            parameters["downsample"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
    elif isinstance(model, Stage):
        # Extract the stage layer (e.g., layer1, layer2, etc.)
        stage_layer = getattr(model.image_encoder.features, model.stage_name)

        parameters[model.stage_name] = {}

        # Process each bottleneck in the stage
        for block_idx, block_name in enumerate(["b1", "b2"]):
            if hasattr(stage_layer, block_name):
                block = getattr(stage_layer, block_name)
                parameters[model.stage_name][block_name] = {}

                # conv1 (1x1 convolution)
                weight, bias = fold_batch_norm2d_into_conv2d(block.conv1.conv, block.conv1.bn)
                parameters[model.stage_name][block_name]["conv1"] = {
                    "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                    "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                }

                # conv2 (3x3 grouped convolution)
                weight, bias = fold_batch_norm2d_into_conv2d(block.conv2.conv, block.conv2.bn)
                parameters[model.stage_name][block_name]["conv2"] = {
                    "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                    "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                }

                # conv3 (1x1 convolution)
                weight, bias = fold_batch_norm2d_into_conv2d(block.conv3.conv, block.conv3.bn)
                parameters[model.stage_name][block_name]["conv3"] = {
                    "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                    "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                }

                # SE module (no bias as you confirmed)
                parameters[model.stage_name][block_name]["se"] = {
                    "fc1": {
                        "weight": ttnn.from_torch(block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
                    },
                    "fc2": {
                        "weight": ttnn.from_torch(block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
                    },
                }

                # Downsample (if exists)
                if (
                    hasattr(block, "downsample")
                    and block.downsample is not None
                    and not isinstance(block.downsample, torch.nn.Identity)
                ):
                    weight, bias = fold_batch_norm2d_into_conv2d(block.downsample.conv, block.downsample.bn)
                    parameters[model.stage_name][block_name]["downsample"] = {
                        "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                        "bias": ttnn.from_torch(
                            bias.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                        ),
                    }
    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None, device=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper, device
        )

    return custom_mesh_preprocessor
