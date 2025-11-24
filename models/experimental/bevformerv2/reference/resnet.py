import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional, Type, List, Tuple


# ---- Conv Helpers ---- #


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ---- MMDet Bottleneck (style="pytorch") ---- #


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        with_cp: bool = False,  # checkpoint unused but kept for parity
    ):
        super().__init__()

        # MMDet uses pytorch-style: stride is in the 3×3 conv
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)

        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)
        self.bn2 = norm_layer(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.with_cp = with_cp

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


# ---- MMDet ResNet Implementation ---- #


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Bottleneck],
        layers: List[int],
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        zero_init_residual: bool = True,
    ):
        super().__init__()

        self.out_indices = out_indices
        self._norm_layer = norm_layer

        self.inplanes = 64

        # MMDet default: 7x7 stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # MMDet: zero-init last BN
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    # ---- Build One Stage ---- #

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # MMDet uses conv1x1(stride=stride)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, 1, downsample, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, 1, None, norm_layer))

        return nn.Sequential(*layers)

    # ---- Forward ---- #

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for idx, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            if idx in self.out_indices:
                outs.append(x)

        return tuple(outs)


# ---- Factory ---- #
def resnet50_mmdet(out_indices=(1, 2, 3)):
    return ResNet(Bottleneck, [3, 4, 6, 3], out_indices=out_indices)
