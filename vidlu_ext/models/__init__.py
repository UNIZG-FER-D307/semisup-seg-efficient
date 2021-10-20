# From https://github.com/kazuto1011/deeplab-pytorch

from .deeplabv2 import DeepLabV2


def DeepLabV2_ResNet101(n_classes):
    return DeepLabV2(n_classes=n_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
