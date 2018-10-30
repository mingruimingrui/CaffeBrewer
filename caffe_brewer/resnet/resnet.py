"""
Implemented based on
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
to be able to transfer weights
"""
import logging
from caffe2.python import brew

from .config import make_config
from ._misc import get_expansion, get_layer_sizes
from ._bottleneck import get_add_bottleneck
from ._stem import add_stem, add_maxpool
from ._layer import add_layer
from ._classifier import add_classifier

logger = logging.getLogger(__name__)

base_sizes = {
    1: 64,
    2: 64,
    3: 128,
    4: 256,
    5: 512
}


def add_resnet_ops(model, blob_in, config=None):
    """
    Adds the ResNet ops to the model

    Args:
        blob_in: The image blob of shape [num_batch, 3, height, width]
            If FC layers are present, image height width should be (224, 224)
        configs: A resnet config object made using caffe_brewer.resnet.make_config
    """
    if config is None:
        logger.warn('"config" not provided, default config will be used')
        config = make_config()

    features = {}

    expansion = get_expansion(configs.TYPE)
    layer_sizes = get_layer_sizes(configs.TYPE)
    add_bottleneck = get_add_bottleneck(configs.TYPE)

    C1 = add_stem(model, blob_in, dim=base_sizes[1])
    C2 = add_maxpool(model, C1)

    # Handle freezing of C1 differently due to maxpool not containing any weights
    if configs.FREEZE_AT == 1:
        model.StopGradient(C2, C2)

    features[2] = add_layer(
        model, C2, 'layer1',
        dim_in=base_sizes[1],
        dim_inner=base_sizes[1],
        dim_out=base_sizes[2]*expansion,
        add_block=add_bottleneck,
        num_blocks=layer_sizes[0],
        stride=1
    )

    if config.FREEZE_AT == 2:
        model.StopGradient(features[2], features[2])

    if config.LAST_CONV >= 3:
        features[3] = add_layer(
            model, features[2], 'layer2',
            dim_in=base_sizes[2]*expansion,
            dim_inner=base_sizes[3],
            dim_out=base_sizes[3]*expansion,
            add_block=add_bottleneck,
            num_blocks=layer_sizes[1],
            stride=2
        )

    if config.FREEZE_AT == 3:
        model.StopGradient(features[3], features[3])

    if config.LAST_CONV >= 4:
        features[4] = add_layer(
            model, features[3], 'layer3',
            dim_in=base_sizes[3]*expansion,
            dim_inner=base_sizes[4],
            dim_out=base_sizes[4]*expansion,
            add_block=add_bottleneck,
            num_blocks=layer_sizes[2],
            stride=2
        )

    if config.FREEZE_AT == 4:
        model.StopGradient(features[4], features[4])

    if config.LAST_CONV == 5:
        features[5] = add_layer(
            model, features[4], 'layer4',
            dim_in=base_sizes[4]*expansion,
            dim_inner=base_sizes[5],
            dim_out=base_sizes[5]*expansion,
            add_block=add_bottleneck,
            num_blocks=layer_sizes[3],
            stride=2
        )

    if config.FREEZE_AT == 5:
        model.StopGradient(features[5], features[5])

    if config.NO_TOP:
        return features

    else:
        return add_classifier(
            model, features[config.LAST_CONV],
            dim_in=base_sizes[config.LAST_CONV]*expansion,
            num_classes=configs.NUM_CLASSES
        )
