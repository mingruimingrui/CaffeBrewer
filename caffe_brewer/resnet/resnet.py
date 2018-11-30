"""
Implemented to mimic structure from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
to be able to transfer weights using torchvision
"""
from .config import make_config
from ._bottleneck import get_add_bottleneck_ops
from ._stem import add_stem_ops, add_maxpool_ops
from ._layer import add_layer_ops
from ._classifier import add_classifier_ops

base_layer_sizes = {
    1: 64,
    2: 64,
    3: 128,
    4: 256,
    5: 512
}

expansion_dict = {
    'resnet18': 1,
    'resnet34': 1,
    'resnet50': 4,
    'resnet101': 4,
    'resnet152': 4
}

network_structures = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3]
}

__all__ = ['add_resnet_ops']


def add_resnet_ops(model, blob_in, config_file=None, **kwargs):
    """
    Adds the ResNet ops to the model

    Please refer to caffe_brewer.resnet.config.py for
    settings to config_file and kwargs

    Args:
        blob_in: The image blob of shape [num_batch, 3, height, width]
        config_file: A config file storing resnet settings
        kwargs: The key word arguments for building resnet
    """
    config = make_config(config_file, **kwargs)

    # Determine some constants based on ResNet type
    expansion = expansion_dict[config.TYPE]
    structure = network_structures[config.TYPE]
    add_bottleneck_ops = get_add_bottleneck_ops(config.TYPE)

    # Add stem
    C1 = add_stem_ops(model, blob_in, dim=base_layer_sizes[1])
    C2 = add_maxpool_ops(model, C1)

    if config.FREEZE_AT == 1:
        # C1 is frozen here as maxpool does not contain any weights
        model.StopGradient(C2, C2)

    C2 = add_layer_ops(
        model,
        blob_in=C2, prefix='layer1',
        dim_in=base_layer_sizes[1],
        dim_inner=base_layer_sizes[1],
        dim_out=base_layer_sizes[2]*expansion,
        add_block=add_bottleneck_ops,
        num_blocks=structure[0],
        stride=1
    )

    if config.FREEZE_AT == 2:
        model.StopGradient(C2, C2)

    if config.NO_TOP and config.LAST_CONV == 2:
        return C2, config

    C3 = add_layer_ops(
        model,
        blob_in=C2, prefix='layer2',
        dim_in=base_layer_sizes[2]*expansion,
        dim_inner=base_layer_sizes[3],
        dim_out=base_layer_sizes[3]*expansion,
        add_block=add_bottleneck_ops,
        num_blocks=structure[1],
        stride=2
    )

    if config.FREEZE_AT == 3:
        model.StopGradient(C3, C3)

    if config.NO_TOP and config.LAST_CONV == 3:
        return C2, C3, config

    C4 = add_layer_ops(
        model,
        blob_in=C3, prefix='layer3',
        dim_in=base_layer_sizes[3]*expansion,
        dim_inner=base_layer_sizes[4],
        dim_out=base_layer_sizes[4]*expansion,
        add_block=add_bottleneck_ops,
        num_blocks=structure[2],
        stride=2
    )

    if config.FREEZE_AT == 4:
        model.StopGradient(C4, C4)

    if config.NO_TOP and config.LAST_CONV == 4:
        return C2, C3, C4, config

    C5 = add_layer_ops(
        model,
        blob_in=C4, prefix='layer4',
        dim_in=base_layer_sizes[4]*expansion,
        dim_inner=base_layer_sizes[5],
        dim_out=base_layer_sizes[5]*expansion,
        add_block=add_bottleneck_ops,
        num_blocks=structure[3],
        stride=2
    )

    if config.FREEZE_AT == 5:
        model.StopGradient(C5, C5)

    if config.NO_TOP and config.LAST_CONV == 5:
        return C2, C3, C4, C5, config

    return add_classifier_ops(
        model,
        blob_in=C5,
        dim_in=base_layer_sizes[5]*expansion,
        num_classes=config.NUM_CLASSES
    ), config
