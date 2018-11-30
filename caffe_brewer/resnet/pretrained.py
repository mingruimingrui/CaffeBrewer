import logging

import torch
import torchvision

from caffe2.python import core, workspace
from .config import valid_resnet_types
from ..utils import conversion

logger = logging.getLogger(__name__)


def load_pretrained_weights(resnet_type, scope=None, verbosity=0):
    """
    Load ResNet weights from a pretrained torchvision model and insert said
    weights into ResNet blobs

    Ensure that parameters have already been initialized at this point

    Args:
        resnet_type: The type of ResNet
        scope: The scope that ResNet blobs reside in. If None, core.ScopedName
            will be used
        verbosity: Level of logging to use
            0 - Only Errors
            1 - Start and end logged
            2 - Skipped blobs logged
    """
    assert resnet_type in valid_resnet_types, \
        '{} resnet_type is invalid'.format(resnet_type)

    if verbosity >= 1:
        logger.info('Loading pretrained weights for {}'.format(resnet_type))

    # Load a torch model and extract it's weights
    torch_model_fn = getattr(torchvision.models, resnet_type)
    torch_model = torch_model_fn(pretrained=True)
    torch_weights = torch_model.state_dict()

    blob_names = set(workspace.Blobs())
    count_added = 0
    total_num_weights = len(torch_weights)

    for weight_name in torch_weights.keys():
        # Ignore running_mean, running_var and num_batches_tracked of batchnorm
        # Will be consolidated at batchnorm weights and bias
        if 'running_' in weight_name or 'num_batches_tracked' in weight_name:
            continue

        # Process weight name and check that weight exists in workspace
        processed_name = conversion.format_torch_name(weight_name)
        if scope is None:
            processed_name = core.ScopedName(processed_name)
        else:
            processed_name = scope + processed_name

        if processed_name not in blob_names:
            if verbosity >= 2:
                logger.warn('Skipping {}'.format(weight_name))
            continue

        # Get pretrained weights from torch model
        # Batchnorm layers will have to be handled differently
        if conversion.is_batchnorm_bias(weight_name, torch_weights):
            pretrained_weight = conversion.get_caffe2_batchnorm_bias(weight_name, torch_weights)

        elif conversion.is_batchnorm_weight(weight_name, torch_weights):
            pretrained_weight = conversion.get_caffe2_batchnorm_weight(weight_name, torch_weights)

        else:
            # For conv layers we will be transfering weights directly
            pretrained_weight = torch_weights[weight_name].cpu().data.numpy()

        workspace.FeedBlob(processed_name, pretrained_weight)
        count_added += 1

    del torch_weights, torch_model
    torch.cuda.empty_cache()

    if verbosity >= 1:
        logger.info('{}/{} Pretrained weights are loaded'.format(count_added, total_num_weights))
