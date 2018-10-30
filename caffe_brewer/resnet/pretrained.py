import logging
import numpy as np
import torchvision
from caffe2.python import core, workspace

logger = logging.getLogger(__name__)


def replace_bias(name, replacement=''):
    return name[:-4] + replacement

def replace_weight(name, replacement=''):
    return name[:-6] + replacement

def format_torch_name(name):
    """ Formats a torchvision model layer name into a caffe2 form """
    name = name.replace('.', '_')

    if name.endswith('bias'):
        name = replace_bias(name, 'b')

    if name.endswith('weight'):
        name = replace_weight(name, 'w')

    return name

def is_batchnorm_bias(torch_name, torch_weights):
    """ Crude way of checking if a torch weight is a batchnorm bais """
    if torch_name.endswith('bias'):
        return replace_bias(torch_name, 'running_mean') in torch_weights
    return False

def is_batchnorm_scale(torch_name, torch_weights):
    """ Crude way of checking if a torch weight is a batchnorm scale """
    if torch_name.endswith('weight'):
        return replace_weight(torch_name, 'running_mean') in torch_weights
    return False

def get_batchnorm_weights(torch_name, torch_weights):
    if torch_name.endswith('bias'):
        running_mean_name = replace_bias(torch_name, 'running_mean')
        running_var_name = replace_bias(torch_name, 'running_var')
        bias_name = torch_name
        scale_name = replace_bias(torch_name, 'weight')

    if torch_name.endswith('weight'):
        running_mean_name = replace_weight(torch_name, 'running_mean')
        running_var_name = replace_weight(torch_name, 'running_var')
        bias_name = replace_weight(torch_name, 'bias')
        scale_name = torch_name

    running_mean = torch_weights[running_mean_name].cpu().data.numpy()
    running_var = torch_weights[running_var_name].cpu().data.numpy()
    bias = torch_weights[bias_name].cpu().data.numpy()
    scale = torch_weights[scale_name].cpu().data.numpy()

    return running_mean, running_var, bias, scale


def load_torchvision_resnet_weights(type, scope=None):
    """
    Load ResNet weights from a pretrained torchvision model and inserts said
    weights into ResNet blobs

    Ensure that parameters have already been initialized at this point

    Args:
        type : The type of ResNet this is must be one of [18, 34, 50, 101, 152]
        scope : The scope that ResNet blobs reside in. If None, core.ScopedName will be used
    """
    logger.info('Loading pretrained weights for ResNet{}'.format(type))

    # Load a torch model and extract it's weights
    torch_model = getattr(torchvision.models, 'resnet{}'.format(type))(pretrained=True)
    torch_weights = torch_model.state_dict()

    # Keep track of skipped weights and added weights
    # Can be used to give warning to user
    count_skipped = 0
    count_added = 0
    blob_names = set(workspace.Blobs())

    for weight_name in torch_weights.keys():
        # Ignore running_mean, running_var and num_batches_tracked of batchnorm
        # Will be consolidated at batchnorm weights and bias
        if 'running_' in weight_name or 'num_batches_tracked' in weight_name:
            continue

        # Process weight name and check that weight exists in workspace
        processed_name = format_torch_name(weight_name)
        if scope is None:
            processed_name = core.ScopedName(processed_name)
        else:
            processed_name = scope + processed_name

        if processed_name not in blob_names:
            logger.warn('Skipping {}'.format(weight_name))
            count_skipped += 1
            continue

        # Get pretrained weights from torch model, batchnorm bias and weights will
        # have to be handled differently
        if is_batchnorm_bias(weight_name, torch_weights):
            running_mean, running_var, beta, gamma = get_batchnorm_weights(weight_name, torch_weights)
            pretrained_weight = beta - gamma * running_mean / np.sqrt(running_var + 1e-5)

        elif is_batchnorm_scale(weight_name, torch_weights):
            running_mean, running_var, beta, gamma = get_batchnorm_weights(weight_name, torch_weights)
            pretrained_weight = gamma / np.sqrt(running_var + 1e-5)

        else:
            # For Conv layers we will be trasfering weights directly
            pretrained_weight = torch_weights[weight_name].cpu().data.numpy()

        import pdb; pdb.set_trace()
        workspace.FeedBlob(processed_name, pretrained_weight)
        count_added += 1

    del torch_weights, torch_model
    logger.info('{}/{} pretrained weights are loaded'.format(count_added, count_skipped))
