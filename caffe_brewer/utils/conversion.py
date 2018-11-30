"""
Script containing helper functions to help convert torch weights into
Caffe2 blobs
"""
import numpy as np


def format_torch_name(name):
    """ Formats a torch layer name into a caffe2 blob name """
    name = name.replace('.', '_')

    if name.endswith('bias'):
        name = name[:-4] + 'b'

    if name.endswith('weight'):
        name = name[:-6] + 'w'

    return name


def is_batchnorm_bias(torch_name, torch_weights):
    """
    Crude way of checking if a torch weight is a batchnorm bias
    Args:
        torch_name: The name of a torch weight
        torch_weights: The state_dict of a torch model
    """
    if not torch_name.endswith('bias'):
        return False
    running_mean_name = torch_name[:-4] + 'running_mean'
    return running_mean_name in torch_weights


def is_batchnorm_weight(torch_name, torch_weights):
    """
    Crude way of checking if a torch weight is a batchnorm weight
    Args:
        torch_name: The name of a torch weight
        torch_weights: The state_dict of a torch model
    """
    if not torch_name.endswith('weight'):
        return False
    running_mean_name = torch_name[:-6] + 'running_mean'
    return running_mean_name in torch_weights


def get_caffe2_batchnorm_bias(torch_name, torch_weights):
    """
    Compute the batch norm weight
    Args:
        torch_name: The name of a torch weight
        torch_weights: The state_dict of a torch model
    """
    assert is_batchnorm_bias(torch_name, torch_weights)
    running_mean_name = torch_name[:-4] + 'running_mean'
    running_var_name = torch_name[:-4] + 'running_var'
    bias_name = torch_name
    scale_name = torch_name[:-4] + 'weight'

    running_mean = torch_weights[running_mean_name].cpu().data.numpy()
    running_var = torch_weights[running_var_name].cpu().data.numpy()
    bias = torch_weights[bias_name].cpu().data.numpy()
    scale = torch_weights[scale_name].cpu().data.numpy()

    return bias - scale * running_mean / np.sqrt(running_var + 1e-5)


def get_caffe2_batchnorm_weight(torch_name, torch_weights):
    """
    Compute the batch norm weight
    Args:
        torch_name: The name of a torch weight
        torch_weights: The state_dict of a torch model
    """
    assert is_batchnorm_weight(torch_name, torch_weights)
    running_var_name = torch_name[:-6] + 'running_var'
    scale_name = torch_name

    running_var = torch_weights[running_var_name].cpu().data.numpy()
    scale = torch_weights[scale_name].cpu().data.numpy()

    return scale / np.sqrt(running_var + 1e-5)
