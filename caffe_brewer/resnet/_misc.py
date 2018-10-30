"""
Script containing misc functions for ResNet
"""

class InvalidResNetType(Exception):
    pass

def raise_invalid_resnet_type_error(type):
    raise InvalidResNetType('{} is an invalid resnet type'.format(type))


def get_layer_sizes(type='resnet50'):
    """
    Get the structure of the hidden layers based on the type of resnet
    """
    if type == 'resnet18':
        return [2, 2, 2, 2]
    elif type == 'resnet34':
        return [3, 4, 6, 3]
    elif type == 'resnet50':
        return [3, 4, 6, 3]
    elif type == 'resnet101':
        return [3, 4, 23, 3]
    elif type == 'resnet152':
        return [3, 8, 36, 3]
    else:
        raise_invalid_resnet_type_error(type)


def get_expansion(type='resnet50'):
    """
    Get expansion rates of features based on the type of resnet
    """
    if type in {'resnet18', 'resnet34'}:
        return 1
    if type in {'resnet50', 'resnet101', 'resnet152'}:
        return 4
    else:
        raise_invalid_resnet_type_error(type)


def handle_post_layer(model, features, level, config):
    """
    Freezes the layers below if appropriate
    Returns a boolean to indicate if it is the end of the network
    """
    if config.FREEZE_AT == level:
        model.StopGradient(features[level], features[level])
    return config.NO_TOP and config.LAST_CONV == level
