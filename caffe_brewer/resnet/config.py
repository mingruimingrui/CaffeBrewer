"""
ResNet config system
"""
import logging
from ..utils.config_system import ConfigSystem

logger = logging.getLogger(__name__)

_C = ConfigSystem()
config = _C

valid_resnet_types = {
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152'
}

resnet_type_to_channel_sizes = {
    'resnet18' : [ 64, 128,  256,  512],
    'resnet34' : [ 64, 128,  256,  512],
    'resnet50' : [256, 512, 1024, 2048],
    'resnet101': [256, 512, 1024, 2048],
    'resnet152': [256, 512, 1024, 2048]
}


# ---------------------------------------------------------------------------- #
# ResNet options
# ---------------------------------------------------------------------------- #

# The type of ResNet to use
# For valid ResNet types refer to above
_C.TYPE = 'resnet50'

# The number of classes to classifiy
_C.NUM_CLASSES = 1000

# If true, uses resnet as a feature extractor, FC layer will not be added
_C.NO_TOP = False

# Last conv layer to use (based on number of pooling)
# eg. 5 will mean the last layer of conv for resnet
_C.LAST_CONV = 5

# Layers below this will be frozen
_C.FREEZE_AT = 0

# Freeze the batch norm layers?
# You have to freeze BN at the moment
# _C.FREEZE_BN = True

# Use group normalization?
# Option not yet available
# _C.USE_GN = False

# The number of groups to use for group normalization
# Option not yet available
# _C.GN_NUM_GROUPS = 32

# ---------------------------------------------------------------------------- #
# End of options
# ---------------------------------------------------------------------------- #
_C.immutable(True)

def make_config(config_file, **kwargs):
    return _C.clone().make_config(config_file, validate_config, **kwargs)

def validate_config(config):
    """
    Check validity of configs
    """
    assert config.TYPE in valid_resnet_types, \
        '{} is invalid ResNet type'.format(config.TYPE)

    assert config.LAST_CONV in {2, 3, 4, 5}, \
        'Currently only [2, 3, 4, 5] are accepted values for LAST_CONV'

    assert config.FREEZE_AT in {0, 1, 2, 3, 4, 5}, \
        'FREEZE_AT must be a one of [0, 1, 2, 3, 4, 5]'

    assert config.FREEZE_AT <= config.LAST_CONV, \
        'FREEZE_AT must be less than or equal to LAST_CONV'
