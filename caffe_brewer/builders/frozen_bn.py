from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags


def add_frozen_bn_ops(model, blob_in, blob_out, dim, inplace=False):
    """
    Affine transformation to replace BN in networks where BN cannot be used.
    (eg. when minibatch size is too small)

    The operation can be done inplace to save memory.
    """
    prefix = blob_out

    weight = model.create_param(
        param_name=prefix + '_w',
        initializer=initializers.Initializer('ConstantFill', value=1.0),
        tags=ParameterTags.WEIGHT,
        shape=[dim, ]
    )

    bias = model.create_param(
        param_name=prefix + '_b',
        initializer=initializers.Initializer('ConstantFill', value=0.0),
        tags=ParameterTags.BIAS,
        shape=[dim, ]
    )

    if inplace:
        return model.net.AffineChannel([blob_in, weight, bias], blob_in)
    else:
        return model.net.AffineChannel([blob_in, weight, bias], blob_out)
