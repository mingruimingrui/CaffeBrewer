"""
Current behavior of stem and maxpool ops are probably less than ideal
blob_out names are fixed
"""
from caffe2.python import brew
from ..builders.frozen_bn import add_frozen_bn


def add_stem_ops(model, blob_in, dim):
    blob_inner = brew.conv(
        model,
        blob_in=blob_in, blob_out='conv1',
        dim_in=3, dim_out=dim,
        kernel=7, stride=2, pad=3, no_bias=True
    )
    blob_out = add_frozen_bn(
        model,
        blob_in=blob_inner, dim_out='bn1',
        dim=dim, inplace=True
    )
    return brew.relu(model, blob_out, blob_out)


def add_maxpool_ops(model, blob_in):
    return brew.max_pool(model, blob_in, 'pool1', kernel=3, pad=1, stride=2)
