from caffe2.python import brew
from ..builders.frozen_bn import add_frozen_bn


def add_stem(model, blob_in, dim):
    blob_inner = brew.conv(
        model, blob_in, 'conv1',
        3, dim,
        kernel=7, stride=2, pad=3, no_bias=True
    )
    blob_inner = add_frozen_bn(
        model, blob_inner, 'bn1',
        dim=dim, inplace=True
    )
    blob_out = brew.relu(model, blob_inner, blob_inner)

    return blob_out


def add_maxpool(model, blob_in):
    return brew.max_pool(model, blob_in, 'pool1', kernel=3, pad=1, stride=2)
