from caffe2.python import brew
from ..builders.frozen_bn import add_frozen_bn


def add_bottleneck_1(
    model,
    blob_in,
    prefix,
    dim_in,
    dim_inner,
    dim_out,
    stride=1,
    downsample=False
):
    """
    Default bottleneck layer for resnet18 and resnet34
    prefix : Of the format 'layer{layer_nb}_{block_nb}'
    """
    residual = blob_in
    if downsample:
        residual = brew.conv(
            model, blob_in, prefix + '_downsample_0',
            dim_in=dim_in, dim_out=dim_out,
            kernel=1, stride=stride, pad=0, no_bias=True
        )
        residual = add_frozen_bn(
            model, residual, prefix + '_downsample_1',
            dim=dim_out, inplace=False
        )

    # 1x1 BN  ReLU
    blob_a = brew.conv(
        model, blob_in, prefix + '_conv1',
        dim_in=dim_in, dim_out=dim_inner,
        kernel=3, stride=stride, pad=1, no_bias=True
    )
    blob_a = add_frozen_bn(
        model, blob_a, prefix + '_bn1',
        dim=dim_inner, inplace=True
    )
    blob_a = brew.relu(model, blob_a, blob_a)

    # 1x1 BN
    blob_b = brew.conv(
        model, blob_a, prefix + '_conv2',
        dim_in=dim_inner, dim_out=dim_out,
        kernel=3, stride=1, pad=1, no_bias=True
    )
    blob_b = add_frozen_bn(
        model, blob_b, prefix + '_bn2',
        dim=dim_out, inplace=False
    )

    # Sum and ReLU
    blob_out = model.net.Sum([residual, blob_b], prefix + '_sum')
    return brew.relu(model, blob_out, blob_out)


def add_bottleneck_2(
    model,
    blob_in,
    prefix,
    dim_in,
    dim_inner,
    dim_out,
    stride=1,
    downsample=False
):
    """
    Default bottleneck layer for resnet50, resnet101 and resnet152
    prefix : Of the format 'layer{layer_nb}_{block_nb}'
    """
    residual = blob_in
    if downsample:
        residual = brew.conv(
            model, blob_in, prefix + '_downsample_0',
            dim_in=dim_in, dim_out=dim_out,
            kernel=1, stride=stride, pad=0, no_bias=True
        )
        residual = add_frozen_bn(
            model, residual, prefix + '_downsample_1',
            dim=dim_out, inplace=False
        )

    # 1x1 BN  ReLU
    blob_a = brew.conv(
        model, blob_in, prefix + '_conv1',
        dim_in=dim_in, dim_out=dim_inner,
        kernel=1, stride=1, pad=0, no_bias=True
    )
    blob_a = add_frozen_bn(
        model, blob_a, prefix + '_bn1',
        dim=dim_inner, inplace=True
    )
    blob_a = brew.relu(model, blob_a, blob_a)

    # 3x3 BN ReLU
    blob_b = brew.conv(
        model, blob_a, prefix + '_conv2',
        dim_in=dim_inner, dim_out=dim_inner,
        kernel=3, stride=stride, pad=1, no_bias=True
    )
    blob_b = add_frozen_bn(
        model, blob_b, prefix + '_bn2',
        dim=dim_inner, inplace=True
    )
    blob_b = brew.relu(model, blob_b, blob_b)

    # 1x1 BN
    blob_c = brew.conv(
        model, blob_b, prefix + '_conv3',
        dim_in=dim_inner, dim_out=dim_out,
        kernel=1, stride=1, pad=0, no_bias=True
    )
    blob_c = add_frozen_bn(
        model, blob_c, prefix + '_bn3',
        dim=dim_out, inplace=False
    )

    # Sum and ReLU
    blob_out = model.net.Sum([residual, blob_c], prefix + '_sum')
    return brew.relu(model, blob_out, blob_out)


def get_add_bottleneck(type=50):
    """
    Get the type of bottleneck block based on the type of resnet
    prefix : Of the format 'layer{layer_nb}_{block_nb}'
    """
    if type in [18, 34]:
        return add_bottleneck_1
    else:
        return add_bottleneck_2
