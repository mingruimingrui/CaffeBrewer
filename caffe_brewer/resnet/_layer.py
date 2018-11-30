

def add_layer_ops(
    model,
    blob_in,
    prefix,
    dim_in,
    dim_inner,
    dim_out,
    add_block,
    num_blocks,
    stride=1
):
    """
    Dynamic resnet layer builder that builds bottleneck layers
    Recommended prefix format: 'layer{layer_nb}'

    Args:
        add_block: An add_bottleneck_ops function,
        num_blocks: Number of bottleneck blocks in this layer
    """
    blob_cur = blob_in

    for i in range(num_blocks):
        is_first_block = i == 0

        blob_cur = add_block(
            model,
            blob_in=blob_cur,
            prefix='{}_{}'.format(prefix, i),
            dim_in=dim_in if is_first_block else dim_out,
            dim_inner=dim_inner,
            dim_out=dim_out,
            stride=stride if is_first_block else 1,
            downsample=is_first_block
        )

    return blob_cur
