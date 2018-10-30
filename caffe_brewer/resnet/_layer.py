
def add_layer(
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
    Args:
        prefix : Of the format 'layer{layer_nb}'
        add_block : Function of either type add_bottleneck_1 or add_bottleneck_2
    """
    blob_cur = blob_in
    for i in range(num_blocks):
        is_first_block = i == 0
        is_last_block = i == num_blocks - 1

        blob_cur = add_block(
            model,
            blob_cur,
            prefix + '_{}'.format(i),
            dim_in if is_first_block else dim_out,
            dim_inner,
            dim_out,
            stride=stride if is_first_block else 1,
            downsample=is_first_block
        )

    return blob_cur
