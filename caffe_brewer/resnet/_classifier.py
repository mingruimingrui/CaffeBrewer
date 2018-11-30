from caffe2.python import brew


def add_classifier_ops(model, blob_in, dim_in, num_classes):
    pool_blob = brew.average_pool(model, blob_in, 'layer5_pool', kernel=7)
    fc_blob = brew.fc(model, pool_blob, 'fc', dim_in, num_classes)
    return brew.softmax(model, fc_blob, 'softmax')
