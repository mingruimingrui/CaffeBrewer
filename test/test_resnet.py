import os
import sys

# Append CaffeBrewer root directory to sys
test_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(test_dir, os.pardir))
sys.path.append(root_dir)

from PIL import Image
import numpy as np

from caffe2.python import workspace, core, model_helper
from caffe_brewer.resnet import add_resnet_ops, load_pretrained_weights

sample_image_path = os.path.join(test_dir, 'data', 'mug.jpg')
assert os.path.isfile(sample_image_path)


def preproc_image(image):
    # Performs VGG preprocessing and formats image into NCHW format
    image = image.resize((224, 224))
    image = np.array(image).astype('float32')
    image = image.transpose(2, 0, 1) / 255.0
    image[0] = (image[0] - 0.485) / 0.229
    image[1] = (image[1] - 0.456) / 0.224
    image[2] = (image[2] - 0.406) / 0.225
    return np.expand_dims(image, 0)


def add_image_blob(image_blob_name='image'):
    if image_blob_name in workspace.Blobs():
        return image_blob_name

    image = Image.open(sample_image_path)
    image = preproc_image(image)

    device_opt = core.scope.CurrentDeviceScope()
    scoped_image_blob = core.ScopedName(image_blob_name)

    if device_opt is None:
        workspace.CreateBlob(scoped_image_blob)
        workspace.FeedBlob(scoped_image_blob, image)
    else:
        workspace.CreateBlob(scoped_image_blob, device_option=device_opt)
        workspace.FeedBlob(scoped_image_blob, image, device_option=device_opt)

    return image_blob_name


def make_default_model_helper(model_name='resnet'):
    return model_helper.ModelHelper(
        name=model_name,
        arg_scope={
            'order': 'NCHW',
            'use_cudnn': True,
            'cudnn_exhaustive_search': True
        }
    )


def test_resnet_build():
    """
    Test if add_resnet_ops can be built
    """
    scope_name = 'test'
    model_name = 'resnet'
    image_blob_name = 'image'

    workspace.ResetWorkspace()
    model = make_default_model_helper(model_name)

    with core.NameScope(scope_name):
        blob_in = add_image_blob(image_blob_name)
        blob_out, resnet_config = add_resnet_ops(model, blob_in)

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net, overwrite=True)

    with core.NameScope(scope_name):
        load_pretrained_weights(resnet_config.TYPE)

    workspace.RunNet(model_name)
