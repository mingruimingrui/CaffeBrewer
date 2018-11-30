import os
import sys

# Append CaffeBrewer root directory to sys
test_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(test_dir, os.pardir))
sys.path.append(root_dir)

import pytest

from PIL import Image
import numpy as np

import torch
import torchvision

from caffe2.python import workspace, core, model_helper
from caffe_brewer.resnet import add_resnet_ops, load_pretrained_weights

sample_image_path = os.path.join(test_dir, 'data', 'mug.jpg')
assert os.path.isfile(sample_image_path)


def calc_mean_diff(A, B):
    diff = np.array(A) - np.array(B)
    return np.mean(np.abs(diff))


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

    return image_blob_name, image


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
    pass
    scope_name = 'test'
    model_name = 'test_resnet_build'
    image_blob_name = 'image'

    workspace.ResetWorkspace()
    model = make_default_model_helper(model_name)

    with core.NameScope(scope_name):
        blob_in, _ = add_image_blob(image_blob_name)
        blob_out, resnet_config = add_resnet_ops(model, blob_in)

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net, overwrite=True)

    with core.NameScope(scope_name):
        load_pretrained_weights(resnet_config.TYPE)

    workspace.RunNet(model_name)


@pytest.mark.parametrize('resnet_type', [
    'resnet18',
    'resnet50'
])
def test_resnet_pretrained(resnet_type):
    """
    Test that weights are transferred correctly
    """
    scope_name = 'test'
    model_name = 'test_resnet_pretrained_' + resnet_type
    image_blob_name = 'image'

    # Generate Caffe output
    workspace.ResetWorkspace()
    model = make_default_model_helper(model_name)

    with core.NameScope(scope_name):
        blob_in, image = add_image_blob(image_blob_name)
        blob_out, resnet_config = \
            add_resnet_ops(model, blob_in, TYPE=resnet_type)

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net, overwrite=True)

    with core.NameScope(scope_name):
        load_pretrained_weights(resnet_config.TYPE)

    workspace.RunNet(model_name)
    caffe_output = workspace.FetchBlob('{}/softmax'.format(scope_name))

    # Generate Torch output
    torch_model_fn = getattr(torchvision.models, resnet_type)
    torch_model = torch_model_fn(pretrained=True).eval()
    torch_output = torch_model(torch.Tensor(image))
    torch_output = torch.softmax(torch_output, 1)
    torch_output = torch_output.cpu().data.numpy()

    # Ensure that the difference is not too much
    mean_diff = calc_mean_diff(caffe_output, torch_output)
    assert mean_diff < 1e-7, \
        'Difference between caffe_output and torch_output too large {}'.format()
