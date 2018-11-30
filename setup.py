#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

exclude_dirs = ['dep', 'test']


setup(
    name='CaffeBrewer',
    version='0.1',
    author='mingrui',
    url='https://github.com/mingruimingrui/CaffeBrewer',
    description='model zoo in caffe2',
    packages=find_packages(exclude=exclude_dirs)
)
