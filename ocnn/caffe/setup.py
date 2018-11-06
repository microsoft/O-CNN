from __future__ import print_function

from os import sys

try:
    from skbuild import setup
except ImportError:
    print('Scikit-build is needed to build package.',
          file=sys.stderr)
    print('Run \'pip install scikit-build\' before installing this package',
          file=sys.stderr)
    sys.exit(1)

setup(
    name="ocnn.caffe",
    version="18.09.05",
    description="ONN Caffe Modules",
    author='Microsoft',
    author_email="dapisani@microsoft.com",
    packages=['ocnn', 'ocnn.caffe'],
    install_requires=['ocnn.base'],
    zip_safe=False,
    package_dir={'': 'python'},
    package_data = {'ocnn.caffe': ['*.pxd']}
)
