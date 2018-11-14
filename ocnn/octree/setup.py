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
    name="ocnn.base",
    version="18.11.01",
    description="Octree utilities",
    author='Microsoft',
    author_email="dapisani@microsoft.com",
    packages=['ocnn', 'ocnn.octree', 'ocnn.dataset', 'ocnn.utils'],
    zip_safe=False,
    install_requires=['six', 'Cython', 'numpy', 'pyyaml'],
    package_dir={'': 'python'},
    package_data={'ocnn.octree': ['*.pxd'],
                  'ocnn.dataset': ['*.pxd']}
)
