from skbuild import setup

setup(
    name="ocnn.base",
    version="18.11.01",
    description="Octree utilities",
    author='Microsoft',
    author_email="dapisani@microsoft.com",
    packages=['ocnn', 'ocnn.octree', 'ocnn.dataset'],
    zip_safe=False,
    install_requires=['numpy'],
    package_dir = {'': 'python'},
    package_data = {'ocnn.octree': ['*.pxd'],
                    'ocnn.dataset': ['*.pxd']}
)
