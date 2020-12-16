import os
import sys
import shutil
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# build octree
curr_dir = os.path.dirname(os.path.realpath(__file__))
octree_dir = os.path.join(os.path.dirname(curr_dir), 'octree')
include_dirs = [octree_dir]
library_dirs = [os.path.join(octree_dir, 'build')]
libraries = ['octree_lib', 'cublas']
if sys.platform == 'win32':
  library_dirs[0] = os.path.join(library_dirs[0], 'Release')

def build_octree():
  octree_ext = os.path.join(octree_dir, 'external/octree-ext')
  if not os.path.exists(octree_ext):
    url = 'https://github.com/wang-ps/octree-ext.git'
    cmd = 'git clone --recursive {} {}'.format(url, octree_ext)
    print(cmd)
    os.system(cmd)

  octree_build = os.path.join(octree_dir, 'build')
  if os.path.exists(octree_build):
    shutil.rmtree(octree_build, ignore_errors=True)

  x64 = '-A x64' if sys.platform == 'win32' else ''
  cmd = 'mkdir {} && cd {} && cmake .. -DABI=ON -DKEY64=ON {} && ' \
        'cmake --build . --config Release && ./octree_test'.format(
            octree_build, octree_build, x64)
  print(cmd)
  os.system(cmd)
  

if "--build_octree" in sys.argv:
  build_octree()
  sys.argv.remove("--build_octree")

# builid ocnn
src = './cpp'  # '{}/cpp'.format(curr_dir)
sources = ['{}/{}'.format(src, cpp) for cpp in os.listdir(src)
           if cpp.endswith('.cpp') or cpp.endswith('.cu')]

setup(
    name='ocnn',
    version='1.0',
    install_requires=["torch", "numpy"],
    packages=['ocnn'],
    package_dir={'ocnn': 'ocnn'},
    ext_modules=[
        CUDAExtension(
            name='ocnn.nn',
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args={'cxx': ['-DKEY64']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
