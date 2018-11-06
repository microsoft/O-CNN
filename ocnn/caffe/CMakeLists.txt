cmake_minimum_required(VERSION 3.3)

project(python_caffe)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

add_subdirectory(cpp/ "${CMAKE_CURRENT_BINARY_DIR}/_octree_generator")
add_subdirectory(python/ocnn/caffe/)
