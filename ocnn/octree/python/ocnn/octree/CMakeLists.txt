add_cython_target(_octree CXX)

add_library(_octree MODULE ${_octree})
python_extension_module(_octree)

target_include_directories(_octree
    PRIVATE
        ${NumPy_INCLUDE_DIR})

target_link_libraries(_octree octree_lib)
install(TARGETS _octree LIBRARY DESTINATION python/ocnn/octree)
