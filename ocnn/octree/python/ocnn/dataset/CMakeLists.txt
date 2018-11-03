add_cython_target(_writable_data CXX)

add_library(_writable_data MODULE ${_writable_data})
python_extension_module(_writable_data)

install(TARGETS _writable_data LIBRARY DESTINATION python/ocnn/dataset)
