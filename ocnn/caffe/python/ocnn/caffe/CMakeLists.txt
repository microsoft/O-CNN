add_cython_target(_ocnn_caffe CXX)

add_library(_ocnn_caffe MODULE ${_ocnn_caffe})
python_extension_module(_ocnn_caffe)

target_link_libraries(_ocnn_caffe OCNN::Caffe)
install(TARGETS _ocnn_caffe LIBRARY DESTINATION python/ocnn/caffe)
