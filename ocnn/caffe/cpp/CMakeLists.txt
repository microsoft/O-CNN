cmake_minimum_required(VERSION 3.3)
project(libocnncaffe VERSION 18.10.09 LANGUAGES CXX)

find_package(Caffe REQUIRED)

add_library(ocnn_caffe
    src/lmdb_builder.cpp)

add_library(OCNN::Caffe ALIAS ocnn_caffe)

target_include_directories(ocnn_caffe
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_features(ocnn_caffe PUBLIC cxx_nullptr)
target_compile_options(ocnn_caffe
    PRIVATE 
        $<$<CXX_COMPILER_ID:GNU>:-Wall -fPIC -Wno-sign-compare -Wno-uninitialized>
        $<$<CXX_COMPILER_ID:MSVC>: /Wall>)

target_link_libraries(ocnn_caffe
    PUBLIC
        ${Caffe_LIBRARIES}
)
