# download the pybind11
set(pybind11_path "${PROJECT_SOURCE_DIR}/external/octree-ext/pybind11")
# set(pybind11_file "${PROJECT_SOURCE_DIR}/external/pybind11.zip")
# if(NOT EXISTS ${pybind11_path})
#   file(DOWNLOAD
#     https://github.com/pybind/pybind11/archive/master.zip
#     ${pybind11_file}
#     SHOW_PROGRESS
#   )
#   execute_process(COMMAND ${CMAKE_COMMAND} -E tar xfz 
#     ${pybind11_file}
#     WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external)
#   file(REMOVE ${pybind11_file})
# endif()

# add the python interface
add_subdirectory(${pybind11_path} ${pybind11_path}/build)
pybind11_add_module(pyoctree pyoctree.cpp)
target_link_libraries(pyoctree PRIVATE octree_lib)
set_target_properties(pyoctree PROPERTIES FOLDER "python")
