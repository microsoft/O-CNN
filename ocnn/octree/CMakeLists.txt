cmake_minimum_required(VERSION 3.3)
project(Octree)

# The version number.
set(OCTREE_VERSION_MAJOR 1)
set(OCTREE_VERSION_MINOR 0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# should we use the Miniball.hpp
option(USE_MINIBALL "Use the Miniball.hpp" ON)
option(USE_WINDOWS_IO "Use the header io.h provided by Windows" OFF)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file(
  "${PROJECT_SOURCE_DIR}/OctreeConfig.h.in"
  "${PROJECT_BINARY_DIR}/octree_config.h")

# miniball
if(USE_MINIBALL)
  set(miniball_path "${PROJECT_SOURCE_DIR}/external/miniball") 
  set(miniball_hpp  "${miniball_path}/Miniball.hpp")
  if(NOT EXISTS ${miniball_hpp})
    file(DOWNLOAD
      https://people.inf.ethz.ch/gaertner/subdir/software/miniball/Miniball.hpp
      ${miniball_hpp}
      SHOW_PROGRESS
    )
  endif()
  include_directories(${miniball_path})
endif()

# rply
file(GLOB  src_rply 
  "${PROJECT_SOURCE_DIR}/external/rply-1.1.4/*.h"
  "${PROJECT_SOURCE_DIR}/external/rply-1.1.4/*.c")
if(NOT src_rply)
  # download and extract
  set(rply_file ${PROJECT_SOURCE_DIR}/external/rply.tar.gz)
  file(DOWNLOAD
    http://w3.impa.br/~diego/software/rply/rply-1.1.4.tar.gz
    ${rply_file}
    SHOW_PROGRESS
  )
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xfz 
    ${rply_file} 
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external)
  file(REMOVE ${rply_file})
  file(GLOB  src_rply 
    "${PROJECT_SOURCE_DIR}/external/rply-1.1.4/*.h"
    "${PROJECT_SOURCE_DIR}/external/rply-1.1.4/*.c")
endif()
include_directories("${PROJECT_SOURCE_DIR}/external/rply-1.1.4")
add_library(rply ${src_rply})

# files
file(GLOB  src_octree_lib 
  "${PROJECT_SOURCE_DIR}/include/octree/*.h"
  "${PROJECT_SOURCE_DIR}/src/octree/*.cpp")

# includes
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include/octree")

# add the octree lib
add_library(octree_lib ${src_octree_lib})
target_compile_features(octree_lib PUBLIC cxx_nullptr)
target_compile_options(octree_lib
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU>: -fPIC>
        )

if (SKBUILD)
    find_package(PythonExtensions REQUIRED)
    find_package(Cython REQUIRED)
    find_package(NumPy REQUIRED)

    add_subdirectory(python/ocnn/dataset)
    add_subdirectory(python/ocnn/octree)
else()
    # add tools
    file(GLOB_RECURSE srcs "${PROJECT_SOURCE_DIR}/tools/*.cpp")
    foreach(source ${srcs})
      get_filename_component(name ${source} NAME_WE)
      if(name MATCHES "build_octree") 
        set(name "octree") 
      endif()
      add_executable(${name} ${source})
      target_link_libraries(${name} octree_lib)
      if(name MATCHES "ply2points") 
        target_link_libraries(${name} rply)
      endif()
    endforeach(source)
endif()
