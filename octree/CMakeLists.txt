cmake_minimum_required(VERSION 3.0)

# options
option(USE_MINIBALL "Use the Miniball.hpp" ON)
option(USE_RPLY "Use the library rply in the project octree" ON)
option(USE_WINDOWS_IO "Use the header io.h provided by Windows" OFF)
option(USE_OPENMP "Use OpenMP for speed up" OFF)
option(USE_GLOG "Use glog" OFF)
option(USE_CUDA "CPU only, without CUDA" ON)
option(USE_PYTHON "Build the python interface with pybind11" OFF)
set(VCPKG "<Set VCPKG path>" CACHE PATH "The VCPKG path, containing glog and gtest")
# option(SHARED_LIB "Build share library, only feasible on Ubuntu" OFF)


# use folder to orgnize the project
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# # c++11 Support
# set (CMAKE_CXX_STANDARD 11) 
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# add definitions
if(USE_MINIBALL) 
  add_definitions(-DUSE_MINIBALL) 
endif()
if(USE_RPLY)
  add_definitions(-DUSE_RPLY)
endif()
if(USE_GLOG)
  add_definitions(-DUSE_GLOG)
endif()
if(USE_WINDOWS_IO)
  add_definitions(-DUSE_WINDOWS_IO)
endif()
if(USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()

# set languanges
if(USE_CUDA)
  project(Octree LANGUAGES CUDA CXX C)
else()
  project(Octree LANGUAGES C CXX)
endif()

# OpenMP Support
find_package(OpenMP)
if (OPENMP_FOUND AND USE_OPENMP)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

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
if(SHARED_LIB)
  add_library(rply SHARED ${src_rply})
else()
  add_library(rply ${src_rply})
endif()

# nanoflann
set(nanoflann_path "${PROJECT_SOURCE_DIR}/external/nanoflann-master/include")
if(NOT EXISTS "${nanoflann_path}/nanoflann.hpp")
  # download and extract
  set(nanoflann_file "${PROJECT_SOURCE_DIR}/external/nanoflann-master.zip")
  file(DOWNLOAD
    https://github.com/jlblancoc/nanoflann/archive/master.zip
    ${nanoflann_file}
    SHOW_PROGRESS
  )
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xfz 
    ${nanoflann_file} 
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external)
  file(REMOVE ${nanoflann_file})
endif()
include_directories(${nanoflann_path})




# files
file(GLOB  src_octree_lib
  "${PROJECT_SOURCE_DIR}/octree/*.h"
  "${PROJECT_SOURCE_DIR}/octree/*.cpp")
if(USE_CUDA)
  file(GLOB  cuda_octree_lib
    "${PROJECT_SOURCE_DIR}/octree/*.cu")
  set(src_octree_lib ${src_octree_lib} ${cuda_octree_lib})
endif()

file(GLOB  src_viewer  
  "${PROJECT_SOURCE_DIR}/viewer/*.h"
  "${PROJECT_SOURCE_DIR}/viewer/*.cpp" )
file(GLOB  src_scanner  
  "${PROJECT_SOURCE_DIR}/scanner/*.h"
  "${PROJECT_SOURCE_DIR}/scanner/*.cpp" )
file(GLOB  src_test 
  "${PROJECT_SOURCE_DIR}/test/*.cpp") 

# includes
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/octree")
include_directories("${PROJECT_SOURCE_DIR}/viewer")
include_directories("${PROJECT_SOURCE_DIR}/scanner")


# add the octree lib
if(SHARED_LIB)
  add_library(octree_lib SHARED ${src_octree_lib} )
else()
  add_library(octree_lib ${src_octree_lib} )
endif()
target_compile_features(octree_lib PUBLIC cxx_std_11)
if(USE_RPLY)
  target_link_libraries(octree_lib rply)
endif(USE_RPLY)

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
  set_target_properties(${name} PROPERTIES FOLDER "tools")
endforeach(source)

