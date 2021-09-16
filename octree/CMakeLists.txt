cmake_minimum_required(VERSION 3.12.3)

# options
option(USE_MINIBALL "Use the Miniball.hpp" ON)
option(USE_WINDOWS_IO "Use the header io.h provided by Windows" OFF)
option(USE_CUDA "CPU only, without CUDA" ON)
option(USE_PYTHON "Build the python interface with pybind11" OFF)
option(USE_GLOG "Use glog" OFF)
option(USE_OPENMP "Use OpenMP for speed up" OFF)
option(USE_RPLY "Use the library rply in the project octree" OFF)
option(KEY64 "Use 64-bit key for deeper octrees" OFF)
option(ABI "Open CXX11_ABI compatibility" OFF)
set(VCPKG "<Set VCPKG path>" CACHE PATH "The VCPKG path, containing glog and gtest")


# set languanges
if(USE_CUDA)
  message(STATUS "octree: USE_CUDA")
  project(Octree LANGUAGES CUDA CXX C)
  add_definitions(-DUSE_CUDA)
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 11)
    set(CMAKE_CUDA_STANDARD 14)
  else()
    set(CMAKE_CUDA_STANDARD 11)
  endif()
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  message(STATUS "octree: CMAKE_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}" )
else()
  project(Octree LANGUAGES C CXX)
endif()

# use folder to orgnize the project
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# # c++11 Support
set (CMAKE_CXX_STANDARD 11) 
set (CMAKE_POSITION_INDEPENDENT_CODE ON)
if(ABI)
  message(STATUS "octree: _GLIBCXX_USE_CXX11_ABI=0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
else()
  message(STATUS "octree: _GLIBCXX_USE_CXX11_ABI=1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
endif()

# add libraries
if(USE_MINIBALL)
  message(STATUS "octree: USE_MINIBALL")
  add_definitions(-DUSE_MINIBALL) 
  set(miniball_path "${PROJECT_SOURCE_DIR}/external/octree-ext/miniball") 
  include_directories(${miniball_path})
endif()

if(USE_RPLY)
  message(STATUS "octree: USE_RPLY")
  add_definitions(-DUSE_RPLY)
endif()
file(GLOB  src_rply 
  "${PROJECT_SOURCE_DIR}/external/octree-ext/rply-1.1.4/*.h"
  "${PROJECT_SOURCE_DIR}/external/octree-ext/rply-1.1.4/*.c")
include_directories("${PROJECT_SOURCE_DIR}/external/octree-ext/rply-1.1.4")
add_library(rply ${src_rply})
set_target_properties(rply PROPERTIES FOLDER "ext")

if(KEY64)
  message(STATUS "octree: KEY64")
  add_definitions(-DKEY64)
endif()

if(USE_GLOG)
  message(STATUS "octree: USE_GLOG")
  add_definitions(-DUSE_GLOG)
  set(glog_path "${PROJECT_SOURCE_DIR}/external/octree-ext/glog")
  set(WITH_GFLAGS OFF CACHE BOOL "Disable WITH_GFLAGS")
  set(BUILD_TESTING OFF CACHE BOOL "Disable BUILD_TESTING")
  add_subdirectory(${glog_path})
  set_target_properties(glog PROPERTIES FOLDER "ext")
endif()

if(USE_WINDOWS_IO)
  message(STATUS "octree: USE_WINDOWS_IO")
  add_definitions(-DUSE_WINDOWS_IO)
endif()

if(USE_OPENMP)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()
endif()

set(happly_path "${PROJECT_SOURCE_DIR}/external/octree-ext/happly") 
include_directories(${happly_path})

set(nanoflann_path "${PROJECT_SOURCE_DIR}/external/octree-ext/nanoflann/include")
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
add_library(octree_lib ${src_octree_lib})
target_compile_features(octree_lib PUBLIC cxx_std_11)
if(USE_RPLY)
  target_link_libraries(octree_lib rply)
endif(USE_RPLY)
if(USE_GLOG)
  target_link_libraries(octree_lib glog::glog)
endif(USE_GLOG)

# add tools
file(GLOB_RECURSE srcs "${PROJECT_SOURCE_DIR}/tools/*.cpp")
foreach(source ${srcs})
  get_filename_component(name ${source} NAME_WE)
  if(name MATCHES "build_octree") 
    set(name "octree") 
  endif()
  add_executable(${name} ${source})
  target_link_libraries(${name} octree_lib)
  set_target_properties(${name} PROPERTIES FOLDER "tools")
endforeach(source)


# add the python interface
if(USE_PYTHON)
  message(STATUS "octree: USE_PYTHON")
  add_subdirectory(python)
endif()

# add gtest
set(gtest_path "${PROJECT_SOURCE_DIR}/external/octree-ext/googletest")
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(${gtest_path}  EXCLUDE_FROM_ALL)
set_target_properties(gtest gtest_main PROPERTIES FOLDER "ext")

# add the test
add_executable(octree_test ${src_test})
target_link_libraries(octree_test octree_lib gtest gtest_main)