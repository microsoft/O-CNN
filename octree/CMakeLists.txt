cmake_minimum_required(VERSION 3.8.2)

# options
option(USE_MINIBALL "Use the Miniball.hpp" ON)
option(USE_WINDOWS_IO "Use the header io.h provided by Windows" OFF)
option(USE_CUDA "CPU only, without CUDA" ON)
option(USE_PYTHON "Build the python interface with pybind11" ON)
option(USE_GLOG "Use glog" OFF)
option(USE_OPENMP "Use OpenMP for speed up" OFF)
option(USE_RPLY "Use the library rply in the project octree" OFF)
set(VCPKG "<Set VCPKG path>" CACHE PATH "The VCPKG path, containing glog and gtest")


# set languanges
if(USE_CUDA)
  project(Octree LANGUAGES CUDA CXX C)
  add_definitions(-DUSE_CUDA)
else()
  project(Octree LANGUAGES C CXX)
endif()

# use folder to orgnize the project
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# # c++11 Support
set (CMAKE_CXX_STANDARD 11) 
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# add target_link_libraries
if(USE_MINIBALL) 
  add_definitions(-DUSE_MINIBALL) 
  set(miniball_path "${PROJECT_SOURCE_DIR}/external/octree-ext/miniball") 
  include_directories(${miniball_path})
endif()

if(USE_RPLY)
  add_definitions(-DUSE_RPLY)
  file(GLOB  src_rply 
    "${PROJECT_SOURCE_DIR}/external/octree-ext/rply-1.1.4/*.h"
    "${PROJECT_SOURCE_DIR}/external/octree-ext/rply-1.1.4/*.c")
  include_directories("${PROJECT_SOURCE_DIR}/external/octree-ext/rply-1.1.4")
  add_library(rply ${src_rply})
endif()

if(USE_GLOG)
  add_definitions(-DUSE_GLOG)
endif()

if(USE_WINDOWS_IO)
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
  add_subdirectory(python)
endif()


# return from here on Ubuntu
if(NOT MSVC)
  return()
endif()

# add library for the viewer
set(antweakbar_path "${PROJECT_SOURCE_DIR}/external/octree-ext/AntTweakBar/include")
include_directories(${antweakbar_path})

# add the viewer
add_executable(viewer ${src_viewer})
target_link_libraries(viewer octree_lib)
target_link_libraries(viewer octree_lib)

# add the scanner
add_executable(virtual_scanner ${src_scanner})
target_link_libraries(virtual_scanner octree_lib)
target_compile_definitions(virtual_scanner PRIVATE CGAL_HEADER_ONLY)

# add the test
add_executable(octree_test ${src_test})
target_link_libraries(octree_test octree_lib)


# VCPKG: link gtest & glog
set(VCPKG "${VCPKG}/installed/x64-windows")
if(USE_GLOG)
  include_directories("${VCPKG}/include")  
  target_link_libraries(octree_lib 
    debug "${VCPKG}/debug/lib/glog.lib" 
    optimized "${VCPKG}/lib/glog.lib")  
endif()
target_link_libraries(octree_test 
  debug "${VCPKG}/debug/lib/manual-link/gtest_maind.lib"  
  optimized "${VCPKG}/lib/manual-link/gtest_main.lib"
  debug "${VCPKG}/debug/lib/gtestd.lib" 
  optimized "${VCPKG}/lib/gtest.lib")


# # add the install targets
# install(TARGETS Tutorial DESTINATION bin)
# install(FILES "${PROJECT_BINARY_DIR}/TutorialConfig.h"
#   DESTINATION include)