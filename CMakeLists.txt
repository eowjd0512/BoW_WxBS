cmake_minimum_required(VERSION 3.0.0)
project(BoW)
# The version number.
set (SSD_VERSION_MAJOR 1)
set (SSD_VERSION_MINOR 0)
add_definitions(-DA64)
add_definitions(-std=c++11)
SET(WITH_CAFFE FALSE)
#
enable_language (Fortran)
FIND_PACKAGE(LAPACK REQUIRED)
if (LAPACK_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LAPACK_CXX_FLAGS}")
endif (LAPACK_FOUND)
#
find_package(OpenMP)
if (OPENMP_FOUND)
  message("--- found OpenMP")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
#
SET(CMAKE_BUILD_TYPE "RELEASE")
#
IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)
#
#
#if(WITH_CAFFE)
#add_definitions(-DCPU_ONLY) #caffe cpu_only
#include_directories(/home/old-ufo/caffe-clear/include)
#include_directories(/home/old-ufo/caffe-clear/build/src/)
#link_directories(/home/old-ufo/caffe-clear/build/lib)
#endif(WITH_CAFFE)


find_package(OpenCV QUIET)
# flags
add_definitions("-DENABLE_SSE")
#set(CMAKE_CXX_FLAGS
#   "${SSE_FLAGS} -O3 -g -std=c++0x -march=native"
#  # "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
#)
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

link_directories(${CMAKE_SOURCE_DIR}/thirdParty/vlfeat/bin/glnxa64)
link_directories(${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/thirdParty/vlfeat/vl
  ${PROJECT_SOURCE_DIR}/thirdParty/vlfeat
) 

#WxBS
include_directories(
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/libdaisy/include/
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/libdaisy/include/daisy
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/matching/
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/degensac/
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/matutls/
  #/home/jun/MODS/mods
) 
#KDE
include_directories(
  ${PROJECT_SOURCE_DIR}/thirdParty/KDE
)

set(BoW_SOURCE_FILES
  #${PROJECT_SOURCE_DIR}/BoW.cpp
  ${PROJECT_SOURCE_DIR}/src/BoW_WxBS.cpp
  ${PROJECT_SOURCE_DIR}/src/WxBSdet_desc.cpp
  #${PROJECT_SOURCE_DIR}/thirdParty/WxBS/correspondencebank.cpp
)
#WxBS
set(WxBS_SOURCE_FILES
  #${PROJECT_SOURCE_DIR}/BoW.cpp
  #${PROJECT_SOURCE_DIR}/thirdParty/WxBS/imagerepresentation.h
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/imagerepresentation.cpp
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/correspondencebank.cpp
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/io_mods.cpp
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/ssdesc-cpp-1.1.1/ssdesc.cc
)
#KDE
set(KDE_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/thirdParty/KDE/kde.cpp
)

#compile main library
add_library(BoW ${BoW_SOURCE_FILES} ${WxBS_SOURCE_FILES} ${KDE_SOURCE_FILES})
target_link_libraries(BoW
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/libmatching.a
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/libdegensac.a
  lapack
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/libsynth-detection.a 
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/detectors/libhelpers.a
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/detectors/mser/libmser.a
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/detectors/affinedetectors/libaffinedetectors.a
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/akaze/lib/libAKAZE.a
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/inih/libinih.a
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/libMatch/libMatch.a
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/libNumerics/libNumerics.a
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/matutls/libmatutls.a
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/mrogh/libmrogh.a
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/opensurf/libopensurf.a
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/TILDE/c++/libTILDE.so
  ${PROJECT_SOURCE_DIR}/thirdParty/wxbs/build/z/libdaisy.a
  ${PROJECT_SOURCE_DIR}/thirdParty/vlfeat/bin/glnxa64/libvl.so
 
 )
 #TARGET_LINK_LIBRARIES(export_descriptors inih matching synth-detection mser affinedetectors  AKAZE vl mrogh  
 #opencv_core m daisy z)
# decide if we have openCV
if (OpenCV_FOUND)
    message("--- found OpenCV, compiling WxBS_BOW_opencv library.")
include_directories( ${OpenCV_INCLUDE_DIRS} )
#set(HAS_OPENCV 1)
add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(main BoW ${OpenCV_LIBS} vl)
  #target_link_libraries(main BoW ${OpenCV_LIBS} vl inih matching synth-detection mser
  #affinedetectors  AKAZE mrogh m daisy z)
endif()
