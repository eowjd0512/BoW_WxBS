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
FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif (OPENMP_FOUND)
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
set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O3 -g -std=c++0x -march=native"
  # "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
)
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

link_directories(${CMAKE_SOURCE_DIR}/thirdParty/vlfeat/bin/glnxa64)
link_directories(${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/thirdParty/vlfeat/vl
  ${PROJECT_SOURCE_DIR}/thirdParty/vlfeat
) 

#WxBS
include_directories(
  ${PROJECT_SOURCE_DIR}/thirdParty/WxBS
  ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/libdaisy/include/
  ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/libdaisy/include/daisy
  ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/matching/
  ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/degensac/
  #/home/jun/MODS/mods
) 
#add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/libdaisy z)
#add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/inih)
#add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/matutls)
#add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/detectors)
#add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/detectors/affinedetectors)
#add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/detectors/mser )
#add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/opensurf )
#add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/libMatch)
#add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/libNumerics)
#add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/akaze)
#add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/mrogh)
#add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/TILDE/c++)

####### DEGENSAC
#set(degensac_srcs
#        ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/degensac/DegUtils.c
#        ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/degensac/exp_ranF.c
#        ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/degensac/exp_ranH.c
#        ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/degensac/Ftools.c
#        ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/degensac/hash.c
#        ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/degensac/Htools.c
#        ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/degensac/ranF.c
#        ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/degensac/ranH2el.c
#        ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/degensac/ranH.c
#        ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/degensac/rtools.c
#        ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/degensac/utools.c
#        ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/degensac/lapwrap.c
#)
#add_library(degensac ${degensac_srcs})
#TARGET_LINK_LIBRARIES(degensac matutls lapack)
##### End of DEGENSAC

###################Matching
#set(matching_srcs
#${PROJECT_SOURCE_DIR}/thirdParty/WxBS/matching/siftdesc.cpp
#${PROJECT_SOURCE_DIR}/thirdParty/WxBS/matching/matching.cpp
#${PROJECT_SOURCE_DIR}/thirdParty/WxBS/orsa.cpp
#)
#add_library(matching ${matching_srcs})
#TARGET_LINK_LIBRARIES(matching ${OpenCV_LIBS} degensac lapack Match Numerics)
#
#
################### View synthesis
#
#set(synth-detection_srcs  
#${PROJECT_SOURCE_DIR}/thirdParty/WxBS/synth-detection.cpp)
#
#add_library(synth-detection ${synth-detection_srcs})
#if(WITH_CAFFE)
#TARGET_LINK_LIBRARIES(synth-detection matching opensurf
#mser AKAZE affinedetectors ${OpenCV_LIBS} helpers vl caffe glog protobuf lmdb blas)
#else(WITH_CAFFE)
#TARGET_LINK_LIBRARIES(synth-detection matching opensurf
#mser affinedetectors AKAZE helpers vl ${OpenCV_LIBS})
#endif(WITH_CAFFE)


#link_directories(/home/jun/MODS/mods/build)



set(BoW_SOURCE_FILES
  #${PROJECT_SOURCE_DIR}/BoW.cpp
  ${PROJECT_SOURCE_DIR}/BoW_WxBS.cpp
  ${PROJECT_SOURCE_DIR}/WxBSdet_desc.cpp
  #${PROJECT_SOURCE_DIR}/thirdParty/WxBS/correspondencebank.cpp
)
#WxBS
set(WxBS_SOURCE_FILES
  #${PROJECT_SOURCE_DIR}/BoW.cpp
  #${PROJECT_SOURCE_DIR}/thirdParty/WxBS/imagerepresentation.h
  ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/imagerepresentation.cpp
  ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/correspondencebank.cpp
  ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/io_mods.cpp
  ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/ssdesc-cpp-1.1.1/ssdesc.cc
)

#compile main library
add_library(BoW ${BoW_SOURCE_FILES} ${WxBS_SOURCE_FILES})
target_link_libraries(BoW
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/libsynth-detection.a 
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/detectors/libhelpers.a
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/detectors/mser/libmser.a
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/detectors/affinedetectors/libaffinedetectors.a
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/akaze/lib/libAKAZE.a
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/inih/libinih.a
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/libMatch/libMatch.a
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/libNumerics/libNumerics.a
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/matutls/libmatutls.a
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/mrogh/libmrogh.a
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/opensurf/libopensurf.a
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/TILDE/c++/libTILDE.so
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/z/libdaisy.a
 ${PROJECT_SOURCE_DIR}/thirdParty/vlfeat/bin/glnxa64
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/libmatching.a
 ${PROJECT_SOURCE_DIR}/thirdParty/WxBS/build/libdegensac.a
 )
 #TARGET_LINK_LIBRARIES(export_descriptors inih matching synth-detection mser affinedetectors  AKAZE vl mrogh  
 #opencv_core m daisy z)
# decide if we have openCV
if (OpenCV_FOUND)
    message("--- found OpenCV, compiling SSD_opencv library.")
include_directories( ${OpenCV_INCLUDE_DIRS} )
#set(HAS_OPENCV 1)
add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(main BoW ${OpenCV_LIBS} vl)
  #target_link_libraries(main BoW ${OpenCV_LIBS} vl inih matching synth-detection mser
  #affinedetectors  AKAZE mrogh m daisy z)
endif()
