cmake_minimum_required(VERSION 3.0.0)
project(BoW)

# The version number.
set (SSD_VERSION_MAJOR 1)
set (SSD_VERSION_MINOR 0)

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

link_directories(${CMAKE_SOURCE_DIR}/thirdParty/vlfeat/bin/glnxa64)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/thirdParty/vlfeat/vl
  ${PROJECT_SOURCE_DIR}/thirdParty/vlfeat
) 

#WxBS
include_directories(
  /home/jun/MODS/mods
  /home/jun/MODS/mods/libdaisy/include/
  /home/jun/MODS/mods/matching/
  #/home/jun/MODS/mods
) 
link_directories(/home/jun/MODS/mods/build)

find_package(OpenCV QUIET)
# flags
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O3 -g -std=c++0x -march=native"
#   "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
)

set(BoW_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/BoW.cpp
  ${PROJECT_SOURCE_DIR}/BoW_WxBS.cpp
  ${PROJECT_SOURCE_DIR}/WxBSdet_desc.cpp
)


#compile main library
add_library(BoW ${BoW_SOURCE_FILES} )

#add_executable(SSDSLAM main.cpp)

# decide if we have openCV
if (OpenCV_FOUND)
    message("--- found OpenCV, compiling SSD_opencv library.")
    include_directories( ${OpenCV_INCLUDE_DIRS} )
    set(HAS_OPENCV 1)
    add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp)
    target_link_libraries(main BoW ${OpenCV_LIBS} vl mods)
endif()
