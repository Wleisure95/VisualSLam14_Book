cmake_minimum_required( VERSION 2.8 )
project ( myslam )

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

############### dependencies ######################
# Eigen
include_directories( "/usr/include/eigen3" )
# OpenCV
set( OpenCV_DIR "/opt/ros/kinetic/share/OpenCV-3.3.1-dev")
find_package( OpenCV 3 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
message(${OpenCV_INCLUDE_DIRS})
# Sophus 
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )
# G2O
include_directories( "/home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/include" )
link_directories("/home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/lib/")
#find_package( G2O REQUIRED )
#include_directories( ${G2O_INCLUDE_DIRS} )

set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    g2o_core g2o_stuff g2o_types_sba
)
############### dependencies ######################
include_directories( ${PROJECT_SOURCE_DIR}/include )
add_subdirectory( src )
add_subdirectory( test )