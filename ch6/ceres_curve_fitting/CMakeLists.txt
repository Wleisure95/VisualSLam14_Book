cmake_minimum_required( VERSION 2.8 )
project( ceres_curve_fitting )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

set( Ceres_DIR  "/home/leisure/all_ws/3rdparty/3rd_lib/Ceres-1.14.0/lib/cmake/Ceres")
set( OpenCV_DIR "/opt/ros/kinetic/share/OpenCV-3.3.1-dev")

# 添加cmake模块以使用ceres库
#list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )


# 寻找Ceres库并添加它的头文件
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )
include_directories( 
    "/usr/include/eigen3"
)
#message(${PROJECT_SOURCE_DIR})

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS}  ${OpenCV_INCLUDE_DIRS})

message(${Ceres_DIR})
message(${OpenCV_INCLUDE_DIRS})
add_executable( curve_fitting main.cpp )
# 与Ceres和OpenCV链接
target_link_libraries( curve_fitting ${CERES_LIBRARIES} ${OpenCV_LIBS} )


