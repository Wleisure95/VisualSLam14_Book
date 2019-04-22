cmake_minimum_required( VERSION 2.8 )
project( g2o_curve_fitting )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用ceres库
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

set( OpenCV_DIR "/opt/ros/kinetic/share/OpenCV-3.3.1-dev")

# 寻找G2O
find_package( G2O REQUIRED )
find_package( Cholmod REQUIRED )
include_directories( 
    ${G2O_INCLUDE_DIR}
    "/usr/include/eigen3"
)

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS}  ${CHOLMOD_INCLUDE_DIR} )

message(${G2O_INCLUDE_DIR})

message(${OpenCV_INCLUDE_DIRS})
add_executable( curve_fitting main.cpp )

# 与G2O和OpenCV链接
target_link_libraries( curve_fitting 
    ${OpenCV_LIBS}
    ${G2O_CORE_LIBRARY} 
    /home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/lib/libg2o_stuff.so
)

add_executable(test_g2o test.cpp )

# 与G2O和OpenCV链接
target_link_libraries( test_g2o 
    ${OpenCV_LIBS}
    /home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/lib/libg2o_core.so 
    /home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/lib/libg2o_types_slam3d.so 
    /home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/lib/libg2o_solver_csparse.so 
    /home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/lib/libg2o_stuff.so 
    /home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/lib/libg2o_csparse_extension.so 
    /home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/lib/libg2o_types_sba.so
    ${CHOLMOD_LIBRARIES}
    )
