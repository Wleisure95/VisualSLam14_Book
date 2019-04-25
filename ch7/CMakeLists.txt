cmake_minimum_required( VERSION 2.8 )
project( vo1 )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

set( OpenCV_DIR "/opt/ros/kinetic/share/OpenCV-3.3.1-dev")
set( Ceres_DIR  "/home/leisure/all_ws/3rdparty/3rd_lib/Ceres-1.14.0/lib/cmake/Ceres")

find_package( OpenCV 3 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 
#find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )
find_package( Ceres REQUIRED )

include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    "/home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/include"
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
    ${CERES_INCLUDE_DIRS} 
    #"/home/leisure/all_ws/3rdparty/3rd_lib/Ceres-1.14.0/include"
)
message(${OpenCV_INCLUDE_DIRS})

link_directories("/home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/lib/")
#link_directories("/home/leisure/all_ws/3rdparty/3rd_lib/Ceres-1.14.0/lib")

#message(${G2O_INCLUDE_DIR})
#message(${G2O_INCLUDE_DIRS})
add_executable( feature_extraction feature_extraction.cpp  )
target_link_libraries( feature_extraction ${OpenCV_LIBS} )

# add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp extra.cpp ) # use this if in OpenCV2 
add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp )
target_link_libraries( pose_estimation_2d2d ${OpenCV_LIBS} )

# add_executable( triangulation triangulation.cpp extra.cpp) # use this if in opencv2 
add_executable( triangulation triangulation.cpp )
target_link_libraries( triangulation ${OpenCV_LIBS} )

add_executable( pose_estimation_3d2d pose_estimation_3d2d.cpp )
target_link_libraries( pose_estimation_3d2d 
   ${OpenCV_LIBS}
   ${CSPARSE_LIBRARY}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
   ${CERES_LIBRARIES} 
   #ceres
)
message( ${CERES_LIBRARIES})
add_executable( pose_estimation_3d3d pose_estimation_3d3d.cpp )
target_link_libraries( pose_estimation_3d3d 
   ${OpenCV_LIBS}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension 
   ${CSPARSE_LIBRARY}
   ${CERES_LIBRARIES} 
#   "/home/leisure/all_ws/3rdparty/3rd_lib/Ceres-1.14.0/lib/libceres.a"
)
