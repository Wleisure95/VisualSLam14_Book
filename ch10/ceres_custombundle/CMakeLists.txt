cmake_minimum_required(VERSION 2.8)

project(ceres_customBundle)

set( Ceres_DIR  "/home/leisure/all_ws/3rdparty/3rd_lib/Ceres-1.14.0/lib/cmake/Ceres")
find_package(Ceres REQUIRED)

find_package(PCL  REQUIRED) 
include_directories(${PCL_INCLUDE_DIRS}) 
link_directories(${PCL_LIBRARY_DIRS}) 
add_definitions(${PCL_DEFINITIONS})


include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
set(CMAKE_BUILD_TYPE "Release") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

include_directories( 
                    ${PROJECT_SOURCE_DIR}/common
                    ${PROJECT_SOURCE_DIR}/common/tools 
                    ${PROJECT_SOURCE_DIR}/common/flags)
include_directories("/usr/include/eigen3/" )

add_library(BALProblem SHARED ${PROJECT_SOURCE_DIR}/common/BALProblem.cpp)
add_library(ParseCmd SHARED ${PROJECT_SOURCE_DIR}/common/flags/command_args.cpp)

add_executable(${PROJECT_NAME} ceresBundle.cpp)
target_link_libraries(${PROJECT_NAME} BALProblem ParseCmd ${CERES_LIBRARIES} ${PCL_LIBRARIES})

#message(${CMAKE_MODULE_PATH})

