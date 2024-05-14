cmake_minimum_required(VERSION 3.0.0)
project(opencv-test VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)

#message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
#message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(opencv-test main.cpp)
target_link_libraries(${PROJECT_NAME} opencv_core opencv_highgui opencv_imgproc)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)