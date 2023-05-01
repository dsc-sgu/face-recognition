set(CMAKE_CXX_STANDARD 14)
cmake_minimum_required(VERSION 3.10)
project(raspli_detector)
add_executable(raspli_detector main.cpp detect.h imgproc.h)

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
target_link_libraries(raspli_detector ${OpenCV_LIBRARIES})

set(CMAKE_PREFIX_PATH "libtorch/share/cmake/Torch" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
target_link_libraries(raspli_detector ${TORCH_LIBRARIES})
