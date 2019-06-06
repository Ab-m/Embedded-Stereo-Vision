set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

cmake_minimum_required(VERSION 3.0.0)
project(stereoPipeline VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -O3 -Wall")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -lineinfo
    -gencode=arch=compute_62,code=sm_62
    )

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(../libSGM/include)

add_library(TX2PowerMonitor TX2PowerMonitor.cpp)
add_library(imagePairReader imagePairReader.cpp)
add_library(stereoPipeline stereoPipeline.cpp)
target_link_libraries(stereoPipeline ${OpenCV_LIBS})


add_executable(testBench testBench.cpp)
target_link_libraries(testBench TX2PowerMonitor stereoPipeline imagePairReader sgm ${OpenCV_LIBS} benchmark)
