cmake_minimum_required(VERSION 3.30)
set(CMAKE_CUDA_ARCHITECTURES 75)
project(Image_processing_pipeline_ CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(Image_processing_pipeline_ main.cu)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O2)

set_target_properties(Image_processing_pipeline_ PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)