cmake_minimum_required(VERSION 3.10)

project(csc367a4 LANGUAGES CXX CUDA)

add_subdirectory(googletest)

set(CMAKE_CUDA_FLAGS "-arch=sm_61")

add_library(a4_kernels
  pgm.cu
  best_cpu.cu
  kernel1.cu
  kernel2.cu
  kernel3.cu
  kernel4.cu
  kernel5.cu
  apply2D.cu
)

add_executable(main
  main.cu
)

target_link_libraries(main
  PUBLIC a4_kernels
)

add_library(pgm
  pgm.cu
)

add_executable(pgm_creator
  pgm_creator.cu
)

target_link_libraries(pgm_creator
  PUBLIC pgm
)

add_executable(test_solution
  tests.cu
)

target_link_libraries(test_solution
  PUBLIC pgm a4_kernels gtest
)
