cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(kcGMLMProject LANGUAGES CXX)

option(WITH_GPU "Enable CUDA code" ON)
option(WITH_DOUBLE_PRECISION "Use double precision for CUDA only" ON)

if(WITH_GPU)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(WARNING "CUDA not found: disabling GPU code.")
    set(WITH_GPU OFF CACHE BOOL "Enable CUDA code" FORCE)
  endif()
endif()

if(WITH_GPU)
  message("Compiling with GPU code.")
  
  enable_language(CUDA)

  if(CMAKE_CUDA_COMPILER_VERSION LESS 11.5)
    message(FATAL_ERROR "Requires CUDA version 11.5 or greater.") # not sure why cmake_minimum_required didn't work
  endif()
  add_definitions(-DUSE_GPU)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message("GPU code disabled.")
endif()

if(WITH_DOUBLE_PRECISION)
  message("Compiling double precision code.")
else()
  message("Compiling single precision code.")
  add_definitions(-DUSE_SINGLE_PRECISION)
endif()

add_subdirectory(src/pyNeuroGMLM/pyGMLMcuda/kcGMLMcuda)
add_subdirectory(src/pyNeuroGMLM/pyGMLMcuda)