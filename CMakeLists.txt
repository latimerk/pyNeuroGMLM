cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(kcGMLMProject LANGUAGES CXX)

option(WITH_GPU "Enable CUDA code" ON)
option(WITH_DOUBLE_PRECISION "Use double precision for CUDA only" ON)



if(DEFINED ENV{GMLM_WITH_GPU})
  message(STATUS "Setting WITH_GPU option from environmental variable (GMLM_WITH_GPU).")
  set(WITH_GPU $ENV{GMLM_WITH_GPU} CACHE BOOL "Enable CUDA code" FORCE)

endif()
if(DEFINED ENV{GMLM_WITH_DOUBLE_PRECISION})
  message(STATUS "Setting WITH_DOUBLE_PRECISION option from environmental variable (GMLM_WITH_DOUBLE_PRECISION).")
  set(WITH_DOUBLE_PRECISION $ENV{GMLM_WITH_DOUBLE_PRECISION} CACHE BOOL "EUse double precision for CUDA only" FORCE)
endif()

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
    message(STATUS "Requires CUDA version 11.5 or greater (found ${CMAKE_CUDA_COMPILER_VERSION}). Disabling GPU code.") # not sure why cmake_minimum_required didn't work
    set(WITH_GPU OFF CACHE BOOL "Enable CUDA code" FORCE)
  else()
    add_definitions(-DUSE_GPU)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # tries to detect a CUDA architecture (I think this should be set if a GPU is installed - this section might be unnecessary)
      include(FindCUDA/select_compute_arch)
      CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
      string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
      string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
      string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
      if( CUDA_ARCH_LIST LESS 61)
        message(STATUS "No GPU found with CUDA architecture >= 6.1 (version indenfitied = ${CUDA_ARCH_LIST}). Using default value.")
        SET(CMAKE_CUDA_ARCHITECTURES 61)
      else()
        message(STATUS "Using detected CUDA architecture from first GPU.")
        SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
      endif()
    endif()
    message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
  endif()
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