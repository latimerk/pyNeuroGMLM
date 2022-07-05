find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(kcGMLMcuda/)
link_directories(${kcGMLMProject_SOURCE_DIR}/obj/)
#link_directories(.)
find_package(CUDAToolkit REQUIRED)

if(WITH_GPU)
    pybind11_add_module(pyNeuroGMLMcuda MODULE  kcGMLMPython.cpp kcGMLMPython_glm.cpp)
    target_link_libraries(pyNeuroGMLMcuda PRIVATE  kcGMLM  CUDA::cuda_driver CUDA::cudadevrt CUDA::cudart  CUDA::cusparse CUDA::cublas)
else()
    pybind11_add_module(pyNeuroGMLMcuda MODULE  kcGMLMPython.cpp kcGMLMPython_glm.cpp)
endif()

# set_property(TARGET pyNeuroGMLMcuda PROPERTY LIBRARY_OUTPUT_DIRECTORY ${kcGMLMProject_SOURCE_DIR}/src/pyNeuroGMLM/)

#if(WITH_GPU)
#    install(TARGETS kcGMLM pyNeuroGMLMcuda DESTINATION .)
#else()
#    install(TARGETS pyNeuroGMLMcuda DESTINATION .)
#endif()
install(TARGETS pyNeuroGMLMcuda DESTINATION .)