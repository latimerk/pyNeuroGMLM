find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(kcGMLMcuda/)
link_directories(${kcGMLMProject_SOURCE_DIR}/obj/)
#link_directories(.)

if(WITH_GPU)
    find_package(CUDAToolkit REQUIRED)
    pybind11_add_module(pyNeuroGMLMcuda MODULE  kcGMLMPython.cpp kcGMLMPython_glm.cpp)
    target_link_libraries(pyNeuroGMLMcuda PRIVATE  kcGMLM  CUDA::cuda_driver  CUDA::cudart  CUDA::cusparse CUDA::cublas)
else()
    pybind11_add_module(pyNeuroGMLMcuda MODULE  kcGMLMPython.cpp kcGMLMPython_glm.cpp)
endif()

install(TARGETS pyNeuroGMLMcuda DESTINATION .)