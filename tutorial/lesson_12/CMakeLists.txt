cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake")

project(gpu_computing_in_robotics_lesson_12) 

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL 1.5 REQUIRED)
find_package(CUDA 7.5 REQUIRED)
find_package(CUDADevice 3.0 REQUIRED)

include_directories(include)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDADEVICE_NVCC_FLAGS} -Xptxas -v")

cuda_add_executable(lesson_12
	src/lesson_12_main.cpp
	src/cudaWrapper.cpp
	src/lesson_12.cu
	src/cuda_SVD.cu
	src/data_model.cpp
	src/CCUDAAXBSolverWrapper.cpp
)
target_link_libraries(lesson_12
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
	${CUDA_cusparse_LIBRARY}
	${CUDA_cusolver_LIBRARY}
	${CUDA_cublas_LIBRARY}
)



