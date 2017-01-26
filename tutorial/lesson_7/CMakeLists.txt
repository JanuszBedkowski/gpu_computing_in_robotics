cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake")

project(GPU_Computing_in_Robotics_tutorial_lesson_7) 

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL 1.5 REQUIRED)
find_package(CUDA 7.5 REQUIRED)
find_package(CUDADevice 3.0 REQUIRED)

include_directories(include)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDADEVICE_NVCC_FLAGS} -Xptxas -v")

cuda_add_executable(lesson_7
	src/lesson_7_main.cpp
	src/cudaWrapper.cpp
	src/lesson_7.cu
	src/cuda_SVD.cu
)
target_link_libraries(lesson_7
	${OPENGL_LIBRARIES} 
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
)
