cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake")

project(GPU_Computing_in_Robotics_tutorial_lesson_1) 

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL 1.5 REQUIRED)
find_package(CUDA 7.5 REQUIRED)
find_package(CUDADevice 3.0 REQUIRED)

include_directories(include)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDADEVICE_NVCC_FLAGS} -Xptxas -v")

cuda_add_executable(lesson_1
	src/lesson_1_main.cpp
	src/cudaWrapper.cpp
	src/lesson_1.cu
)
target_link_libraries(lesson_1
	${OPENGL_LIBRARIES} 
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
)
