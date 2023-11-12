cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(GPU_Computing_in_Robotics_tutorial_lesson_17)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL 1.5 REQUIRED)
find_package(CUDA 7.5 REQUIRED)

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

include_directories(include)
#include_directories(${PCL_INCLUDE_DIRS})
#add_definitions(${PCL_DEFINITIONS})

#check compute capability https://developer.nvidia.com/cuda-gpus
set(CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)

cuda_add_executable(lesson_17
	src/lesson_17_main.cpp
	src/cudaWrapper.cpp
	src/lesson_17.cu
)
target_link_libraries(lesson_17
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
)
