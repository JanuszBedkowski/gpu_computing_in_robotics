cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(GPU_Computing_in_Robotics_tutorial_lesson_18)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL 1.5 REQUIRED)
find_package(CUDA 7.5 REQUIRED)
find_package(OpenCV REQUIRED)

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

include_directories(include)
#include_directories(${PCL_INCLUDE_DIRS})
#add_definitions(${PCL_DEFINITIONS})

#check compute capability https://developer.nvidia.com/cuda-gpus
set(CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std c++20")

cuda_add_executable(lesson_18
	src/lesson_18_main.cpp
	src/CUDA.cu
	src/GPUMatching.cpp
	src/BFROSTMatcher.cpp
	
)
target_link_libraries(lesson_18
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
	${OpenCV_LIBS}
)
