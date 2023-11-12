cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(GPU_Computing_in_Robotics_tutorial_lesson_19)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL 1.5 REQUIRED)
find_package(CUDA 7.5 REQUIRED)
#find_package(CUDADevice 3.0 REQUIRED)
find_package(OpenCV REQUIRED)

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

include_directories(include)
include_directories(${PCL_INCLUDE_DIRS})

#check compute capability https://developer.nvidia.com/cuda-gpus
set(CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDADEVICE_NVCC_FLAGS} -Xptxas -v")

cuda_add_executable(lesson_19
	src/lesson_19_main.cpp
	src/cudaWrapper.cpp
	src/lesson_19.cu
)
target_link_libraries(lesson_19
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
	${OpenCV_LIBS}
)
