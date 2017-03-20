cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake")

project(GPU_Computing_in_Robotics_use_cases_robotic_arm_collision_detection) 

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL 1.5 REQUIRED)
find_package(CUDA 7.5 REQUIRED)
find_package(CUDADevice 3.0 REQUIRED)

include_directories(include)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDADEVICE_NVCC_FLAGS} -Xptxas -v")

cuda_add_executable(robotic_arm_collision_detection_main
	src/robotic_arm_collision_detection_main.cpp
	src/cudaWrapper.cpp
	src/lesson_3.cu
)
target_link_libraries(robotic_arm_collision_detection_main
	${OPENGL_LIBRARIES} 
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
)