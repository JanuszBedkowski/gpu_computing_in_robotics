cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(gpu_computing_in_robotics_lesson_13)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL 1.5 REQUIRED)
find_package(CUDA 7.5 REQUIRED)

message("cuda include dir: ${CUDA_INCLUDE_DIRS}") 
message("cuda libraries: ${CUDA_LIBRARIES}") 
message("cuda cublas libraries: ${CUDA_CUBLAS_LIBRARIES}") 
message("cuda cusolver libraries: ${CUDA_cusolver_LIBRARY}")

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

include_directories(include)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

#check compute capability https://developer.nvidia.com/cuda-gpus
set(CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")


cuda_add_executable(lesson_13
	src/lesson_13_main.cpp
	src/cudaWrapper.cpp
	src/lesson_13.cu
	src/CCUDAAXBSolverWrapper.cpp
)
target_link_libraries(lesson_13
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
	${CUDA_cusparse_LIBRARY}
	${CUDA_cusolver_LIBRARY}
	${CUDA_cublas_LIBRARY}
)

cuda_add_executable(lesson_13_multiple_pose
	src/lesson_13_main_multiple_pose.cpp
	src/cudaWrapper.cpp
	src/lesson_13.cu
	src/CCUDAAXBSolverWrapper.cpp
)
target_link_libraries(lesson_13_multiple_pose
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
	${CUDA_cusparse_LIBRARY}
	${CUDA_cusolver_LIBRARY}
	${CUDA_cublas_LIBRARY}
)



