cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake")

project(GPU_Computing_in_Robotics_tutorial_lesson_13)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(CUDA 7.5 REQUIRED)
find_package(CUDADevice 3.0 REQUIRED)

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

include_directories(include)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDADEVICE_NVCC_FLAGS} -Xptxas -v")

cuda_add_executable(lesson_13
	src/lesson_13_main.cpp
	src/cudaWrapper.cpp
	src/lesson_13.cu
	src/CCUDAAXBSolverWrapper.cpp
)
target_link_libraries(lesson_13
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
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
	${CUDA_cusparse_LIBRARY}
	${CUDA_cusolver_LIBRARY}
	${CUDA_cublas_LIBRARY}
)



