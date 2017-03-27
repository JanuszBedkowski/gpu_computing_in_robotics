cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake")

project(fastSLAM)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL 1.5 REQUIRED)
find_package(CUDA 7.5 REQUIRED)
find_package(CUDADevice 3.0 REQUIRED)

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDADEVICE_NVCC_FLAGS} -Xptxas -v")

include_directories( include )	
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

SET(COMMON_SOURCES
	../../use_cases/common/src/data_model.cpp
)

include_directories(../common/include)

cuda_add_executable(fastSLAM
  	src/fastSLAM.cpp
  	src/rgd_particle_filter_CUDAWrapper.cpp
 	src/cuda_algorithms.cu
  	${COMMON_SOURCES}
  )
target_link_libraries(fastSLAM
	${OPENGL_LIBRARIES} 
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
)


