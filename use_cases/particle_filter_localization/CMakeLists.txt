cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(gpu_computing_in_robotics_use_cases_particle_filter_localization)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL 1.5 REQUIRED)
find_package(CUDA 7.5 REQUIRED)

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

include_directories(include)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

#check compute capability https://developer.nvidia.com/cuda-gpus
set(CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)

SET(COMMON_SOURCES
	../../use_cases/common/src/data_model.cpp
	../common/src/cudaWrapper.cpp
	../common/src/cudaStructures.cpp
	../common/src/basicFunctions.cpp
	../common/src/cudaBasicFunctions.cu
	../common/src/cudaRegularGridDecompositionFunctions.cu
	../common/src/cudaNearestNeighbourFunctions.cu
)

include_directories(../common/include)

cuda_add_executable(particle_filter_localization
	src/particle_filter_localization_main.cpp
	src/particle_filter.cpp
	${COMMON_SOURCES}
)

target_link_libraries(particle_filter_localization
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
)

cuda_add_executable(particle_filter_localization_fast
	src/particle_filter_localization_fast_main.cpp
	src/particle_filter_fast.cpp
	${COMMON_SOURCES}
)

target_link_libraries(particle_filter_localization_fast
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
)

