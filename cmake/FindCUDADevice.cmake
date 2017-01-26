# Detect CUDA devices in the system and set up proper NVCC flags
#
# Usage:
#    find_package(CUDADevice [X.Y] [REQUIRED])
#
# This module looks for CUDA devices that support at least compute
# capabilities X.Y, or all CUDA capable devices if the version is omitted.
# It will gather the necessary compile flags for the minimum architecture
# to support X.Y (if specified), and for all devices detected in the system.
#
# If no suitable CUDA devices are found and REQUIRED is used, a fatal
# error is raised.
#
# The following variables will be set:
#
# CUDADEVICE_FOUND       - system has CUDA device(s)
# CUDADEVICE_NVCC_FLAGS  - Compile flags for NVCC
#
#
# Copyright 2016 Fraunhofer FKIE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of the Fraunhofer organization nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
unset(CUDADEVICE_FOUND)
unset(CUDADEVICE_NVCC_FLAGS)
set(__cuda_arch_list)
if(NOT CMAKE_CROSSCOMPILING)
    set(__cufile "${PROJECT_BINARY_DIR}/detect_cuda_arch.cu")
    file(WRITE "${__cufile}" ""
        "#include <cstdio>\n"
        "int main()\n"
        "{\n"
        "  int count = 0;\n"
        "  if (cudaGetDeviceCount(&count) != cudaSuccess) return 1;\n"
        "  if (count == 0) return 1;\n"
        "  for (int device = 0; device < count; ++device)\n"
        "  {\n"
        "    cudaDeviceProp prop;\n"
        "    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess)\n"
        "    {\n"
        "      printf(\"%d.%d \", prop.major, prop.minor);\n"
        "    }\n"
        "  }\n"
        "  return 0;\n"
        "}\n"
    )
    execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run" "${__cufile}" "-ccbin" "${CMAKE_CXX_COMPILER}"
        WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
        RESULT_VARIABLE __nvcc_result
        OUTPUT_VARIABLE __nvcc_out
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(__nvcc_result EQUAL 0)
        string(REPLACE " " ";" __nvcc_out "${__nvcc_out}")
        if(CUDADevice_FIND_VERSION)
            foreach(__cuda_arch ${__nvcc_out})
                if(NOT __cuda_arch VERSION_LESS "${CUDADevice_FIND_VERSION}")
                    list(APPEND __cuda_arch_list "${__cuda_arch}")
                endif()
            endforeach()
        else()
            set(__cuda_arch_list "${__nvcc_out}")
        endif()
        if(__cuda_arch_list)
            set(CUDADEVICE_FOUND TRUE)
            if(NOT CUDADevice_FIND_QUIET)
                message(STATUS "Found CUDA device(s) with compute capabilities ${__cuda_arch_list}")
            endif()
        endif()
    endif()
    unset(__nvcc_result)
    unset(__nvcc_out)
    unset(__cufile)
endif()

if(CUDADevice_FIND_VERSION)
    list(APPEND __cuda_arch_list ${CUDADevice_FIND_VERSION})
endif()
list(REMOVE_DUPLICATES __cuda_arch_list)
foreach(__cuda_arch ${__cuda_arch_list})
    if(__cuda_arch STREQUAL "2.1")
        list(APPEND CUDADEVICE_NVCC_FLAGS "-gencode arch=compute_20,code=sm_21")
    else()
        string(REPLACE "." "" __cuda_arch "${__cuda_arch}")
        list(APPEND CUDADEVICE_NVCC_FLAGS "-gencode arch=compute_${__cuda_arch},code=sm_${__cuda_arch}")
    endif()
endforeach()
string(REPLACE ";" " " CUDADEVICE_NVCC_FLAGS "${CUDADEVICE_NVCC_FLAGS}")
if(CUDADEVICE_NVCC_FLAGS AND NOT CUDADevice_FIND_QUIET)
    message(STATUS "CUDA NVCC device flags: ${CUDADEVICE_NVCC_FLAGS}")
endif()
unset(__cuda_arch_list)
unset(__cuda_arch)

if(NOT CUDADEVICE_FOUND AND CUDADevice_FIND_REQUIRED)
    message(FATAL_ERROR "This computer has no CUDA device with the required compute capabilities")
endif()

